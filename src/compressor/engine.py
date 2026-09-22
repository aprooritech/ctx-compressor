from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import perf_counter

from src.compressor.summarizer import SummarizationUnavailableError, Summarizer
from src.compressor.tokenizer import MessageTokenizer
from src.config import CompressionStrategy, Settings
from src.schemas import ChatMessage

logger = logging.getLogger(__name__)

MIN_SUMMARY_BUDGET = 96
MIN_LAST_MESSAGE_TOKENS = 64

SUMMARY_HEADER = (
    "[compressed context] Condensed notes covering {count} earlier messages "
    "that were removed to fit the context window:\n"
)


@dataclass(slots=True)
class SummaryOutcome:
    text: str
    calls: int
    total_chunks: int
    failed_chunks: int


@dataclass(slots=True)
class CompressionResult:
    messages: list[ChatMessage]
    original_tokens: int
    compressed_tokens: int
    strategy_applied: str
    budget_tokens: int
    original_message_count: int
    compressed_message_count: int
    summarized_messages: int = 0
    trimmed_messages: int = 0
    dropped_messages: int = 0
    summary_calls: int = 0
    duration_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def saved_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return round(self.saved_tokens / self.original_tokens, 6)

    @property
    def was_compressed(self) -> bool:
        return self.strategy_applied != "none"


class ContextCompressor:
    def __init__(
        self,
        tokenizer: MessageTokenizer,
        settings: Settings,
        summarizer: Summarizer | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._settings = settings
        self._summarizer = summarizer

    async def compress(
        self,
        messages: list[ChatMessage],
        *,
        trigger_tokens: int | None = None,
        target_tokens: int | None = None,
        strategy: CompressionStrategy | None = None,
    ) -> CompressionResult:
        started = perf_counter()
        settings = self._settings
        active_strategy: CompressionStrategy = strategy or settings.compression_strategy
        trigger = trigger_tokens or settings.compression_trigger_tokens
        target = min(target_tokens or settings.compression_target_tokens, trigger)

        original_tokens = self._tokenizer.count_messages(messages)
        untouched = CompressionResult(
            messages=list(messages),
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            strategy_applied="none",
            budget_tokens=target,
            original_message_count=len(messages),
            compressed_message_count=len(messages),
            duration_ms=(perf_counter() - started) * 1000,
        )
        if (
            not settings.compression_enabled
            or active_strategy == "none"
            or not messages
            or original_tokens <= trigger
        ):
            return untouched

        pinned, head, tail = self._partition(messages)
        notes: list[str] = []
        applied: list[str] = []
        summarized = 0
        trimmed = 0
        dropped = 0
        summary_calls = 0

        # Everything in prefix survives trimming: system prompts, and the
        # summary once it exists.
        prefix: list[ChatMessage] = list(pinned)
        body: list[ChatMessage] = list(head) + list(tail)

        if active_strategy in ("summarize", "hybrid") and head:
            summary_budget = self._summary_budget(pinned, tail, target)
            if self._summarizer is None:
                notes.append("summarisation skipped: no summariser configured")
            elif summary_budget is None:
                notes.append("summarisation skipped: system prompt already fills the budget")
            else:
                if summary_budget < MIN_SUMMARY_BUDGET:
                    notes.append(
                        "short-term memory fills the budget; summarising into the "
                        "minimum budget and trimming recent messages instead"
                    )
                try:
                    outcome = await self._summarize_messages(
                        head, max(MIN_SUMMARY_BUDGET, summary_budget)
                    )
                except SummarizationUnavailableError as exc:
                    notes.append(f"summarisation failed, falling back to trimming: {exc}")
                    logger.warning("summarisation unavailable: %s", exc)
                except Exception as exc:
                    notes.append(f"summarisation error, falling back to trimming: {exc}")
                    logger.exception("unexpected summarisation error")
                else:
                    summary_calls = outcome.calls
                    if outcome.failed_chunks:
                        notes.append(
                            f"{outcome.failed_chunks} of {outcome.total_chunks} history "
                            "chunks could not be summarised; that part was dropped"
                        )
                    prefix = [
                        *pinned,
                        ChatMessage(
                            role=settings.summary_message_role,
                            content=SUMMARY_HEADER.format(count=len(head)) + outcome.text,
                        ),
                    ]
                    body = list(tail)
                    summarized = len(head)
                    applied.append("summarize")

        current_tokens = self._tokenizer.count_messages(prefix + body)
        allow_trim = active_strategy in ("trim", "hybrid")
        # Staying above the hard limit breaks the upstream call, so trim even
        # when the configured strategy is summarisation-only.
        emergency = current_tokens > trigger
        if current_tokens > target and (allow_trim or emergency):
            if emergency and not allow_trim:
                notes.append("emergency trimming applied: still above the hard context limit")
            prefix, body, trimmed, dropped, trim_notes = self._trim_to_budget(prefix, body, target)
            notes.extend(trim_notes)
            if trimmed or dropped:
                applied.append("trim")

        if not applied:
            notes.append("no compression step could be applied")
            untouched.notes = notes
            untouched.duration_ms = (perf_counter() - started) * 1000
            return untouched

        compressed = prefix + body
        return CompressionResult(
            messages=compressed,
            original_tokens=original_tokens,
            compressed_tokens=self._tokenizer.count_messages(compressed),
            strategy_applied="+".join(applied),
            budget_tokens=target,
            original_message_count=len(messages),
            compressed_message_count=len(compressed),
            summarized_messages=summarized,
            trimmed_messages=trimmed,
            dropped_messages=dropped,
            summary_calls=summary_calls,
            duration_ms=(perf_counter() - started) * 1000,
            notes=notes,
        )

    def _partition(
        self, messages: list[ChatMessage]
    ) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage]]:
        pinned = [m for m in messages if m.is_pinned]
        conversation = [m for m in messages if not m.is_pinned]
        if not conversation:
            return pinned, [], []

        keep = max(min(self._settings.keep_last_messages, len(conversation)), 1)
        split = len(conversation) - keep
        # A tool result must not become the first kept message.
        while split > 0 and conversation[split].role == "tool":
            split -= 1
        return pinned, conversation[:split], conversation[split:]

    def _summary_budget(
        self, pinned: list[ChatMessage], tail: list[ChatMessage], target: int
    ) -> int | None:
        # None means there is no room for a summary at all; a value below
        # MIN_SUMMARY_BUDGET means the recent messages have to be trimmed too.
        pinned_tokens = self._tokenizer.count_messages(pinned)
        if pinned_tokens + MIN_SUMMARY_BUDGET > target:
            return None
        available = target - self._tokenizer.count_messages(pinned + tail)
        return min(self._settings.summary_max_tokens, max(0, available))

    async def _summarize_messages(self, messages: list[ChatMessage], budget: int) -> SummaryOutcome:
        assert self._summarizer is not None
        settings = self._settings
        chunks = self._chunk_texts(
            [self._render_message(m) for m in messages], settings.summary_chunk_tokens
        )
        total_chunks = len(chunks)
        calls = 0
        failed = 0
        summary = ""
        previous_tokens: int | None = None

        for _ in range(settings.summary_max_depth):
            per_call = max(
                MIN_SUMMARY_BUDGET,
                min(settings.summary_max_tokens, budget // max(1, len(chunks))),
            )
            summaries: list[str] = []
            for chunk in chunks:
                try:
                    text = await self._summarize_chunk(chunk, per_call)
                except SummarizationUnavailableError as exc:
                    failed += 1
                    logger.warning("chunk summarisation failed: %s", exc)
                    continue
                calls += 1
                summaries.append(text)
            if not summaries:
                raise SummarizationUnavailableError(f"all {len(chunks)} chunks failed to summarise")

            summary = "\n".join(s.strip() for s in summaries if s.strip())
            tokens = self._tokenizer.count_text(summary)
            if tokens <= budget:
                break
            if previous_tokens is not None and tokens >= previous_tokens:
                break
            previous_tokens = tokens
            chunks = self._chunk_texts([summary], settings.summary_chunk_tokens)

        if not summary.strip():
            raise SummarizationUnavailableError("summariser produced no usable output")
        if self._tokenizer.count_text(summary) > budget:
            summary = self._tokenizer.truncate_text(summary, budget)
        return SummaryOutcome(
            text=summary, calls=calls, total_chunks=total_chunks, failed_chunks=failed
        )

    async def _summarize_chunk(self, chunk: str, budget: int) -> str:
        assert self._summarizer is not None
        attempts = self._settings.summary_retry_attempts
        backoff = self._settings.summary_retry_backoff_seconds
        last_error: SummarizationUnavailableError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._summarizer.summarize(chunk, budget)
            except SummarizationUnavailableError as exc:
                last_error = exc
                if attempt < attempts and backoff:
                    await asyncio.sleep(backoff * attempt)
        assert last_error is not None
        raise last_error

    def _render_message(self, message: ChatMessage) -> str:
        text = message.text_content
        if message.tool_calls:
            names = ", ".join(
                str((call.get("function") or {}).get("name", "tool")) for call in message.tool_calls
            )
            text = f"{text}\n[tool calls: {names}]".strip()
        label = message.role.upper()
        if message.name:
            label = f"{label} ({message.name})"
        return f"{label}: {text}".strip()

    def _chunk_texts(self, blocks: list[str], max_tokens: int) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for block in blocks:
            block_tokens = self._tokenizer.count_text(block)
            if block_tokens > max_tokens:
                if current:
                    chunks.append("\n\n".join(current))
                    current, current_tokens = [], 0
                chunks.extend(self._split_text(block, max_tokens))
                continue
            if current and current_tokens + block_tokens > max_tokens:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            current.append(block)
            current_tokens += block_tokens
        if current:
            chunks.append("\n\n".join(current))
        return chunks or [""]

    def _split_text(self, text: str, max_tokens: int) -> list[str]:
        total = self._tokenizer.count_text(text)
        if total <= max_tokens:
            return [text]
        chars_per_token = max(1.0, len(text) / max(1, total))
        window = max(200, int(max_tokens * chars_per_token * 0.9))
        pieces: list[str] = []
        position = 0
        while position < len(text):
            piece = text[position : position + window]
            if position + window < len(text):
                newline = piece.rfind("\n")
                if newline > window // 2:
                    piece = piece[:newline]
            pieces.append(piece)
            position += max(1, len(piece))
        return pieces

    def _trim_to_budget(
        self, prefix: list[ChatMessage], body: list[ChatMessage], target: int
    ) -> tuple[list[ChatMessage], list[ChatMessage], int, int, list[str]]:
        notes: list[str] = []
        prefix = list(prefix)
        body = list(body)
        trimmed = 0
        dropped = 0
        cap = self._settings.trim_message_max_tokens

        def total() -> int:
            return self._tokenizer.count_messages(prefix + body)

        for index in range(max(0, len(body) - 1)):
            if total() <= target:
                break
            message = body[index]
            if self._tokenizer.count_message(message) <= cap:
                continue
            body[index] = message.with_content(
                self._tokenizer.truncate_text(message.text_content, cap)
            )
            trimmed += 1

        while total() > target and len(body) > 1:
            body.pop(0)
            dropped += 1 + self._drop_orphan_tool_messages(body)

        if total() > target and body:
            remaining = target - self._tokenizer.count_messages(prefix) - 8
            if remaining >= MIN_LAST_MESSAGE_TOKENS:
                last = body[-1]
                body[-1] = last.with_content(
                    self._tokenizer.truncate_text(last.text_content, remaining)
                )
                trimmed += 1
            else:
                notes.append(
                    "system prompt alone exceeds the token budget; "
                    "the latest message was kept unmodified"
                )

        if total() > target and prefix:
            body_tokens = self._tokenizer.count_messages(body)
            per_prefix = max(MIN_SUMMARY_BUDGET, (target - body_tokens) // max(1, len(prefix)))
            for index, message in enumerate(prefix):
                if self._tokenizer.count_message(message) <= per_prefix:
                    continue
                prefix[index] = message.with_content(
                    self._tokenizer.truncate_text(message.text_content, per_prefix)
                )
                trimmed += 1
            notes.append("system prompt/summary had to be truncated to fit the budget")

        return prefix, body, trimmed, dropped, notes

    @staticmethod
    def _drop_orphan_tool_messages(body: list[ChatMessage]) -> int:
        removed = 0
        while len(body) > 1 and body[0].role == "tool":
            body.pop(0)
            removed += 1
        return removed
