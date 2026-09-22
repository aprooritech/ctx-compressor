from __future__ import annotations

from src.compressor.engine import ContextCompressor
from src.compressor.summarizer import SummarizationUnavailableError
from src.compressor.tokenizer import MessageTokenizer
from src.config import Settings
from src.schemas import ChatMessage
from tests.conftest import make_conversation

SUMMARY_MARKER = "[compressed context]"


class RecordingSummarizer:
    def __init__(self, text: str = "- user asked about X, assistant decided Y") -> None:
        self.text = text
        self.calls: list[tuple[str, int]] = []

    async def summarize(self, text: str, max_tokens: int) -> str:
        self.calls.append((text, max_tokens))
        return self.text


class BrokenSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    async def summarize(self, text: str, max_tokens: int) -> str:
        self.calls += 1
        raise SummarizationUnavailableError("ollama is not running")


async def test_short_conversation_is_passed_through_untouched(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hi!"),
    ]
    compressor = ContextCompressor(tokenizer, settings, RecordingSummarizer())

    result = await compressor.compress(messages)

    assert result.strategy_applied == "none"
    assert result.was_compressed is False
    assert result.messages == messages
    assert result.saved_tokens == 0
    assert result.saved_ratio == 0.0


async def test_summarisation_preserves_system_prompt_and_recent_messages(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    messages = make_conversation()
    summarizer = RecordingSummarizer()
    compressor = ContextCompressor(tokenizer, settings, summarizer)

    result = await compressor.compress(messages)

    assert "summarize" in result.strategy_applied
    assert summarizer.calls, "the summariser should have been invoked"
    assert result.messages[0].role == "system"
    assert result.messages[0].text_content == "You are a helpful assistant."
    assert any(SUMMARY_MARKER in m.text_content for m in result.messages)
    assert result.summarized_messages > 0
    assert result.messages[-1].text_content == messages[-1].text_content
    assert result.compressed_tokens < result.original_tokens
    assert result.compressed_tokens <= settings.compression_trigger_tokens
    assert 0.0 < result.saved_ratio < 1.0
    assert result.saved_tokens == result.original_tokens - result.compressed_tokens


async def test_summarisation_receives_only_the_long_term_memory(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    messages = make_conversation()
    summarizer = RecordingSummarizer()

    await ContextCompressor(tokenizer, settings, summarizer).compress(messages)

    condensed = "\n".join(text for text, _ in summarizer.calls)
    assert "So what did we decide in the end?" not in condensed
    assert "Question 0" in condensed


async def test_failing_summariser_falls_back_to_trimming(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    messages = make_conversation()
    summarizer = BrokenSummarizer()
    compressor = ContextCompressor(tokenizer, settings, summarizer)

    result = await compressor.compress(messages)

    assert summarizer.calls > 0
    assert result.strategy_applied == "trim"
    assert any("falling back to trimming" in note for note in result.notes)
    assert result.compressed_tokens <= settings.compression_trigger_tokens
    assert result.messages[0].role == "system"
    assert result.messages[-1].text_content == messages[-1].text_content


async def test_trim_strategy_respects_budget_and_keeps_the_latest_message(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    messages = make_conversation()
    compressor = ContextCompressor(tokenizer, settings, RecordingSummarizer())

    result = await compressor.compress(messages, strategy="trim")

    assert result.strategy_applied == "trim"
    assert result.compressed_tokens <= settings.compression_trigger_tokens
    assert len(result.messages) < len(messages)
    assert result.dropped_messages + result.trimmed_messages > 0
    assert result.messages[-1].text_content == messages[-1].text_content


async def test_disabled_compression_is_a_noop(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    disabled = settings.model_copy(update={"compression_enabled": False})
    messages = make_conversation()

    result = await ContextCompressor(tokenizer, disabled, RecordingSummarizer()).compress(messages)

    assert result.strategy_applied == "none"
    assert result.messages == messages


async def test_tool_results_stay_attached_to_their_call(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    messages = make_conversation(turns=15)
    messages.append(
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
        )
    )
    messages.append(ChatMessage(role="tool", tool_call_id="call_1", content="search result"))
    messages.append(ChatMessage(role="user", content="thanks, summarise that"))

    result = await ContextCompressor(tokenizer, settings, RecordingSummarizer()).compress(messages)

    roles = [m.role for m in result.messages]
    for index, role in enumerate(roles):
        if role == "tool":
            assert "assistant" in roles[:index], "a tool result lost its assistant call"


async def test_no_summariser_configured_degrades_to_trimming(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    result = await ContextCompressor(tokenizer, settings, None).compress(make_conversation())

    assert result.strategy_applied == "trim"
    assert any("no summariser configured" in note for note in result.notes)
    assert result.compressed_tokens <= settings.compression_trigger_tokens


class FlakySummarizer:
    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls = 0

    async def summarize(self, text: str, max_tokens: int) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise SummarizationUnavailableError("429 rate limited")
        return "- condensed notes"


class PartiallyFailingSummarizer:
    def __init__(self, poison: str) -> None:
        self.poison = poison
        self.failed = 0
        self.succeeded = 0

    async def summarize(self, text: str, max_tokens: int) -> str:
        if self.poison in text:
            self.failed += 1
            raise SummarizationUnavailableError("upstream pool exhausted")
        self.succeeded += 1
        return "- condensed notes about this part of the conversation"


async def test_transient_summariser_failure_is_retried(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    summarizer = FlakySummarizer(failures=1)

    result = await ContextCompressor(tokenizer, settings, summarizer).compress(make_conversation())

    assert summarizer.calls >= 2, "the failed chunk should have been retried"
    assert "summarize" in result.strategy_applied
    assert any(SUMMARY_MARKER in m.text_content for m in result.messages)


async def test_a_failing_chunk_does_not_discard_the_whole_summary(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    tight = settings.model_copy(update={"summary_chunk_tokens": 200})
    summarizer = PartiallyFailingSummarizer(poison="Question 0:")

    result = await ContextCompressor(tokenizer, tight, summarizer).compress(make_conversation())

    assert summarizer.failed > 0 and summarizer.succeeded > 0
    assert "summarize" in result.strategy_applied
    assert any("could not be summarised" in note for note in result.notes)
    assert result.summarized_messages > 0
    assert result.messages[0].role == "system"
    assert result.messages[-1].text_content.endswith("in the end?")


async def test_all_chunks_failing_still_degrades_to_trimming(
    settings: Settings, tokenizer: MessageTokenizer
) -> None:
    summarizer = PartiallyFailingSummarizer(poison="Question")  # matches every chunk

    result = await ContextCompressor(tokenizer, settings, summarizer).compress(make_conversation())

    assert result.strategy_applied == "trim"
    assert any("falling back to trimming" in note for note in result.notes)
    assert result.compressed_tokens <= settings.compression_trigger_tokens
