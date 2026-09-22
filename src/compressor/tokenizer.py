from __future__ import annotations

import logging
import math
import re
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

from src.config import Settings, TokenizerBackend
from src.schemas import ChatMessage

logger = logging.getLogger(__name__)

TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
TOKENS_REPLY_PRIMING = 3

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@runtime_checkable
class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    name = "approx"

    def count(self, text: str) -> int:
        if not text:
            return 0
        words = len(_WORD_RE.findall(text))
        return max(1, math.ceil(max(words * 1.3, len(text) / 4)))


class TiktokenCounter:
    def __init__(self, encoding_name: str) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = f"tiktoken:{encoding_name}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))


class HuggingFaceCounter:
    def __init__(self, tokenizer_name: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.name = f"huggingface:{tokenizer_name}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))


@contextmanager
def socket_deadline(seconds: float) -> Iterator[None]:
    # tiktoken and transformers download their vocabulary with HTTP clients that
    # ship without a timeout, so an unreachable CDN would hang startup forever.
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def build_token_counter(settings: Settings) -> TokenCounter:
    backend: TokenizerBackend = settings.tokenizer_backend
    timeout = settings.tokenizer_init_timeout_seconds
    if backend == "tiktoken":
        try:
            with socket_deadline(timeout):
                return TiktokenCounter(settings.tiktoken_encoding)
        except Exception as exc:
            logger.warning("tiktoken backend unavailable (%s); falling back to approximation", exc)
    elif backend == "huggingface":
        try:
            with socket_deadline(timeout):
                return HuggingFaceCounter(settings.hf_tokenizer_name)
        except Exception as exc:
            logger.warning(
                "huggingface backend unavailable (%s); falling back to approximation", exc
            )
    return ApproxTokenCounter()


class MessageTokenizer:
    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    @property
    def name(self) -> str:
        return self._counter.name

    def count_text(self, text: str) -> int:
        return self._counter.count(text)

    def count_message(self, message: ChatMessage) -> int:
        total = TOKENS_PER_MESSAGE + self._counter.count(message.role)
        total += self._counter.count(message.text_content)
        if message.name:
            total += TOKENS_PER_NAME + self._counter.count(message.name)
        if message.tool_calls:
            for call in message.tool_calls:
                function = call.get("function") or {}
                total += self._counter.count(str(function.get("name", "")))
                total += self._counter.count(str(function.get("arguments", "")))
        return total

    def count_messages(self, messages: list[ChatMessage]) -> int:
        if not messages:
            return 0
        return sum(self.count_message(m) for m in messages) + TOKENS_REPLY_PRIMING

    def truncate_text(self, text: str, max_tokens: int, *, marker: str = " [...]") -> str:
        if max_tokens <= 0:
            return ""
        if self.count_text(text) <= max_tokens:
            return text
        budget = max(1, max_tokens - self.count_text(marker))

        low, high = 0, len(text)
        best = ""
        for _ in range(12):
            if low >= high:
                break
            mid = (low + high + 1) // 2
            candidate = text[:mid]
            if self.count_text(candidate) <= budget:
                best = candidate
                low = mid
            else:
                high = mid - 1
        return (best.rstrip() + marker) if best else marker.strip()


def build_message_tokenizer(settings: Settings) -> MessageTokenizer:
    return MessageTokenizer(build_token_counter(settings))
