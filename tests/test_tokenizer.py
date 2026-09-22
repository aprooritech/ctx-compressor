from __future__ import annotations

import socket

import pytest

from src.compressor.tokenizer import (
    TOKENS_PER_MESSAGE,
    TOKENS_REPLY_PRIMING,
    ApproxTokenCounter,
    MessageTokenizer,
    build_message_tokenizer,
    build_token_counter,
    socket_deadline,
)
from src.config import Settings
from src.schemas import ChatMessage


def test_approx_counter_is_monotonic_and_handles_empty_text() -> None:
    counter = ApproxTokenCounter()
    assert counter.count("") == 0
    short = counter.count("hello world")
    longer = counter.count("hello world " * 10)
    assert 0 < short < longer


def test_count_messages_includes_framing_overhead(tokenizer: MessageTokenizer) -> None:
    message = ChatMessage(role="user", content="hello world")
    content_tokens = tokenizer.count_text("hello world")

    single = tokenizer.count_message(message)
    conversation = tokenizer.count_messages([message])

    assert single > content_tokens
    assert conversation == single + TOKENS_REPLY_PRIMING
    assert single >= content_tokens + TOKENS_PER_MESSAGE


def test_count_messages_is_additive(tokenizer: MessageTokenizer) -> None:
    first = ChatMessage(role="user", content="one")
    second = ChatMessage(role="assistant", content="two")
    total = tokenizer.count_messages([first, second])
    assert total == (
        tokenizer.count_message(first) + tokenizer.count_message(second) + TOKENS_REPLY_PRIMING
    )


def test_multimodal_content_is_flattened_for_counting(tokenizer: MessageTokenizer) -> None:
    message = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    )
    assert "describe this" in message.text_content
    assert tokenizer.count_message(message) > 0


def test_unavailable_backend_falls_back_to_approximation() -> None:
    settings = Settings(
        _env_file=None, TOKENIZER_BACKEND="tiktoken", TIKTOKEN_ENCODING="definitely-not-an-encoding"
    )
    counter = build_token_counter(settings)
    assert isinstance(counter, ApproxTokenCounter)
    assert counter.count("hello") > 0


def test_tiktoken_backend_is_used_when_available() -> None:
    settings = Settings(
        _env_file=None, TOKENIZER_BACKEND="tiktoken", TOKENIZER_INIT_TIMEOUT_SECONDS=2
    )
    tokenizer = build_message_tokenizer(settings)
    if not tokenizer.name.startswith("tiktoken"):
        pytest.skip("tiktoken encoding not available in this environment")
    assert tokenizer.count_text("hello world") == 2


def test_truncate_text_respects_the_budget(tokenizer: MessageTokenizer) -> None:
    text = "word " * 500
    budget = 40

    truncated = tokenizer.truncate_text(text, budget)

    assert tokenizer.count_text(truncated) <= budget
    assert truncated.endswith("[...]")
    assert len(truncated) < len(text)


def test_truncate_text_keeps_short_text_untouched(tokenizer: MessageTokenizer) -> None:
    assert tokenizer.truncate_text("short", 100) == "short"


def test_socket_deadline_applies_and_restores_the_default() -> None:
    previous = socket.getdefaulttimeout()

    with socket_deadline(1.5):
        assert socket.getdefaulttimeout() == 1.5

    assert socket.getdefaulttimeout() == previous


def test_socket_deadline_restores_the_default_on_error() -> None:
    previous = socket.getdefaulttimeout()

    with pytest.raises(ValueError), socket_deadline(1.5):
        raise ValueError("boom")

    assert socket.getdefaulttimeout() == previous


def test_stalled_download_degrades_to_approximation(monkeypatch: pytest.MonkeyPatch) -> None:

    def explode(encoding_name: str) -> None:
        raise TimeoutError("the socket deadline fired")

    monkeypatch.setattr("src.compressor.tokenizer.TiktokenCounter", explode)
    settings = Settings(_env_file=None, TOKENIZER_BACKEND="tiktoken")

    counter = build_token_counter(settings)

    assert isinstance(counter, ApproxTokenCounter)
