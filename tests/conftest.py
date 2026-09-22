from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from src.api.deps import AppState
from src.app import create_app
from src.compressor.tokenizer import ApproxTokenCounter, MessageTokenizer
from src.config import Settings
from src.db.database import Database
from src.db.repository import MetricsRepository
from src.errors import ProviderUnavailableError
from src.providers.base import SSE_DONE, ChatProvider, sse_event
from src.providers.registry import ProviderRegistry
from src.schemas import ChatCompletionRequest, ChatMessage


class FakeProvider(ChatProvider):
    def __init__(self, name: str, *, settings: Settings, reply: str = "pong") -> None:
        super().__init__(cast(httpx.AsyncClient, None), settings)
        self.name = name  # type: ignore[misc]
        self.reply = reply
        self.calls: list[ChatCompletionRequest] = []
        self.available = True
        self.summary_text = "- condensed notes from the earlier conversation"

    def go_offline(self) -> None:
        self.available = False

    @property
    def last_call(self) -> ChatCompletionRequest:
        assert self.calls, "provider was never called"
        return self.calls[-1]

    @property
    def chat_calls(self) -> list[ChatCompletionRequest]:
        return [c for c in self.calls if not _is_summary_call(c)]

    @property
    def summary_calls(self) -> list[ChatCompletionRequest]:
        return [c for c in self.calls if _is_summary_call(c)]

    def _guard(self) -> None:
        if not self.available:
            raise ProviderUnavailableError(
                f"Cannot reach {self.name}. Is the service running?", provider=self.name
            )

    async def chat(
        self, request: ChatCompletionRequest, *, timeout: float | None = None
    ) -> dict[str, Any]:
        self._guard()
        self.calls.append(request)
        content = self.summary_text if _is_summary_call(request) else self.reply
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

    async def stream(
        self, request: ChatCompletionRequest, *, timeout: float | None = None
    ) -> AsyncIterator[bytes]:
        self._guard()
        self.calls.append(request)
        for index, token in enumerate(self.reply.split()):
            yield sse_event(
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": (
                                {"role": "assistant", "content": token}
                                if index == 0
                                else {"content": f" {token}"}
                            ),
                            "finish_reason": None,
                        }
                    ],
                }
            )
        yield SSE_DONE

    async def list_models(self) -> list[dict[str, Any]]:
        self._guard()
        return [{"id": f"{self.name}/fake-model", "object": "model", "owned_by": self.name}]

    async def health(self) -> bool:
        return self.available


def _is_summary_call(request: ChatCompletionRequest) -> bool:
    first = request.messages[0]
    return first.role == "system" and "context compression engine" in first.text_content


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        DB_PATH=tmp_path / "metrics.db",
        MAX_CONTEXT_TOKENS=600,
        RESERVE_OUTPUT_TOKENS=100,
        COMPRESSION_TARGET_RATIO=0.5,
        KEEP_LAST_MESSAGES=4,
        TOKENIZER_BACKEND="approx",
        DEFAULT_PROVIDER="ollama",
        SUMMARY_RETRY_BACKOFF_SECONDS=0,
    )


@pytest.fixture
def tokenizer() -> MessageTokenizer:
    return MessageTokenizer(ApproxTokenCounter())


@pytest.fixture
def fake_ollama(settings: Settings) -> FakeProvider:
    return FakeProvider("ollama", settings=settings, reply="hello from ollama")


@pytest.fixture
def fake_openrouter(settings: Settings) -> FakeProvider:
    return FakeProvider("openrouter", settings=settings, reply="hello from openrouter")


@pytest.fixture
async def app_state(
    settings: Settings,
    tokenizer: MessageTokenizer,
    fake_ollama: FakeProvider,
    fake_openrouter: FakeProvider,
) -> AsyncIterator[AppState]:
    database = Database(settings.db_path, enabled=settings.db_enabled)
    await database.connect()
    client = httpx.AsyncClient()
    registry = ProviderRegistry(client, settings)
    registry.register("ollama", fake_ollama)
    registry.register("openrouter", fake_openrouter)
    state = AppState(
        settings=settings,
        http_client=client,
        registry=registry,
        tokenizer=tokenizer,
        database=database,
        metrics=MetricsRepository(database),
    )
    try:
        yield state
    finally:
        await database.close()
        await client.aclose()


@pytest.fixture
def app(settings: Settings, app_state: AppState) -> FastAPI:
    application = create_app(settings)
    application.state.app_state = app_state
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as http:
        yield http


def make_conversation(turns: int = 20, words: int = 40) -> list[ChatMessage]:
    filler = "lorem ipsum dolor sit amet consectetur "
    messages = [ChatMessage(role="system", content="You are a helpful assistant.")]
    for index in range(turns):
        messages.append(
            ChatMessage(role="user", content=f"Question {index}: " + filler * (words // 5))
        )
        messages.append(
            ChatMessage(role="assistant", content=f"Answer {index}: " + filler * (words // 5))
        )
    messages.append(ChatMessage(role="user", content="So what did we decide in the end?"))
    return messages


@pytest.fixture
def long_conversation() -> list[ChatMessage]:
    return make_conversation()


@pytest.fixture(autouse=True)
def _no_env_leakage(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for variable in ("OPENROUTER_API_KEY", "DEFAULT_PROVIDER", "OLLAMA_HOST"):
        monkeypatch.delenv(variable, raising=False)
    yield
