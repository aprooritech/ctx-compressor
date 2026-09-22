from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ContentPart = dict[str, Any]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    @field_validator("role")
    @classmethod
    def _non_empty_role(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message role must not be empty")
        return value

    @property
    def text_content(self) -> str:
        content = self.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for part in content:
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif part.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(parts)

    @property
    def is_pinned(self) -> bool:
        return self.role in ("system", "developer")

    def with_content(self, content: str) -> ChatMessage:
        return self.model_copy(update={"content": content})


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str = ""
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    n: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    user: str | None = None

    @property
    def effective_max_tokens(self) -> int | None:
        return self.max_tokens or self.max_completion_tokens

    @property
    def stop_sequences(self) -> list[str] | None:
        if self.stop is None:
            return None
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    def upstream_payload(self, *, model: str, messages: list[ChatMessage]) -> dict[str, Any]:
        payload = self.model_dump(exclude_none=True, exclude={"model", "messages"})
        payload["model"] = model
        payload["messages"] = [m.model_dump(exclude_none=True) for m in messages]
        return payload


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: str | None = None


class ProviderUsageStats(BaseModel):
    provider: str
    requests: int
    errors: int
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    avg_saved_ratio: float
    avg_latency_ms: float


class ModelUsageStats(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    provider: str
    model: str
    requests: int
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    avg_saved_ratio: float


class StatsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    window_hours: float | None = None
    generated_at: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    compressed_requests: int = 0
    sessions: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_saved_tokens: int = 0
    overall_saved_ratio: float = 0.0
    avg_saved_ratio: float = 0.0
    avg_total_latency_ms: float = 0.0
    p95_total_latency_ms: float = 0.0
    avg_upstream_latency_ms: float = 0.0
    avg_compression_latency_ms: float = 0.0
    by_provider: list[ProviderUsageStats] = Field(default_factory=list)
    by_model: list[ModelUsageStats] = Field(default_factory=list)


class RequestLogItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    session_id: str
    created_at: str
    provider: str
    model: str
    strategy: str
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    saved_ratio: float
    status: str
    error_type: str | None = None
    total_latency_ms: float
    upstream_latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    tokenizer: str
    compression_enabled: bool
    providers: dict[str, bool]
    database: bool
