from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import httpx

from src.config import ProviderName, Settings
from src.errors import ProviderTimeoutError, ProviderUnavailableError, UpstreamError
from src.schemas import ChatCompletionRequest

logger = logging.getLogger(__name__)

SSE_DONE = b"data: [DONE]\n\n"


class ChatProvider(ABC):
    name: ClassVar[ProviderName]

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    @property
    def default_model(self) -> str:
        return self._settings.default_model_for(self.name)

    @abstractmethod
    async def chat(
        self, request: ChatCompletionRequest, *, timeout: float | None = None
    ) -> dict[str, Any]: ...

    @abstractmethod
    def stream(
        self, request: ChatCompletionRequest, *, timeout: float | None = None
    ) -> AsyncIterator[bytes]: ...

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def health(self) -> bool: ...

    def _timeout(self, timeout: float | None) -> httpx.Timeout:
        return httpx.Timeout(
            timeout or self._settings.request_timeout_seconds,
            connect=self._settings.connect_timeout_seconds,
        )

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        if not response.is_closed:
            await response.aread()
        body: Any
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = response.text[:2000]
        message = _extract_error_message(body) or (
            f"{self.name} returned HTTP {response.status_code}"
        )
        raise UpstreamError(
            message,
            upstream_status=response.status_code,
            provider=self.name,
            body=body,
        )


@contextmanager
def translate_transport_errors(provider: str, endpoint: str) -> Iterator[None]:
    try:
        yield
    except httpx.TimeoutException as exc:
        raise ProviderTimeoutError(
            f"{provider} did not respond in time ({endpoint}).",
            provider=provider,
        ) from exc
    except httpx.ConnectError as exc:
        raise ProviderUnavailableError(
            f"Cannot reach {provider} at {endpoint}. Is the service running?",
            provider=provider,
            details={"reason": str(exc)},
        ) from exc
    except httpx.HTTPError as exc:
        raise ProviderUnavailableError(
            f"Transport error while talking to {provider} ({endpoint}): {exc}",
            provider=provider,
        ) from exc


def _extract_error_message(body: Any) -> str | None:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        if isinstance(error, str):
            return error
        message = body.get("message")
        if isinstance(message, str):
            return message
    if isinstance(body, str) and body.strip():
        return body.strip()[:500]
    return None


def sse_event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
