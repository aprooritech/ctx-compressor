from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.errors import ProxyError
from src.schemas import ChatCompletionRequest, ChatMessage

if TYPE_CHECKING:
    from src.providers.base import ChatProvider

logger = logging.getLogger(__name__)


class SummarizationUnavailableError(RuntimeError):
    pass


@runtime_checkable
class Summarizer(Protocol):
    async def summarize(self, text: str, max_tokens: int) -> str: ...


class NullSummarizer:
    async def summarize(self, text: str, max_tokens: int) -> str:
        raise SummarizationUnavailableError("no summarisation backend configured")


class LLMSummarizer:
    def __init__(
        self,
        provider: ChatProvider,
        model: str,
        *,
        prompt: str,
        timeout: float | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._prompt = prompt
        self._timeout = timeout

    async def summarize(self, text: str, max_tokens: int) -> str:
        request = ChatCompletionRequest(
            model=self._model,
            messages=[
                ChatMessage(role="system", content=self._prompt),
                ChatMessage(role="user", content=text),
            ],
            stream=False,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        try:
            response = await self._provider.chat(request, timeout=self._timeout)
        except ProxyError as exc:
            raise SummarizationUnavailableError(
                f"summarisation via {self._provider.name} failed: {exc.message}"
            ) from exc
        except Exception as exc:
            raise SummarizationUnavailableError(
                f"summarisation via {self._provider.name} failed: {exc}"
            ) from exc

        summary = _extract_text(response)
        if not summary.strip():
            raise SummarizationUnavailableError("summariser returned an empty response")
        return summary.strip()


def _extract_text(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "\n".join(str(p) for p in parts if p)
    return ""
