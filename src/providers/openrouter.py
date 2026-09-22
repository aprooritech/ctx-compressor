from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx

from src.config import ProviderName, Settings
from src.errors import ConfigurationError
from src.providers.base import ChatProvider, translate_transport_errors
from src.schemas import ChatCompletionRequest

logger = logging.getLogger(__name__)


class OpenRouterProvider(ChatProvider):
    name: ClassVar[ProviderName] = "openrouter"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        super().__init__(client, settings)
        self._base_url = settings.openrouter_base_url

    async def chat(
        self, request: ChatCompletionRequest, *, timeout: float | None = None
    ) -> dict[str, Any]:
        payload = self._build_payload(request, stream=False)
        url = f"{self._base_url}/chat/completions"
        with translate_transport_errors(self.name, url):
            response = await self._client.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self._timeout(timeout),
            )
        await self._raise_for_status(response)
        data = response.json()
        if not isinstance(data, dict):
            raise ConfigurationError("OpenRouter returned an unexpected payload shape")
        return data

    async def stream(
        self, request: ChatCompletionRequest, *, timeout: float | None = None
    ) -> AsyncIterator[bytes]:
        payload = self._build_payload(request, stream=True)
        url = f"{self._base_url}/chat/completions"
        with translate_transport_errors(self.name, url):
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers=self._headers(),
                timeout=self._timeout(timeout),
            ) as response:
                await self._raise_for_status(response)
                # OpenRouter already emits OpenAI SSE frames.
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk

    async def list_models(self) -> list[dict[str, Any]]:
        url = f"{self._base_url}/models"
        with translate_transport_errors(self.name, url):
            response = await self._client.get(
                url, headers=self._headers(), timeout=self._timeout(None)
            )
        await self._raise_for_status(response)
        payload = response.json()
        entries = payload.get("data", []) if isinstance(payload, dict) else []
        result: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id", ""))
            if not model_id:
                continue
            result.append(
                {
                    "id": f"openrouter/{model_id}",
                    "object": "model",
                    "owned_by": "openrouter",
                    "created": entry.get("created", 0),
                    "context_length": entry.get("context_length"),
                    "pricing": entry.get("pricing"),
                }
            )
        return result

    async def health(self) -> bool:
        if not self._settings.has_openrouter_credentials():
            return False
        try:
            response = await self._client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
                timeout=httpx.Timeout(5.0, connect=2.0),
            )
            return response.is_success
        except httpx.HTTPError:
            return False
        except ConfigurationError:
            return False

    def _headers(self) -> dict[str, str]:
        api_key = self._settings.openrouter_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationError(
                "OPENROUTER_API_KEY is not configured; cannot use the OpenRouter backend.",
                provider=self.name,
            )
        headers = {
            "Authorization": f"Bearer {api_key.get_secret_value().strip()}",
            "Content-Type": "application/json",
        }
        if self._settings.openrouter_site_url:
            headers["HTTP-Referer"] = self._settings.openrouter_site_url
        if self._settings.openrouter_app_name:
            headers["X-Title"] = self._settings.openrouter_app_name
        return headers

    def _build_payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        payload = request.upstream_payload(
            model=request.model or self.default_model, messages=request.messages
        )
        payload["stream"] = stream
        if payload.get("max_tokens"):
            payload.pop("max_completion_tokens", None)
        return payload
