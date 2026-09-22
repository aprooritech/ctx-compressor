from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Depends, Header, Request

from src.compressor.engine import ContextCompressor
from src.compressor.summarizer import LLMSummarizer, NullSummarizer, Summarizer
from src.compressor.tokenizer import MessageTokenizer
from src.config import Settings
from src.db.database import Database
from src.db.repository import MetricsRepository
from src.errors import AuthenticationError, InvalidRequestError
from src.providers.registry import ProviderRegistry
from src.routing import Route


@dataclass(slots=True)
class AppState:
    settings: Settings
    http_client: httpx.AsyncClient
    registry: ProviderRegistry
    tokenizer: MessageTokenizer
    database: Database
    metrics: MetricsRepository

    def build_summarizer(self, route: Route) -> Summarizer:
        settings = self.settings
        provider_name = settings.summary_provider or route.provider
        if provider_name == "openrouter" and not settings.has_openrouter_credentials():
            return NullSummarizer()
        if provider_name == route.provider:
            model = settings.summary_model or route.model
        else:
            model = settings.summary_model or settings.default_model_for(provider_name)
        try:
            provider = self.registry.get(provider_name)
        except InvalidRequestError:
            return NullSummarizer()
        return LLMSummarizer(
            provider,
            model,
            prompt=settings.summary_prompt,
            timeout=settings.summary_timeout_seconds,
        )

    def build_compressor(self, route: Route) -> ContextCompressor:
        summarizer: Summarizer | None = None
        if self.settings.compression_strategy in ("summarize", "hybrid"):
            summarizer = self.build_summarizer(route)
        return ContextCompressor(self.tokenizer, self.settings, summarizer)


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.app_state
    return state


StateDep = Annotated[AppState, Depends(get_state)]


async def require_api_key(
    state: StateDep,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    allowed = state.settings.proxy_api_keys
    if not allowed:
        return
    presented = x_api_key or ""
    if not presented and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            presented = token.strip()
    if not presented:
        raise AuthenticationError("Missing API key. Send 'Authorization: Bearer <key>'.")
    if not any(secrets.compare_digest(presented, key) for key in allowed):
        raise AuthenticationError("Invalid API key.")


AuthDep = Annotated[None, Depends(require_api_key)]
