from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from src.config import PROVIDER_NAMES, ProviderName, Settings
from src.errors import InvalidRequestError
from src.providers.base import ChatProvider
from src.providers.ollama import OllamaProvider
from src.providers.openrouter import OpenRouterProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._providers: dict[str, ChatProvider] = {
            "ollama": OllamaProvider(client, settings),
            "openrouter": OpenRouterProvider(client, settings),
        }

    def get(self, name: str) -> ChatProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise InvalidRequestError(
                f"Unknown provider '{name}'. Available providers: "
                f"{', '.join(sorted(self._providers))}.",
                details={"available_providers": sorted(self._providers)},
            ) from None

    def register(self, name: str, provider: ChatProvider) -> None:
        self._providers[name] = provider

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._providers)

    async def health(self) -> dict[str, bool]:
        names = list(self._providers)
        results = await asyncio.gather(
            *(self._providers[name].health() for name in names),
            return_exceptions=True,
        )
        health: dict[str, bool] = {}
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                logger.debug("health probe for %s failed: %s", name, result)
                health[name] = False
            else:
                health[name] = bool(result)
        return health

    async def list_models(self) -> list[dict[str, Any]]:
        names = list(self._providers)
        results = await asyncio.gather(
            *(self._providers[name].list_models() for name in names),
            return_exceptions=True,
        )
        models: list[dict[str, Any]] = []
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                logger.info("could not list models of %s: %s", name, result)
                continue
            models.extend(result)
        return models


def is_provider_name(value: str) -> bool:
    return value in PROVIDER_NAMES


def coerce_provider_name(value: str) -> ProviderName:
    if value == "ollama":
        return "ollama"
    if value == "openrouter":
        return "openrouter"
    raise InvalidRequestError(
        f"Unknown provider '{value}'. Available providers: {', '.join(PROVIDER_NAMES)}.",
        details={"available_providers": list(PROVIDER_NAMES)},
    )
