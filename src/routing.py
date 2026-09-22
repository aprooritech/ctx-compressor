from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import PROVIDER_NAMES, ProviderName, Settings
from src.errors import InvalidRequestError
from src.providers.registry import coerce_provider_name

RouteSource = Literal["header", "model_prefix", "default"]

PROVIDER_HEADER = "X-Provider"


@dataclass(frozen=True, slots=True)
class Route:
    provider: ProviderName
    model: str
    source: RouteSource

    @property
    def qualified_model(self) -> str:
        return f"{self.provider}/{self.model}"


def split_model(model: str) -> tuple[ProviderName | None, str]:
    # Only the first path segment counts, so "anthropic/claude-3.5-sonnet"
    # stays intact while "openrouter/anthropic/claude-3.5-sonnet" is split.
    candidate = model.strip()
    if "/" not in candidate:
        return None, candidate
    prefix, remainder = candidate.split("/", 1)
    prefix = prefix.strip().lower()
    if prefix in PROVIDER_NAMES and remainder.strip():
        return coerce_provider_name(prefix), remainder.strip()
    return None, candidate


def resolve_route(model: str | None, provider_header: str | None, settings: Settings) -> Route:
    raw_model = (model or "").strip()
    prefixed_provider, bare_model = split_model(raw_model)

    header = (provider_header or "").strip().lower()
    if header:
        provider = coerce_provider_name(header)
        if prefixed_provider is not None and prefixed_provider != provider:
            raise InvalidRequestError(
                f"Conflicting routing: header '{PROVIDER_HEADER}: {header}' selects "
                f"'{provider}' but the model id '{raw_model}' targets "
                f"'{prefixed_provider}'. Remove one of them.",
                details={"header_provider": provider, "model_provider": prefixed_provider},
            )
        return Route(
            provider=provider,
            model=bare_model or settings.default_model_for(provider),
            source="header",
        )

    if prefixed_provider is not None:
        return Route(provider=prefixed_provider, model=bare_model, source="model_prefix")

    provider = settings.default_provider
    return Route(
        provider=provider,
        model=bare_model or settings.default_model_for(provider),
        source="default",
    )


def fallback_route(route: Route, settings: Settings) -> Route | None:
    if not settings.enable_provider_fallback:
        return None
    target = settings.fallback_provider
    if target == route.provider:
        return None
    if target == "openrouter" and not settings.has_openrouter_credentials():
        return None
    # Model ids are provider specific, so the fallback uses its own default.
    return Route(provider=target, model=settings.default_model_for(target), source="default")
