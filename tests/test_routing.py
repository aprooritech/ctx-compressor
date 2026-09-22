from __future__ import annotations

import pytest

from src.config import Settings
from src.errors import InvalidRequestError
from src.routing import fallback_route, resolve_route, split_model


@pytest.mark.parametrize(
    ("model", "header", "expected_provider", "expected_model", "expected_source"),
    [
        ("ollama/llama3", None, "ollama", "llama3", "model_prefix"),
        (
            "openrouter/anthropic/claude-3.5-sonnet",
            None,
            "openrouter",
            "anthropic/claude-3.5-sonnet",
            "model_prefix",
        ),
        ("llama3", None, "ollama", "llama3", "default"),
        ("mistral:7b", "openrouter", "openrouter", "mistral:7b", "header"),
        ("ollama/llama3", "ollama", "ollama", "llama3", "header"),
        ("anthropic/claude-3.5-sonnet", None, "ollama", "anthropic/claude-3.5-sonnet", "default"),
    ],
)
def test_route_resolution(
    settings: Settings,
    model: str,
    header: str | None,
    expected_provider: str,
    expected_model: str,
    expected_source: str,
) -> None:
    route = resolve_route(model, header, settings)

    assert route.provider == expected_provider
    assert route.model == expected_model
    assert route.source == expected_source


def test_empty_model_falls_back_to_the_provider_default(settings: Settings) -> None:
    route = resolve_route("", None, settings)

    assert route.provider == "ollama"
    assert route.model == settings.ollama_default_model
    assert route.qualified_model == f"ollama/{settings.ollama_default_model}"


def test_header_wins_and_uses_that_providers_default_model(settings: Settings) -> None:
    route = resolve_route(None, "openrouter", settings)

    assert route.provider == "openrouter"
    assert route.model == settings.openrouter_default_model


def test_conflicting_header_and_model_prefix_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidRequestError) as excinfo:
        resolve_route("openrouter/gpt-4o", "ollama", settings)

    assert "Conflicting routing" in excinfo.value.message
    assert excinfo.value.status_code == 400


def test_unknown_provider_header_is_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidRequestError) as excinfo:
        resolve_route("llama3", "groq", settings)

    assert excinfo.value.details["available_providers"] == ["ollama", "openrouter"]


def test_split_model_only_strips_known_prefixes() -> None:
    assert split_model("ollama/llama3") == ("ollama", "llama3")
    assert split_model("meta/llama3") == (None, "meta/llama3")
    assert split_model("llama3") == (None, "llama3")
    assert split_model("ollama/") == (None, "ollama/")


def test_fallback_is_disabled_by_default(settings: Settings) -> None:
    route = resolve_route("llama3", None, settings)

    assert fallback_route(route, settings) is None


def test_fallback_requires_credentials_for_openrouter(settings: Settings) -> None:
    enabled = settings.model_copy(
        update={"enable_provider_fallback": True, "fallback_provider": "openrouter"}
    )
    route = resolve_route("llama3", None, enabled)

    assert fallback_route(route, enabled) is None

    with_key = enabled.model_copy(update={"openrouter_api_key": _secret("sk-or-test")})
    fallback = fallback_route(route, with_key)

    assert fallback is not None
    assert fallback.provider == "openrouter"
    assert fallback.model == with_key.openrouter_default_model


def _secret(value: str):  # type: ignore[no-untyped-def]
    from pydantic import SecretStr

    return SecretStr(value)
