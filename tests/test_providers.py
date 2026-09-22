from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.config import Settings
from src.errors import (
    ConfigurationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UpstreamError,
)
from src.providers.ollama import OllamaProvider
from src.providers.openrouter import OpenRouterProvider
from src.schemas import ChatCompletionRequest, ChatMessage

OLLAMA_URL = "http://localhost:11434/api/chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OLLAMA_REPLY = {
    "model": "llama3",
    "created_at": "2026-09-20T10:00:00Z",
    "message": {"role": "assistant", "content": "Hello world"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 17,
    "eval_count": 2,
}

OLLAMA_STREAM = (
    b'{"model":"llama3","message":{"role":"assistant","content":"Hello"},"done":false}\n'
    b'{"model":"llama3","message":{"role":"assistant","content":" world"},"done":true,'
    b'"done_reason":"stop","prompt_eval_count":17,"eval_count":2}\n'
)


def build_request(**overrides: object) -> ChatCompletionRequest:
    payload: dict[str, object] = {
        "model": "llama3",
        "messages": [ChatMessage(role="user", content="hi")],
    }
    payload.update(overrides)
    return ChatCompletionRequest(**payload)


@pytest.fixture
def provider_settings() -> Settings:
    from pydantic import SecretStr

    return Settings(
        _env_file=None,
        OPENROUTER_API_KEY=SecretStr("sk-or-test-key"),
        OPENROUTER_SITE_URL="http://localhost:8000",
        OPENROUTER_APP_NAME="ccproxy-test",
    )


async def test_ollama_maps_openai_parameters_onto_options(provider_settings: Settings) -> None:
    async with respx.mock:
        route = respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=OLLAMA_REPLY))
        async with httpx.AsyncClient() as client:
            await OllamaProvider(client, provider_settings).chat(
                build_request(temperature=0.2, max_tokens=64, stop=["END"], seed=7)
            )

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "llama3"
    assert sent["stream"] is False
    assert sent["keep_alive"] == provider_settings.ollama_keep_alive
    assert sent["options"] == {
        "temperature": 0.2,
        "seed": 7,
        "num_predict": 64,
        "stop": ["END"],
    }
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


async def test_ollama_response_is_normalised_to_openai_shape(
    provider_settings: Settings,
) -> None:
    async with respx.mock:
        respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=OLLAMA_REPLY))
        async with httpx.AsyncClient() as client:
            data = await OllamaProvider(client, provider_settings).chat(build_request())

    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "Hello world"}
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 17, "completion_tokens": 2, "total_tokens": 19}


async def test_ollama_stream_is_translated_into_openai_sse(
    provider_settings: Settings,
) -> None:
    async with respx.mock:
        respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, content=OLLAMA_STREAM))
        async with httpx.AsyncClient() as client:
            provider = OllamaProvider(client, provider_settings)
            chunks = [chunk async for chunk in provider.stream(build_request(stream=True))]

    assert chunks[-1] == b"data: [DONE]\n\n"
    frames = [json.loads(chunk[len(b"data: ") :]) for chunk in chunks[:-1]]
    assert frames[0]["object"] == "chat.completion.chunk"
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant", "content": "Hello"}
    assert frames[1]["choices"][0]["delta"] == {"content": " world"}
    assert frames[1]["choices"][0]["finish_reason"] == "stop"
    assert frames[1]["usage"]["total_tokens"] == 19


async def test_ollama_offline_raises_provider_unavailable(
    provider_settings: Settings,
) -> None:
    async with respx.mock:
        respx.post(OLLAMA_URL).mock(side_effect=httpx.ConnectError("connection refused"))
        async with httpx.AsyncClient() as client:
            with pytest.raises(ProviderUnavailableError) as excinfo:
                await OllamaProvider(client, provider_settings).chat(build_request())

    assert excinfo.value.status_code == 503
    assert excinfo.value.provider == "ollama"
    assert "Is the service running?" in excinfo.value.message


async def test_ollama_timeout_raises_provider_timeout(provider_settings: Settings) -> None:
    async with respx.mock:
        respx.post(OLLAMA_URL).mock(side_effect=httpx.ReadTimeout("too slow"))
        async with httpx.AsyncClient() as client:
            with pytest.raises(ProviderTimeoutError) as excinfo:
                await OllamaProvider(client, provider_settings).chat(build_request())

    assert excinfo.value.status_code == 504


async def test_ollama_error_status_is_forwarded_as_upstream_error(
    provider_settings: Settings,
) -> None:
    async with respx.mock:
        respx.post(OLLAMA_URL).mock(
            return_value=httpx.Response(404, json={"error": 'model "nope" not found'})
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(UpstreamError) as excinfo:
                await OllamaProvider(client, provider_settings).chat(build_request(model="nope"))

    assert excinfo.value.status_code == 404
    assert excinfo.value.upstream_status == 404
    assert "not found" in excinfo.value.message


async def test_ollama_lists_models_with_provider_prefix(provider_settings: Settings) -> None:
    async with respx.mock:
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})
        )
        async with httpx.AsyncClient() as client:
            models = await OllamaProvider(client, provider_settings).list_models()

    assert models[0]["id"] == "ollama/llama3:8b"
    assert models[0]["owned_by"] == "ollama"


async def test_openrouter_sends_auth_and_attribution_headers(
    provider_settings: Settings,
) -> None:
    reply = {"id": "gen-1", "object": "chat.completion", "choices": []}
    async with respx.mock:
        route = respx.post(OPENROUTER_URL).mock(return_value=httpx.Response(200, json=reply))
        async with httpx.AsyncClient() as client:
            data = await OpenRouterProvider(client, provider_settings).chat(
                build_request(model="anthropic/claude-3.5-sonnet")
            )

    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-or-test-key"
    assert request.headers["http-referer"] == "http://localhost:8000"
    assert request.headers["x-title"] == "ccproxy-test"
    body = json.loads(request.content)
    assert body["model"] == "anthropic/claude-3.5-sonnet"
    assert body["stream"] is False
    assert data == reply


async def test_openrouter_without_api_key_raises_configuration_error() -> None:
    settings = Settings(_env_file=None, OPENROUTER_API_KEY=None)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ConfigurationError) as excinfo:
            await OpenRouterProvider(client, settings).chat(build_request())

    assert "OPENROUTER_API_KEY" in excinfo.value.message
    assert excinfo.value.status_code == 500


async def test_openrouter_streams_sse_frames_unchanged(provider_settings: Settings) -> None:
    upstream = b'data: {"id":"gen-1","choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    async with respx.mock:
        respx.post(OPENROUTER_URL).mock(return_value=httpx.Response(200, content=upstream))
        async with httpx.AsyncClient() as client:
            provider = OpenRouterProvider(client, provider_settings)
            received = b"".join(
                [chunk async for chunk in provider.stream(build_request(stream=True))]
            )

    assert received == upstream
