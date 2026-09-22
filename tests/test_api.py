from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.deps import AppState
from src.app import create_app
from src.config import Settings
from tests.conftest import FakeProvider, make_conversation

SHORT_BODY = {
    "model": "llama3",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hi."},
    ],
}


async def test_chat_completion_returns_openai_payload_and_metric_headers(
    client: httpx.AsyncClient, fake_ollama: FakeProvider
) -> None:
    response = await client.post("/v1/chat/completions", json=SHORT_BODY)

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "hello from ollama"

    headers = response.headers
    assert int(headers["x-original-tokens"]) > 0
    assert int(headers["x-compressed-tokens"]) == int(headers["x-original-tokens"])
    assert headers["x-saved-ratio"] == "0.0000"
    assert headers["x-compression-strategy"] == "none"
    assert headers["x-provider"] == "ollama"
    assert headers["x-model"] == "llama3"
    assert headers["x-route-source"] == "default"
    assert headers["x-session-id"].startswith("sess-")
    assert "x-upstream-ms" in headers

    assert len(fake_ollama.chat_calls[-1].messages) == 2


async def test_long_conversation_is_compressed_before_it_reaches_the_backend(
    client: httpx.AsyncClient, fake_ollama: FakeProvider, settings: Settings
) -> None:
    messages = [m.model_dump(exclude_none=True) for m in make_conversation()]

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "llama3", "messages": messages},
        headers={"X-Session-Id": "session-under-test"},
    )

    assert response.status_code == 200
    headers = response.headers
    original = int(headers["x-original-tokens"])
    compressed = int(headers["x-compressed-tokens"])
    assert compressed < original
    assert float(headers["x-saved-ratio"]) > 0.0
    assert "summarize" in headers["x-compression-strategy"]
    assert headers["x-session-id"] == "session-under-test"
    assert compressed <= settings.compression_trigger_tokens

    forwarded = fake_ollama.chat_calls[-1]
    assert len(forwarded.messages) < len(messages)
    assert forwarded.messages[0].role == "system"
    assert forwarded.messages[-1].text_content == "So what did we decide in the end?"
    assert fake_ollama.summary_calls


@pytest.mark.parametrize(
    ("body_model", "header", "expected"),
    [
        ("openrouter/anthropic/claude-3.5-sonnet", None, "openrouter"),
        ("ollama/llama3", None, "ollama"),
        ("some-model", "openrouter", "openrouter"),
    ],
)
async def test_provider_is_selected_by_prefix_or_header(
    client: httpx.AsyncClient,
    fake_ollama: FakeProvider,
    fake_openrouter: FakeProvider,
    body_model: str,
    header: str | None,
    expected: str,
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={**SHORT_BODY, "model": body_model},
        headers={"X-Provider": header} if header else None,
    )

    assert response.status_code == 200
    assert response.headers["x-provider"] == expected
    target = fake_openrouter if expected == "openrouter" else fake_ollama
    other = fake_ollama if expected == "openrouter" else fake_openrouter
    assert target.chat_calls
    assert not other.chat_calls
    assert not target.chat_calls[-1].model.startswith(expected + "/")


async def test_conflicting_routing_returns_400(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={**SHORT_BODY, "model": "openrouter/gpt-4o"},
        headers={"X-Provider": "ollama"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_unreachable_backend_returns_503_with_openai_error_shape(
    client: httpx.AsyncClient, fake_ollama: FakeProvider
) -> None:
    fake_ollama.go_offline()

    response = await client.post("/v1/chat/completions", json=SHORT_BODY)

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "provider_unavailable"
    assert error["provider"] == "ollama"
    assert "Is the service running?" in error["message"]


async def test_streaming_returns_sse_frames_and_done_sentinel(
    client: httpx.AsyncClient,
) -> None:
    async with client.stream(
        "POST", "/v1/chat/completions", json={**SHORT_BODY, "stream": True}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-compression-strategy"] == "none"
        assert response.headers["x-provider"] == "ollama"
        body = "".join([chunk async for chunk in response.aiter_text()])

    frames = [line for line in body.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "data: [DONE]"
    assert '"chat.completion.chunk"' in frames[0]


async def test_validation_error_returns_422(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/chat/completions", json={"model": "llama3"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_models_endpoint_aggregates_backends(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/models")

    assert response.status_code == 200
    ids = {entry["id"] for entry in response.json()["data"]}
    assert ids == {"ollama/fake-model", "openrouter/fake-model"}


async def test_stats_endpoint_aggregates_token_savings(
    client: httpx.AsyncClient, app_state: AppState
) -> None:
    messages = [m.model_dump(exclude_none=True) for m in make_conversation()]
    await client.post("/v1/chat/completions", json=SHORT_BODY)
    await client.post("/v1/chat/completions", json={"model": "llama3", "messages": messages})

    response = await client.get("/v1/stats")

    assert response.status_code == 200
    stats = response.json()
    assert stats["total_requests"] == 2
    assert stats["successful_requests"] == 2
    assert stats["failed_requests"] == 0
    assert stats["compressed_requests"] == 1
    assert stats["sessions"] == 2
    assert stats["total_saved_tokens"] > 0
    assert 0.0 < stats["overall_saved_ratio"] < 1.0
    assert stats["avg_total_latency_ms"] >= 0.0
    providers = {row["provider"]: row for row in stats["by_provider"]}
    assert providers["ollama"]["requests"] == 2
    assert providers["ollama"]["saved_tokens"] > 0


async def test_failed_requests_are_recorded_in_the_stats(
    client: httpx.AsyncClient, fake_ollama: FakeProvider
) -> None:
    fake_ollama.go_offline()
    await client.post("/v1/chat/completions", json=SHORT_BODY)

    stats = (await client.get("/v1/stats")).json()
    assert stats["failed_requests"] == 1

    recent = (await client.get("/v1/stats/requests?limit=5")).json()
    assert recent[0]["status"] == "error"
    assert recent[0]["error_type"] == "provider_unavailable"


async def test_window_filter_limits_the_aggregation(client: httpx.AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=SHORT_BODY)

    recent = (await client.get("/v1/stats?window_hours=24")).json()
    assert recent["total_requests"] == 1
    assert recent["window_hours"] == 24.0


async def test_api_key_is_enforced_when_configured(settings: Settings, app_state: AppState) -> None:
    secured = settings.model_copy(update={"proxy_api_keys_raw": "secret-key, other-key"})
    app_state.settings = secured
    app = create_app(secured)
    app.state.app_state = app_state

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as http:
        unauthorized = await http.post("/v1/chat/completions", json=SHORT_BODY)
        wrong = await http.post(
            "/v1/chat/completions", json=SHORT_BODY, headers={"Authorization": "Bearer nope"}
        )
        authorized = await http.post(
            "/v1/chat/completions",
            json=SHORT_BODY,
            headers={"Authorization": "Bearer secret-key"},
        )

    assert unauthorized.status_code == 401
    assert wrong.status_code == 401
    assert authorized.status_code == 200


def test_application_lifespan_boots_and_reports_health(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        _env_file=None,
        DB_PATH=tmp_path / "lifespan.db",
        TOKENIZER_BACKEND="approx",
        OLLAMA_HOST="http://127.0.0.1:1",  # guaranteed connection refused
    )
    app: FastAPI = create_app(settings)

    with TestClient(app) as client:
        health = client.get("/healthz")
        root = client.get("/")

    assert health.status_code == 200
    payload = health.json()
    assert payload["providers"] == {"ollama": False, "openrouter": False}
    assert payload["status"] == "degraded"
    assert payload["database"] is True
    assert payload["tokenizer"] == "approx"
    assert (tmp_path / "lifespan.db").exists()
    assert root.json()["default_provider"] == "ollama"


def test_startup_survives_a_stalled_tokenizer_download(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    import time

    def never_returns(settings: Settings) -> None:
        time.sleep(2)  # longer than the budget below, short enough not to stall teardown

    monkeypatch.setattr("src.app.build_message_tokenizer", never_returns)
    monkeypatch.setattr("src.app.TOKENIZER_INIT_SLACK_SECONDS", 0.05)
    settings = Settings(
        _env_file=None,
        DB_PATH=tmp_path / "stalled.db",
        TOKENIZER_INIT_TIMEOUT_SECONDS=0.05,
        OLLAMA_HOST="http://127.0.0.1:1",
    )

    started = time.perf_counter()
    with TestClient(create_app(settings)) as client:
        elapsed = time.perf_counter() - started
        health = client.get("/healthz").json()

    assert elapsed < 5.0, "startup blocked on the stalled download"
    assert health["tokenizer"] == "approx"
    assert health["status"] == "degraded"
