from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Annotated, Any

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.deps import AppState, AuthDep, StateDep
from src.compressor.engine import CompressionResult
from src.db.repository import RequestLogEntry
from src.errors import ProviderTimeoutError, ProviderUnavailableError, ProxyError
from src.providers.base import SSE_DONE, ChatProvider, sse_event
from src.routing import Route, fallback_route, resolve_route
from src.schemas import ChatCompletionRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

PREVIEW_LENGTH = 200


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    payload: ChatCompletionRequest,
    state: StateDep,
    _: AuthDep,
    x_provider: Annotated[str | None, Header(alias="X-Provider")] = None,
    x_session_id: Annotated[str | None, Header(alias="X-Session-Id")] = None,
) -> JSONResponse | StreamingResponse:
    started = perf_counter()
    session_id = (x_session_id or "").strip() or f"sess-{uuid.uuid4().hex[:16]}"
    route = resolve_route(payload.model, x_provider, state.settings)

    result = await state.build_compressor(route).compress(payload.messages)
    if result.was_compressed:
        logger.info(
            "compressed session=%s %s tokens -> %s (%.1f%%, %s)",
            session_id,
            result.original_tokens,
            result.compressed_tokens,
            result.saved_ratio * 100,
            result.strategy_applied,
        )

    upstream_request = payload.model_copy(
        update={"messages": result.messages, "model": route.model}
    )

    if payload.stream:
        return await _stream_response(state, route, upstream_request, result, session_id, started)
    return await _json_response(state, route, upstream_request, result, session_id, started)


@router.get("/models")
async def list_models(state: StateDep, _: AuthDep) -> dict[str, Any]:
    return {"object": "list", "data": await state.registry.list_models()}


async def _json_response(
    state: AppState,
    route: Route,
    request: ChatCompletionRequest,
    result: CompressionResult,
    session_id: str,
    started: float,
) -> JSONResponse:
    upstream_started = perf_counter()
    used_route = route
    fell_back = False
    try:
        try:
            data = await state.registry.get(route.provider).chat(request)
        except (ProviderUnavailableError, ProviderTimeoutError) as exc:
            alternative = fallback_route(route, state.settings)
            if alternative is None:
                raise
            logger.warning(
                "%s unavailable (%s); falling back to %s",
                route.provider,
                exc.message,
                alternative.provider,
            )
            used_route = alternative
            fell_back = True
            data = await state.registry.get(alternative.provider).chat(
                request.model_copy(update={"model": alternative.model})
            )
    except ProxyError as exc:
        await _log(
            state,
            session_id,
            used_route,
            result,
            request,
            status="error",
            error_type=exc.error_type,
            started=started,
            upstream_started=upstream_started,
            streamed=False,
        )
        raise

    upstream_ms = (perf_counter() - upstream_started) * 1000
    await _log(
        state,
        session_id,
        used_route,
        result,
        request,
        status="ok",
        error_type=None,
        started=started,
        upstream_started=upstream_started,
        streamed=False,
        completion_tokens=_completion_tokens(data),
    )
    headers = _metric_headers(
        result, used_route, session_id, state, upstream_ms=upstream_ms, fell_back=fell_back
    )
    return JSONResponse(content=data, headers=headers)


async def _open_stream(
    state: AppState,
    route: Route,
    request: ChatCompletionRequest,
) -> tuple[AsyncIterator[bytes], bytes | None, Route, bool]:
    # Connection problems surface on the first __anext__, so pulling one chunk
    # here allows a fallback before the HTTP response has begun.
    provider: ChatProvider = state.registry.get(route.provider)
    iterator = provider.stream(request)
    try:
        return iterator, await anext(iterator, None), route, False
    except (ProviderUnavailableError, ProviderTimeoutError) as exc:
        alternative = fallback_route(route, state.settings)
        if alternative is None:
            raise
        logger.warning(
            "%s unavailable (%s); streaming from %s instead",
            route.provider,
            exc.message,
            alternative.provider,
        )
        fallback_iterator = state.registry.get(alternative.provider).stream(
            request.model_copy(update={"model": alternative.model})
        )
        return fallback_iterator, await anext(fallback_iterator, None), alternative, True


async def _stream_response(
    state: AppState,
    route: Route,
    request: ChatCompletionRequest,
    result: CompressionResult,
    session_id: str,
    started: float,
) -> StreamingResponse:
    upstream_started = perf_counter()
    try:
        iterator, first_chunk, used_route, fell_back = await _open_stream(state, route, request)
    except ProxyError as exc:
        await _log(
            state,
            session_id,
            route,
            result,
            request,
            status="error",
            error_type=exc.error_type,
            started=started,
            upstream_started=upstream_started,
            streamed=True,
        )
        raise

    headers = _metric_headers(
        result, used_route, session_id, state, upstream_ms=None, fell_back=fell_back
    )
    headers["Cache-Control"] = "no-cache"
    headers["X-Accel-Buffering"] = "no"

    async def body() -> AsyncIterator[bytes]:
        status = "ok"
        error_type: str | None = None
        try:
            if first_chunk is not None:
                yield first_chunk
            async for chunk in iterator:
                yield chunk
        except ProxyError as exc:
            status, error_type = "error", exc.error_type
            logger.warning("stream aborted: %s", exc.message)
            yield sse_event(exc.to_payload())
            yield SSE_DONE
        except Exception as exc:
            status, error_type = "error", "proxy_error"
            logger.exception("unexpected streaming failure")
            yield sse_event({"error": {"message": str(exc), "type": "proxy_error"}})
            yield SSE_DONE
        finally:
            try:
                await _log(
                    state,
                    session_id,
                    used_route,
                    result,
                    request,
                    status=status,
                    error_type=error_type,
                    started=started,
                    upstream_started=upstream_started,
                    streamed=True,
                )
            except Exception:
                logger.debug("could not persist streaming metrics", exc_info=True)

    return StreamingResponse(body(), media_type="text/event-stream", headers=headers)


def _metric_headers(
    result: CompressionResult,
    route: Route,
    session_id: str,
    state: AppState,
    *,
    upstream_ms: float | None,
    fell_back: bool,
) -> dict[str, str]:
    headers = {
        "X-Original-Tokens": str(result.original_tokens),
        "X-Compressed-Tokens": str(result.compressed_tokens),
        "X-Saved-Tokens": str(result.saved_tokens),
        "X-Saved-Ratio": f"{result.saved_ratio:.4f}",
        "X-Compression-Strategy": result.strategy_applied,
        "X-Compression-Budget": str(result.budget_tokens),
        "X-Compression-Ms": f"{result.duration_ms:.2f}",
        "X-Messages-In": str(result.original_message_count),
        "X-Messages-Out": str(result.compressed_message_count),
        "X-Provider": route.provider,
        "X-Model": route.model,
        "X-Route-Source": route.source,
        "X-Session-Id": session_id,
        "X-Tokenizer": state.tokenizer.name,
    }
    if upstream_ms is not None:
        headers["X-Upstream-Ms"] = f"{upstream_ms:.2f}"
    if fell_back:
        headers["X-Provider-Fallback"] = "true"
    return {key: value.encode("ascii", "replace").decode("ascii") for key, value in headers.items()}


def _completion_tokens(data: dict[str, Any]) -> int | None:
    usage = data.get("usage")
    if isinstance(usage, dict):
        value = usage.get("completion_tokens")
        if isinstance(value, int):
            return value
    return None


async def _log(
    state: AppState,
    session_id: str,
    route: Route,
    result: CompressionResult,
    request: ChatCompletionRequest,
    *,
    status: str,
    error_type: str | None,
    started: float,
    upstream_started: float,
    streamed: bool,
    completion_tokens: int | None = None,
) -> None:
    preview: str | None = None
    if state.settings.persist_message_preview:
        for message in reversed(request.messages):
            if message.role == "user":
                preview = message.text_content[:PREVIEW_LENGTH]
                break
    await state.metrics.log_request(
        RequestLogEntry(
            session_id=session_id,
            provider=route.provider,
            model=route.model,
            route_source=route.source,
            strategy=result.strategy_applied,
            status=status,
            error_type=error_type,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            saved_tokens=result.saved_tokens,
            saved_ratio=result.saved_ratio,
            message_count=result.original_message_count,
            compressed_message_count=result.compressed_message_count,
            summarized_messages=result.summarized_messages,
            dropped_messages=result.dropped_messages,
            trimmed_messages=result.trimmed_messages,
            summary_calls=result.summary_calls,
            completion_tokens=completion_tokens,
            streamed=streamed,
            total_latency_ms=(perf_counter() - started) * 1000,
            upstream_latency_ms=(perf_counter() - upstream_started) * 1000,
            compression_latency_ms=result.duration_ms,
            preview=preview,
        )
    )
