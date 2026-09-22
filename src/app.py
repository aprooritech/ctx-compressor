from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import chat, stats
from src.api.deps import AppState
from src.compressor.tokenizer import (
    ApproxTokenCounter,
    MessageTokenizer,
    build_message_tokenizer,
)
from src.config import Settings, get_settings
from src.db.database import Database
from src.db.repository import MetricsRepository
from src.errors import register_exception_handlers
from src.providers.registry import ProviderRegistry
from src.schemas import HealthResponse

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
TOKENIZER_INIT_SLACK_SECONDS = 5.0

EXPOSED_HEADERS = [
    "X-Original-Tokens",
    "X-Compressed-Tokens",
    "X-Saved-Tokens",
    "X-Saved-Ratio",
    "X-Compression-Strategy",
    "X-Compression-Budget",
    "X-Compression-Ms",
    "X-Messages-In",
    "X-Messages-Out",
    "X-Provider",
    "X-Provider-Fallback",
    "X-Model",
    "X-Route-Source",
    "X-Session-Id",
    "X-Tokenizer",
    "X-Upstream-Ms",
]


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


async def build_tokenizer(settings: Settings) -> MessageTokenizer:
    # socket_deadline bounds the download itself; this guard covers anything
    # else that could stall, so the proxy always comes up.
    budget = settings.tokenizer_init_timeout_seconds + TOKENIZER_INIT_SLACK_SECONDS
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(build_message_tokenizer, settings), timeout=budget
        )
    except TimeoutError:
        logger.warning(
            "tokenizer initialisation exceeded %.1fs; using the approximation instead", budget
        )
        return MessageTokenizer(ApproxTokenCounter())


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    configure_logging(active_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=active_settings.max_connections,
                max_keepalive_connections=active_settings.max_keepalive_connections,
            ),
            timeout=httpx.Timeout(
                active_settings.request_timeout_seconds,
                connect=active_settings.connect_timeout_seconds,
            ),
            follow_redirects=True,
        )
        database = Database(
            active_settings.db_path,
            enabled=active_settings.db_enabled,
            busy_timeout_ms=active_settings.db_busy_timeout_ms,
        )
        await database.connect()
        tokenizer = await build_tokenizer(active_settings)
        app.state.app_state = AppState(
            settings=active_settings,
            http_client=client,
            registry=ProviderRegistry(client, active_settings),
            tokenizer=tokenizer,
            database=database,
            metrics=MetricsRepository(database),
        )
        logger.info(
            "%s ready | default provider=%s | tokenizer=%s | budget=%s/%s tokens",
            active_settings.app_name,
            active_settings.default_provider,
            tokenizer.name,
            active_settings.compression_target_tokens,
            active_settings.compression_trigger_tokens,
        )
        try:
            yield
        finally:
            await database.close()
            await client.aclose()

    app = FastAPI(
        title=active_settings.app_name,
        version=VERSION,
        summary="OpenAI-compatible proxy with semantic context-window compression.",
        lifespan=lifespan,
    )
    app.state.settings = active_settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=EXPOSED_HEADERS,
    )
    register_exception_handlers(app)
    app.include_router(chat.router, prefix="/v1")
    app.include_router(stats.router, prefix="/v1")

    @app.get("/healthz", response_model=HealthResponse, tags=["observability"])
    async def healthz() -> HealthResponse:
        state: AppState = app.state.app_state
        providers = await state.registry.health()
        return HealthResponse(
            status="ok" if any(providers.values()) else "degraded",
            version=VERSION,
            tokenizer=state.tokenizer.name,
            compression_enabled=state.settings.compression_enabled,
            providers=providers,
            database=state.database.enabled,
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, object]:
        return {
            "name": active_settings.app_name,
            "version": VERSION,
            "endpoints": ["/v1/chat/completions", "/v1/models", "/v1/stats", "/healthz", "/docs"],
            "default_provider": active_settings.default_provider,
            "max_context_tokens": active_settings.max_context_tokens,
        }

    return app
