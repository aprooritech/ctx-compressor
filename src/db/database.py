from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        1,
        (
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id              TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                last_seen_at    TEXT NOT NULL,
                request_count   INTEGER NOT NULL DEFAULT 0
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS request_logs (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id             TEXT NOT NULL,
                created_at             TEXT NOT NULL,
                provider               TEXT NOT NULL,
                model                  TEXT NOT NULL,
                route_source           TEXT NOT NULL,
                strategy               TEXT NOT NULL,
                streamed               INTEGER NOT NULL DEFAULT 0,
                status                 TEXT NOT NULL,
                error_type             TEXT,
                original_tokens        INTEGER NOT NULL DEFAULT 0,
                compressed_tokens      INTEGER NOT NULL DEFAULT 0,
                saved_tokens           INTEGER NOT NULL DEFAULT 0,
                saved_ratio            REAL NOT NULL DEFAULT 0,
                message_count          INTEGER NOT NULL DEFAULT 0,
                compressed_message_count INTEGER NOT NULL DEFAULT 0,
                summarized_messages    INTEGER NOT NULL DEFAULT 0,
                dropped_messages       INTEGER NOT NULL DEFAULT 0,
                trimmed_messages       INTEGER NOT NULL DEFAULT 0,
                summary_calls          INTEGER NOT NULL DEFAULT 0,
                completion_tokens      INTEGER,
                total_latency_ms       REAL NOT NULL DEFAULT 0,
                upstream_latency_ms    REAL,
                compression_latency_ms REAL,
                preview                TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_logs_created_at ON request_logs (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_logs_session ON request_logs (session_id)",
            "CREATE INDEX IF NOT EXISTS idx_logs_provider ON request_logs (provider, model)",
        ),
    ),
)


class Database:
    def __init__(
        self,
        path: Path | str,
        *,
        enabled: bool = True,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._path = str(path)
        self._enabled = enabled
        self._busy_timeout_ms = busy_timeout_ms
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled and self._connection is not None

    @property
    def path(self) -> str:
        return self._path

    async def connect(self) -> None:
        if not self._enabled or self._connection is not None:
            return
        if self._path not in (":memory:", ""):
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = await aiosqlite.connect(self._path)
        except Exception as exc:
            logger.error("could not open the SQLite database at %s: %s", self._path, exc)
            self._enabled = False
            return
        connection.row_factory = aiosqlite.Row
        self._connection = connection
        await self._apply_pragmas()
        await self._migrate()
        logger.info("SQLite ready at %s (WAL)", self._path)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def _apply_pragmas(self) -> None:
        assert self._connection is not None
        journal_mode = "WAL" if self._path not in (":memory:", "") else "MEMORY"
        for pragma in (
            f"PRAGMA journal_mode={journal_mode}",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA foreign_keys=ON",
            "PRAGMA temp_store=MEMORY",
            f"PRAGMA busy_timeout={self._busy_timeout_ms}",
        ):
            await self._connection.execute(pragma)
        await self._connection.commit()

    async def _migrate(self) -> None:
        assert self._connection is not None
        cursor = await self._connection.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        await cursor.close()
        current = int(row[0]) if row else 0
        for version, statements in MIGRATIONS:
            if version <= current:
                continue
            for statement in statements:
                await self._connection.execute(statement)
            await self._connection.execute(f"PRAGMA user_version={version}")
            current = version
        await self._connection.commit()

    async def execute(self, sql: str, parameters: Sequence[Any] = ()) -> None:
        if self._connection is None:
            return
        async with self._write_lock:
            await self._connection.execute(sql, parameters)
            await self._connection.commit()

    async def execute_many(self, statements: Iterable[tuple[str, Sequence[Any]]]) -> None:
        if self._connection is None:
            return
        async with self._write_lock:
            for sql, parameters in statements:
                await self._connection.execute(sql, parameters)
            await self._connection.commit()

    async def fetch_one(self, sql: str, parameters: Sequence[Any] = ()) -> aiosqlite.Row | None:
        if self._connection is None:
            return None
        cursor = await self._connection.execute(sql, parameters)
        try:
            return await cursor.fetchone()
        finally:
            await cursor.close()

    async def fetch_all(self, sql: str, parameters: Sequence[Any] = ()) -> list[aiosqlite.Row]:
        if self._connection is None:
            return []
        cursor = await self._connection.execute(sql, parameters)
        try:
            return list(await cursor.fetchall())
        finally:
            await cursor.close()

    async def fetch_value(self, sql: str, parameters: Sequence[Any] = ()) -> Any:
        row = await self.fetch_one(sql, parameters)
        return row[0] if row is not None else None
