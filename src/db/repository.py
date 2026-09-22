from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src.db.database import Database
from src.schemas import (
    ModelUsageStats,
    ProviderUsageStats,
    RequestLogItem,
    StatsResponse,
)

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="milliseconds")


@dataclass(slots=True)
class RequestLogEntry:
    session_id: str
    provider: str
    model: str
    route_source: str
    strategy: str
    status: str
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    saved_ratio: float
    message_count: int
    compressed_message_count: int
    total_latency_ms: float
    streamed: bool = False
    error_type: str | None = None
    summarized_messages: int = 0
    dropped_messages: int = 0
    trimmed_messages: int = 0
    summary_calls: int = 0
    completion_tokens: int | None = None
    upstream_latency_ms: float | None = None
    compression_latency_ms: float | None = None
    preview: str | None = None
    created_at: datetime = field(default_factory=utc_now)


class MetricsRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def log_request(self, entry: RequestLogEntry) -> None:
        if not self._db.enabled:
            return
        timestamp = _iso(entry.created_at)
        try:
            await self._db.execute_many(
                [
                    (
                        """
                        INSERT INTO sessions (id, created_at, last_seen_at, request_count)
                        VALUES (?, ?, ?, 1)
                        ON CONFLICT(id) DO UPDATE SET
                            last_seen_at = excluded.last_seen_at,
                            request_count = sessions.request_count + 1
                        """,
                        (entry.session_id, timestamp, timestamp),
                    ),
                    (
                        """
                        INSERT INTO request_logs (
                            session_id, created_at, provider, model, route_source, strategy,
                            streamed, status, error_type, original_tokens, compressed_tokens,
                            saved_tokens, saved_ratio, message_count, compressed_message_count,
                            summarized_messages, dropped_messages, trimmed_messages,
                            summary_calls, completion_tokens, total_latency_ms,
                            upstream_latency_ms, compression_latency_ms, preview
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entry.session_id,
                            timestamp,
                            entry.provider,
                            entry.model,
                            entry.route_source,
                            entry.strategy,
                            int(entry.streamed),
                            entry.status,
                            entry.error_type,
                            entry.original_tokens,
                            entry.compressed_tokens,
                            entry.saved_tokens,
                            entry.saved_ratio,
                            entry.message_count,
                            entry.compressed_message_count,
                            entry.summarized_messages,
                            entry.dropped_messages,
                            entry.trimmed_messages,
                            entry.summary_calls,
                            entry.completion_tokens,
                            entry.total_latency_ms,
                            entry.upstream_latency_ms,
                            entry.compression_latency_ms,
                            entry.preview,
                        ),
                    ),
                ]
            )
        except Exception as exc:
            logger.warning("could not persist request metrics: %s", exc)

    async def aggregate(
        self, *, window_hours: float | None = None, model_limit: int = 20
    ) -> StatsResponse:
        since = utc_now() - timedelta(hours=window_hours) if window_hours else None
        where, params = ("WHERE created_at >= ?", [_iso(since)]) if since else ("", [])

        stats = StatsResponse(window_hours=window_hours, generated_at=_iso(utc_now()))
        if not self._db.enabled:
            return stats

        totals = await self._db.fetch_one(
            f"""
            SELECT
                COUNT(*)                                        AS total_requests,
                COALESCE(SUM(status = 'ok'), 0)                 AS successful,
                COALESCE(SUM(status != 'ok'), 0)                AS failed,
                COALESCE(SUM(strategy != 'none'), 0)            AS compressed_requests,
                COUNT(DISTINCT session_id)                      AS sessions,
                COALESCE(SUM(original_tokens), 0)               AS original_tokens,
                COALESCE(SUM(compressed_tokens), 0)             AS compressed_tokens,
                COALESCE(SUM(saved_tokens), 0)                  AS saved_tokens,
                COALESCE(AVG(saved_ratio), 0.0)                 AS avg_saved_ratio,
                COALESCE(AVG(total_latency_ms), 0.0)            AS avg_latency,
                COALESCE(AVG(upstream_latency_ms), 0.0)         AS avg_upstream_latency,
                COALESCE(AVG(compression_latency_ms), 0.0)      AS avg_compression_latency
            FROM request_logs
            {where}
            """,
            params,
        )
        if totals is None or int(totals["total_requests"]) == 0:
            return stats

        original = int(totals["original_tokens"])
        saved = int(totals["saved_tokens"])
        stats.total_requests = int(totals["total_requests"])
        stats.successful_requests = int(totals["successful"])
        stats.failed_requests = int(totals["failed"])
        stats.compressed_requests = int(totals["compressed_requests"])
        stats.sessions = int(totals["sessions"])
        stats.total_original_tokens = original
        stats.total_compressed_tokens = int(totals["compressed_tokens"])
        stats.total_saved_tokens = saved
        stats.overall_saved_ratio = round(saved / original, 6) if original else 0.0
        stats.avg_saved_ratio = round(float(totals["avg_saved_ratio"]), 6)
        stats.avg_total_latency_ms = round(float(totals["avg_latency"]), 3)
        stats.avg_upstream_latency_ms = round(float(totals["avg_upstream_latency"]), 3)
        stats.avg_compression_latency_ms = round(float(totals["avg_compression_latency"]), 3)
        stats.p95_total_latency_ms = await self._percentile_latency(0.95, where, params)
        stats.by_provider = await self._by_provider(where, params)
        stats.by_model = await self._by_model(where, params, model_limit)
        return stats

    async def _percentile_latency(self, percentile: float, where: str, params: list[Any]) -> float:
        # SQLite has no percentile function, so emulate it with ORDER BY/OFFSET.
        count = await self._db.fetch_value(f"SELECT COUNT(*) FROM request_logs {where}", params)
        total = int(count or 0)
        if total == 0:
            return 0.0
        offset = min(total - 1, max(0, int(total * percentile)))
        value = await self._db.fetch_value(
            f"""
            SELECT total_latency_ms FROM request_logs {where}
            ORDER BY total_latency_ms ASC LIMIT 1 OFFSET ?
            """,
            [*params, offset],
        )
        return round(float(value or 0.0), 3)

    async def _by_provider(self, where: str, params: list[Any]) -> list[ProviderUsageStats]:
        rows = await self._db.fetch_all(
            f"""
            SELECT provider,
                   COUNT(*)                              AS requests,
                   COALESCE(SUM(status != 'ok'), 0)      AS errors,
                   COALESCE(SUM(original_tokens), 0)     AS original_tokens,
                   COALESCE(SUM(compressed_tokens), 0)   AS compressed_tokens,
                   COALESCE(SUM(saved_tokens), 0)        AS saved_tokens,
                   COALESCE(AVG(saved_ratio), 0.0)       AS avg_saved_ratio,
                   COALESCE(AVG(total_latency_ms), 0.0)  AS avg_latency
            FROM request_logs
            {where}
            GROUP BY provider
            ORDER BY requests DESC
            """,
            params,
        )
        return [
            ProviderUsageStats(
                provider=str(row["provider"]),
                requests=int(row["requests"]),
                errors=int(row["errors"]),
                original_tokens=int(row["original_tokens"]),
                compressed_tokens=int(row["compressed_tokens"]),
                saved_tokens=int(row["saved_tokens"]),
                avg_saved_ratio=round(float(row["avg_saved_ratio"]), 6),
                avg_latency_ms=round(float(row["avg_latency"]), 3),
            )
            for row in rows
        ]

    async def _by_model(self, where: str, params: list[Any], limit: int) -> list[ModelUsageStats]:
        rows = await self._db.fetch_all(
            f"""
            SELECT provider, model,
                   COUNT(*)                            AS requests,
                   COALESCE(SUM(original_tokens), 0)   AS original_tokens,
                   COALESCE(SUM(compressed_tokens), 0) AS compressed_tokens,
                   COALESCE(SUM(saved_tokens), 0)      AS saved_tokens,
                   COALESCE(AVG(saved_ratio), 0.0)     AS avg_saved_ratio
            FROM request_logs
            {where}
            GROUP BY provider, model
            ORDER BY requests DESC
            LIMIT ?
            """,
            [*params, limit],
        )
        return [
            ModelUsageStats(
                provider=str(row["provider"]),
                model=str(row["model"]),
                requests=int(row["requests"]),
                original_tokens=int(row["original_tokens"]),
                compressed_tokens=int(row["compressed_tokens"]),
                saved_tokens=int(row["saved_tokens"]),
                avg_saved_ratio=round(float(row["avg_saved_ratio"]), 6),
            )
            for row in rows
        ]

    async def recent_requests(
        self, *, limit: int = 50, session_id: str | None = None
    ) -> list[RequestLogItem]:
        if not self._db.enabled:
            return []
        where, params = ("WHERE session_id = ?", [session_id]) if session_id else ("", [])
        rows = await self._db.fetch_all(
            f"""
            SELECT id, session_id, created_at, provider, model, strategy,
                   original_tokens, compressed_tokens, saved_tokens, saved_ratio,
                   status, error_type, total_latency_ms, upstream_latency_ms
            FROM request_logs
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            [*params, limit],
        )
        return [RequestLogItem(**dict(row)) for row in rows]
