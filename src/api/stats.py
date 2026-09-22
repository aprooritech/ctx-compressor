from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from src.api.deps import AuthDep, StateDep
from src.schemas import RequestLogItem, StatsResponse

router = APIRouter(tags=["observability"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    state: StateDep,
    _: AuthDep,
    window_hours: Annotated[
        float | None,
        Query(gt=0, description="Only aggregate the last N hours (default: all time)."),
    ] = None,
    model_limit: Annotated[
        int, Query(ge=1, le=200, description="Maximum number of per-model rows.")
    ] = 20,
) -> StatsResponse:
    return await state.metrics.aggregate(window_hours=window_hours, model_limit=model_limit)


@router.get("/stats/requests", response_model=list[RequestLogItem])
async def get_recent_requests(
    state: StateDep,
    _: AuthDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    session_id: Annotated[str | None, Query(description="Filter by session id.")] = None,
) -> list[RequestLogItem]:
    return await state.metrics.recent_requests(limit=limit, session_id=session_id)
