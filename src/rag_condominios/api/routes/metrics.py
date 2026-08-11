"""GET /metrics — last N query records."""

from fastapi import APIRouter, Query, Request

from rag_condominios.api.schemas import MetricsResponse
from rag_condominios.api.state import AppState

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request, last: int = Query(default=20, ge=1, le=100)) -> MetricsResponse:
    state: AppState = request.app.state.rag
    recent = state.recent_queries[-last:]
    return MetricsResponse(recent_queries=list(recent), total_count=len(state.recent_queries))
