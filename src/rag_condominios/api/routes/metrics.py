"""GET /metrics — last N query records."""

from fastapi import APIRouter, Query, Request

from rag_condominios.api.schemas import MetricsResponse
from rag_condominios.api.state import AppState

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request, last: int = Query(default=20, ge=1, le=100)) -> MetricsResponse:
    state: AppState = request.app.state.rag
    all_queries = list(state.recent_queries)
    return MetricsResponse(recent_queries=all_queries[-last:], total_count=len(all_queries))
