"""GET /health — liveness + dependency status."""

from fastapi import APIRouter, Request

from rag_condominios.api.schemas import HealthResponse
from rag_condominios.api.state import AppState
from rag_condominios.ingest.indexer import COLLECTION_NAME

router = APIRouter()


def _qdrant_status(state: AppState) -> str:
    if state.qdrant_client is None:
        return "error"
    try:
        state.qdrant_client.get_collection(COLLECTION_NAME)
        return "ok"
    except Exception:  # noqa: BLE001 — health check must never raise, any Qdrant error maps to "error"
        return "error"


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state: AppState = request.app.state.rag
    return HealthResponse(
        qdrant=_qdrant_status(state),  # type: ignore[arg-type]
        openai_key_present=state.openai_client is not None,
        groq_key_present=state.groq_client is not None,
        bm25_loaded=state.bm25_retriever is not None,
    )
