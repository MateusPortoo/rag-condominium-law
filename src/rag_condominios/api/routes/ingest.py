"""POST /ingest - rebuild BM25 index from existing Qdrant data."""

import json
from typing import Any

import bm25s
from fastapi import APIRouter, HTTPException, Request

from rag_condominios.api.schemas import IngestResponse
from rag_condominios.api.state import AppState
from rag_condominios.core.config import COLLECTION_NAME
from rag_condominios.retrieval.pipeline import DEFAULT_BM25_PATH

router = APIRouter()

_SCROLL_LIMIT = 1000


def _rebuild_bm25(state: AppState) -> tuple[Any, list[str], int]:
    """Scroll Qdrant, rebuild BM25 in-memory, return (retriever, chunk_ids, count)."""
    if state.qdrant_client is None:
        raise RuntimeError("qdrant_client is None — called before initialization")

    texts: list[str] = []
    chunk_ids: list[str] = []
    offset: str | int | None = None

    while True:
        response = state.qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=_SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
        )
        points, next_offset = response
        for point in points:
            if point.payload:
                texts.append(str(point.payload.get("text", "")))
                chunk_ids.append(str(point.payload.get("chunk_id", point.id)))
        if next_offset is None:
            break
        offset = next_offset  # type: ignore[assignment]

    if not texts:
        raise HTTPException(status_code=404, detail="Qdrant collection is empty. Run run_ingest.py first.")

    tokenized = bm25s.tokenize(texts)
    retriever = bm25s.BM25()
    retriever.index(tokenized)

    DEFAULT_BM25_PATH.mkdir(parents=True, exist_ok=True)
    retriever.save(str(DEFAULT_BM25_PATH), show_progress=False)
    (DEFAULT_BM25_PATH / "chunk_ids.json").write_text(
        json.dumps(chunk_ids), encoding="utf-8"
    )

    return retriever, chunk_ids, len(texts)


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: Request) -> IngestResponse:
    state: AppState = request.app.state.rag
    if state.qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant nao inicializado.")

    retriever, chunk_ids, count = _rebuild_bm25(state)
    state.bm25_retriever = retriever
    state.bm25_chunk_ids = chunk_ids

    return IngestResponse(status="ok", chunks_indexed=count)
