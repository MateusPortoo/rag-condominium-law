"""POST /ingest - rebuild BM25 index from existing Qdrant data."""

import json
import logging
import shutil
from typing import Any

import bm25s
from fastapi import APIRouter, Header, HTTPException, Request

from rag_condominios.api.schemas import IngestResponse
from rag_condominios.api.state import AppState
from rag_condominios.core.config import COLLECTION_NAME, settings
from rag_condominios.retrieval.pipeline import DEFAULT_BM25_PATH

router = APIRouter()
_log = logging.getLogger(__name__)

_SCROLL_LIMIT = 1000


def _rebuild_bm25(state: AppState) -> tuple[Any, list[str], int]:
    """Scroll Qdrant, rebuild BM25 in-memory, return (retriever, chunk_ids, count).

    Raises ValueError when the collection is empty (not HTTPException — callers
    that are not HTTP handlers should not receive HTTP-layer exceptions).
    """
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
                text = str(point.payload.get("text", "")).strip()
                if text:
                    texts.append(text)
                    chunk_ids.append(str(point.payload.get("chunk_id", point.id)))
        if next_offset is None:
            break
        offset = next_offset  # type: ignore[assignment]

    if not texts:
        raise ValueError("Qdrant collection is empty. Run run_ingest.py first.")

    tokenized = bm25s.tokenize(texts)
    retriever = bm25s.BM25()
    retriever.index(tokenized)

    tmp_path = DEFAULT_BM25_PATH.parent / (DEFAULT_BM25_PATH.name + ".tmp")
    try:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True)
        retriever.save(str(tmp_path), show_progress=False)
        (tmp_path / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")
        if DEFAULT_BM25_PATH.exists():
            shutil.rmtree(DEFAULT_BM25_PATH)
        tmp_path.rename(DEFAULT_BM25_PATH)
    except Exception:
        _log.exception("BM25 atomic save failed — cleaning up tmp_path=%s", tmp_path)
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    return retriever, chunk_ids, len(texts)


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> IngestResponse:
    if settings.ingest_api_key and x_api_key != settings.ingest_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized.")

    state: AppState = request.app.state.rag
    if state.qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant nao inicializado.")

    try:
        retriever, chunk_ids, count = _rebuild_bm25(state)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    with state._lock:
        state.bm25_retriever = retriever
        state.bm25_chunk_ids = chunk_ids

    return IngestResponse(status="ok", chunks_indexed=count)
