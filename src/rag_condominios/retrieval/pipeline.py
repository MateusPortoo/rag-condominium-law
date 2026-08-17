"""Hybrid retrieval pipeline: dense + sparse + RRF fusion + reranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

_log = logging.getLogger(__name__)

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from rag_condominios.core.config import COLLECTION_NAME
from rag_condominios.core.protocols import BM25Retriever
from rag_condominios.retrieval.dense import embed_query, search_dense
from rag_condominios.retrieval.fusion import reciprocal_rank_fusion
from rag_condominios.retrieval.sparse import search_sparse

if TYPE_CHECKING:
    from rag_condominios.retrieval.reranker import RankedResult

from rag_condominios.core.protocols import BaseCRAGEvaluator, BaseReranker

# Resolved relative to this file so the path is CWD-independent.
# pipeline.py lives at src/rag_condominios/retrieval/pipeline.py
# → 4 parents up → repo root → data/bm25_index  (directory, not a .pkl file)
DEFAULT_BM25_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "bm25_index"


@dataclass
class RetrievalResult:
    chunk_id: str
    rrf_score: float
    # Populated by the caller that has access to Qdrant payloads
    text: str = ""
    lei: str = ""
    artigo: str = ""


def retrieve(
    query: str,
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    bm25_retriever: BM25Retriever,
    bm25_chunk_ids: list[str],
    top_k: int = 20,
) -> list[RetrievalResult]:
    """
    Full hybrid retrieval:
      1. Embed query with OpenAI
      2. Dense search in Qdrant
      3. Sparse BM25 search in-memory
      4. RRF fusion
      5. Fetch payloads for top results
    """
    query_vector = embed_query(query, openai_client)

    dense_hits: list[ScoredPoint] = search_dense(query_vector, qdrant_client, top_k=top_k)
    sparse_hits: list[tuple[str, float]] = search_sparse(
        query, bm25_retriever, bm25_chunk_ids, top_k=top_k
    )

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, top_k=top_k)

    # Build a lookup from chunk_id → Qdrant payload for text hydration
    payload_by_id: dict[str, dict[str, Any]] = {}
    for point in dense_hits:
        if point.payload:
            cid = str(point.payload.get("chunk_id", point.id))
            payload_by_id[cid] = point.payload

    # Fetch payloads for chunk_ids that appear only in BM25 (not covered by dense results)
    bm25_only_ids = [cid for cid, _ in fused if cid not in payload_by_id]
    if bm25_only_ids:
        payload_by_id.update(fetch_bm25_only_payloads(qdrant_client, bm25_only_ids))

    results: list[RetrievalResult] = []
    for chunk_id, rrf_score in fused:
        payload = payload_by_id.get(chunk_id)
        if payload is None:
            _log.warning("No payload for chunk_id=%s after BM25-only lookup — dropping chunk", chunk_id)
            continue
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                rrf_score=rrf_score,
                text=payload.get("text", ""),
                lei=payload.get("lei", ""),
                artigo=payload.get("artigo", ""),
            )
        )
    return results


def fetch_bm25_only_payloads(
    qdrant_client: QdrantClient,
    bm25_only_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Scroll Qdrant to fetch payloads for chunk_ids not returned by any dense search.

    Returns a mapping of chunk_id → payload dict. Missing IDs are silently omitted.
    Non-fatal: on error, logs a warning and returns an empty dict.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    try:
        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="chunk_id", match=MatchAny(any=bm25_only_ids))]
            ),
            with_payload=True,
            limit=len(bm25_only_ids),
        )
        return {
            str(pt.payload.get("chunk_id", pt.id)): pt.payload
            for pt in points
            if pt.payload
        }
    except Exception as exc:  # noqa: BLE001 — payload fetch failure is non-fatal
        _log.warning("BM25-only payload fetch failed: %s", exc)
        return {}


def rerank_and_evaluate(
    query: str,
    results: list[RetrievalResult],
    reranker: BaseReranker | None = None,
    evaluator: BaseCRAGEvaluator | None = None,
) -> tuple[list[RankedResult], Literal["correct", "ambiguous", "incorrect"]]:
    """Rerank retrieved results and return a CRAG verdict.

    Open for extension: inject any BaseReranker/BaseCRAGEvaluator implementation
    without modifying this function — Dependency Inversion via Protocol.
    """
    from typing import cast

    from rag_condominios.core.config import settings as _settings
    from rag_condominios.retrieval.crag import CRAGEvaluator as _CRAGEvaluator
    from rag_condominios.retrieval.reranker import RankedResult as _RankedResult
    from rag_condominios.retrieval.reranker import create_reranker as _create_reranker

    if reranker is None:
        _log.warning("reranker not injected — instantiating %s as fallback", _settings.reranker_model)
    _reranker: BaseReranker = reranker or _create_reranker(_settings.reranker_model)
    _evaluator: BaseCRAGEvaluator = evaluator or _CRAGEvaluator(
        correct_threshold=_settings.crag_correct_threshold,
        ambiguous_threshold=_settings.crag_ambiguous_threshold,
    )
    ranked = cast(list[_RankedResult], _reranker.rerank(query, results))
    verdict = _evaluator.evaluate(ranked)
    return ranked, verdict
