"""Hybrid retrieval pipeline: dense + sparse + RRF fusion + reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from rag_condominios.core.protocols import BM25Retriever
from rag_condominios.retrieval.dense import embed_query, search_dense
from rag_condominios.retrieval.fusion import reciprocal_rank_fusion
from rag_condominios.retrieval.sparse import search_sparse

if TYPE_CHECKING:
    from rag_condominios.retrieval.reranker import RankedResult

from rag_condominios.core.protocols import BaseCRAGEvaluator, BaseReranker

DEFAULT_BM25_PATH = Path("data/bm25_index.pkl")


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

    results: list[RetrievalResult] = []
    for chunk_id, rrf_score in fused:
        payload = payload_by_id.get(chunk_id, {})
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

    from rag_condominios.retrieval.crag import CRAGEvaluator as _CRAGEvaluator
    from rag_condominios.retrieval.reranker import MsMarcoReranker as _MsMarcoReranker, RankedResult as _RankedResult

    _reranker: BaseReranker = reranker or _MsMarcoReranker()
    _evaluator: BaseCRAGEvaluator = evaluator or _CRAGEvaluator()
    ranked = cast(list[_RankedResult], _reranker.rerank(query, results))
    verdict = _evaluator.evaluate(ranked)
    return ranked, verdict
