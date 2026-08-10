"""Hybrid retrieval pipeline: dense + sparse + RRF fusion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from rag_condominios.retrieval.dense import embed_query, search_dense
from rag_condominios.retrieval.fusion import reciprocal_rank_fusion
from rag_condominios.retrieval.sparse import search_sparse

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
    bm25_retriever: Any,
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
