"""Reciprocal Rank Fusion (RRF) — combines dense and sparse result lists."""

from qdrant_client.models import ScoredPoint

RRF_K = 60
TOP_K = 20


def reciprocal_rank_fusion(
    dense_results: list[ScoredPoint],
    sparse_results: list[tuple[str, float]],
    k: int = RRF_K,
    top_k: int = TOP_K,
) -> list[tuple[str, float]]:
    """
    Fuse dense and sparse rankings using RRF.

    RRF score for a document d = sum over each ranked list L of: 1 / (k + rank_in_L(d))

    k=60 is the standard default from the original RRF paper (Cormack et al. 2009).
    Higher k makes the fusion smoother (less sensitive to exact rank positions).
    """
    scores: dict[str, float] = {}

    # Dense results: extract chunk_id from payload
    for rank, point in enumerate(dense_results, start=1):
        chunk_id = str(point.payload.get("chunk_id", point.id)) if point.payload else str(point.id)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    # Sparse results: already (chunk_id, bm25_score) pairs
    for rank, (chunk_id, _bm25_score) in enumerate(sparse_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_ids[:top_k]
