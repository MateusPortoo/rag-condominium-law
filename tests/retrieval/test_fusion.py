"""Unit tests for RRF fusion — no external dependencies needed."""

from unittest.mock import MagicMock

from qdrant_client.models import ScoredPoint

from rag_condominios.retrieval.fusion import reciprocal_rank_fusion

RRF_K = 60


def _make_point(chunk_id: str, score: float) -> ScoredPoint:
    point = MagicMock(spec=ScoredPoint)
    point.id = chunk_id
    point.score = score
    point.payload = {"chunk_id": chunk_id}
    return point


def test_rrf_document_in_both_lists_scores_higher() -> None:
    """A chunk appearing in both dense and sparse rankings should outscore one appearing in only one."""
    dense = [_make_point("chunk_a", 0.9), _make_point("chunk_b", 0.8)]
    sparse = [("chunk_a", 5.0), ("chunk_c", 4.0)]

    results = reciprocal_rank_fusion(dense, sparse)
    ids = [r[0] for r in results]

    # chunk_a appears in both lists — must rank first
    assert ids[0] == "chunk_a"


def test_rrf_score_formula_correct() -> None:
    """Verify that RRF score = 1/(k+rank_dense) + 1/(k+rank_sparse) for a shared doc."""
    dense = [_make_point("chunk_x", 0.95)]
    sparse = [("chunk_x", 10.0)]

    results = reciprocal_rank_fusion(dense, sparse, k=RRF_K)
    chunk_x_score = dict(results)["chunk_x"]

    expected = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 1)
    assert abs(chunk_x_score - expected) < 1e-9


def test_rrf_top_k_limits_output() -> None:
    dense = [_make_point(f"chunk_{i}", float(10 - i)) for i in range(15)]
    sparse = [(f"chunk_{i}", float(10 - i)) for i in range(15)]

    results = reciprocal_rank_fusion(dense, sparse, top_k=5)
    assert len(results) <= 5


def test_rrf_unique_from_one_list_included() -> None:
    """Chunks that appear in only one list should still be returned."""
    dense = [_make_point("only_dense", 0.9)]
    sparse = [("only_sparse", 5.0)]

    results = reciprocal_rank_fusion(dense, sparse)
    ids = [r[0] for r in results]

    assert "only_dense" in ids
    assert "only_sparse" in ids


def test_rrf_empty_lists_return_empty() -> None:
    results = reciprocal_rank_fusion([], [])
    assert results == []


def test_rrf_scores_are_positive() -> None:
    dense = [_make_point("chunk_1", 0.8), _make_point("chunk_2", 0.7)]
    sparse = [("chunk_3", 3.0)]

    results = reciprocal_rank_fusion(dense, sparse)
    for _, score in results:
        assert score > 0
