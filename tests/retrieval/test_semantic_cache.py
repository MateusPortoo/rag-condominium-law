"""Tests for QdrantSemanticCache — Qdrant mocked to avoid network calls."""

from unittest.mock import MagicMock

from rag_condominios.retrieval.semantic_cache import CachedResponse, QdrantSemanticCache

_EMBEDDING = [0.1] * 1536
_THRESHOLD = 0.95


def _cache(qdrant: MagicMock) -> QdrantSemanticCache:
    return QdrantSemanticCache(qdrant, _THRESHOLD)


def _hit_point(score: float) -> MagicMock:
    point = MagicMock()
    point.score = score
    point.payload = {
        "question": "Qual o prazo?",
        "answer": "60 dias.",
        "crag_verdict": "correct",
        "sources": [],
    }
    return point


def test_lookup_returns_none_on_empty_results() -> None:
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = []
    assert _cache(qdrant).lookup(_EMBEDDING) is None


def test_lookup_returns_none_below_threshold() -> None:
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = [_hit_point(0.90)]
    assert _cache(qdrant).lookup(_EMBEDDING) is None


def test_lookup_returns_cached_response_above_threshold() -> None:
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = [_hit_point(0.97)]
    result = _cache(qdrant).lookup(_EMBEDDING)
    assert result is not None
    assert result.answer == "60 dias."
    assert result.crag_verdict == "correct"


def test_lookup_returns_none_at_exact_threshold() -> None:
    qdrant = MagicMock()
    qdrant.query_points.return_value.points = [_hit_point(0.95)]
    # score >= threshold is a hit (0.95 >= 0.95)
    result = _cache(qdrant).lookup(_EMBEDDING)
    assert result is not None


def test_store_calls_upsert_once() -> None:
    qdrant = MagicMock()
    cache = _cache(qdrant)
    response = CachedResponse(
        question="Qual o prazo?",
        answer="60 dias.",
        crag_verdict="correct",
        sources=[],
    )
    cache.store(_EMBEDDING, response)
    qdrant.upsert.assert_called_once()


def test_store_payload_contains_question_and_answer() -> None:
    qdrant = MagicMock()
    cache = _cache(qdrant)
    response = CachedResponse(
        question="Qual o prazo?",
        answer="60 dias.",
        crag_verdict="correct",
        sources=[{"artigo": "Art. 1.350"}],
    )
    cache.store(_EMBEDDING, response)
    call_kwargs = qdrant.upsert.call_args
    points = call_kwargs.kwargs["points"]
    assert len(points) == 1
    payload = points[0].payload
    assert payload["question"] == "Qual o prazo?"
    assert payload["answer"] == "60 dias."
    assert payload["crag_verdict"] == "correct"
