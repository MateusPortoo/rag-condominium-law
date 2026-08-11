"""Unit tests for API Pydantic schemas."""

import pytest
from pydantic import ValidationError

from rag_condominios.api.schemas import (
    EvaluateResponse,
    HealthResponse,
    MetricsEntry,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)


def test_query_request_valid() -> None:
    req = QueryRequest(question="Qual o prazo?")
    assert req.question == "Qual o prazo?"


def test_query_request_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_rejects_over_limit() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="a" * 2001)


def test_query_response_defaults() -> None:
    resp = QueryResponse(
        answer="Texto",
        crag_verdict="correct",
        sources=[],
        model_used="llama-3.3-70b-versatile",
        latency_ms=100,
    )
    assert resp.cached is False
    assert resp.web_search_timeout is False
    assert resp.query_transformed is None


def test_source_item_fields() -> None:
    src = SourceItem(chunk="trecho", score=0.82, artigo="Art. 22")
    assert src.artigo == "Art. 22"


def test_health_response_valid() -> None:
    h = HealthResponse(qdrant="ok", openai_key_present=True, groq_key_present=True, bm25_loaded=True)
    assert h.qdrant == "ok"


def test_health_response_error_qdrant() -> None:
    h = HealthResponse(qdrant="error", openai_key_present=False, groq_key_present=False, bm25_loaded=False)
    assert h.qdrant == "error"


def test_metrics_response_structure() -> None:
    entry = MetricsEntry(
        question="Q?", crag_verdict="correct", latency_ms=200, model_used="llama", cached=False
    )
    resp = MetricsResponse(recent_queries=[entry], total_count=1)
    assert resp.total_count == 1
    assert len(resp.recent_queries) == 1


def test_evaluate_response_passed_flag() -> None:
    r = EvaluateResponse(
        context_precision=0.80,
        context_recall=0.80,
        faithfulness=0.80,
        answer_relevancy=0.80,
        n_cases=10,
        passed=True,
        report_path="data/eval_report.json",
    )
    assert r.passed is True
