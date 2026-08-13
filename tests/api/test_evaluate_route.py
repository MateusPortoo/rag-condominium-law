"""Integration tests for POST /evaluate using TestClient with mocked pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_condominios.api.app import create_app
from rag_condominios.api.state import AppState
from rag_condominios.eval.evaluator import EvalReport


def _ready_state() -> AppState:
    return AppState(
        openai_client=MagicMock(),
        groq_client=MagicMock(),
        qdrant_client=MagicMock(),
        bm25_retriever=MagicMock(),
        bm25_chunk_ids=["chunk-001"],
    )


def _fake_report(n_cases: int = 3) -> EvalReport:
    return EvalReport(
        context_precision=0.82,
        context_recall=0.79,
        faithfulness=0.88,
        answer_relevancy=0.75,
        n_cases=n_cases,
        verdict_distribution={"correct": n_cases},
    )


_SAMPLE_CASES = [
    {
        "id": "gs-001",
        "category": "simple",
        "question": "Q1",
        "reference_answer": "A1",
        "reference_articles": [],
        "expected_crag_verdict": "correct",
    },
    {
        "id": "gs-002",
        "category": "multi_chunk",
        "question": "Q2",
        "reference_answer": "A2",
        "reference_articles": [],
        "expected_crag_verdict": "correct",
    },
]


def test_evaluate_503_when_uninitialized() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/evaluate")
    assert resp.status_code == 503


def test_evaluate_503_when_golden_set_missing() -> None:
    app = create_app()
    with (
        patch(
            "rag_condominios.api.routes.evaluate.GOLDEN_SET_PATH",
            Path("/nonexistent/golden_set.json"),
        ),
        TestClient(app) as client,
    ):
        app.state.rag = _ready_state()
        resp = client.post("/evaluate")

    assert resp.status_code == 503
    assert "Golden set not found" in resp.json()["detail"]


def test_evaluate_422_when_no_evaluable_cases() -> None:
    app = create_app()

    with (
        patch("rag_condominios.api.routes.evaluate.GOLDEN_SET_PATH") as mock_path,
        patch("rag_condominios.api.routes.evaluate.evaluable_cases", return_value=[]),
        TestClient(app) as client,
    ):
        mock_path.exists.return_value = True
        app.state.rag = _ready_state()
        resp = client.post("/evaluate")

    assert resp.status_code == 422
    assert "No evaluable cases" in resp.json()["detail"]


def test_evaluate_503_when_all_cases_fail() -> None:
    app = create_app()

    zero_report = EvalReport(
        context_precision=0.0,
        context_recall=0.0,
        faithfulness=0.0,
        answer_relevancy=0.0,
        n_cases=0,
    )

    with (
        patch("rag_condominios.api.routes.evaluate.GOLDEN_SET_PATH") as mock_path,
        patch(
            "rag_condominios.api.routes.evaluate.evaluable_cases",
            return_value=_SAMPLE_CASES,
        ),
        patch(
            "rag_condominios.api.routes.evaluate.evaluate_golden_set",
            return_value=zero_report,
        ),
        TestClient(app) as client,
    ):
        mock_path.exists.return_value = True
        app.state.rag = _ready_state()
        resp = client.post("/evaluate")

    assert resp.status_code == 503
    assert "All evaluation cases failed" in resp.json()["detail"]


def test_evaluate_returns_report_and_saves_file() -> None:
    app = create_app()
    report = _fake_report(n_cases=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "eval_report.json"

        with (
            patch("rag_condominios.api.routes.evaluate.GOLDEN_SET_PATH") as mock_path,
            patch(
                "rag_condominios.api.routes.evaluate.evaluable_cases",
                return_value=_SAMPLE_CASES,
            ),
            patch(
                "rag_condominios.api.routes.evaluate.evaluate_golden_set",
                return_value=report,
            ),
            patch("rag_condominios.api.routes.evaluate._REPORT_PATH", report_path),
            TestClient(app) as client,
        ):
            mock_path.exists.return_value = True
            app.state.rag = _ready_state()
            resp = client.post("/evaluate")

        assert resp.status_code == 200
        body = resp.json()
        assert body["n_cases"] == 2
        assert body["context_precision"] == pytest.approx(0.82)
        assert body["context_recall"] == pytest.approx(0.79)
        assert body["faithfulness"] == pytest.approx(0.88)
        assert body["answer_relevancy"] == pytest.approx(0.75)
        assert isinstance(body["passed"], bool)
        assert "report_path" in body

        assert report_path.exists()
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        assert saved["n_cases"] == 2
        assert saved["verdict_distribution"] == {"correct": 2}


def test_evaluate_passes_reranker_from_state() -> None:
    app = create_app()
    report = _fake_report()
    mock_evaluate = MagicMock(return_value=report)
    fake_reranker = MagicMock()

    with (
        patch("rag_condominios.api.routes.evaluate.GOLDEN_SET_PATH") as mock_path,
        patch(
            "rag_condominios.api.routes.evaluate.evaluable_cases",
            return_value=_SAMPLE_CASES,
        ),
        patch("rag_condominios.api.routes.evaluate.evaluate_golden_set", mock_evaluate),
        patch("rag_condominios.api.routes.evaluate._REPORT_PATH", Path("/dev/null")),
        TestClient(app) as client,
    ):
        mock_path.exists.return_value = True
        state = _ready_state()
        state.reranker = fake_reranker
        app.state.rag = state
        client.post("/evaluate")

    _, kwargs = mock_evaluate.call_args
    assert kwargs.get("reranker") is fake_reranker


def test_evaluate_tolerates_report_write_failure() -> None:
    """If the report cannot be written to disk, the endpoint still returns 200."""
    app = create_app()
    report = _fake_report()
    unwritable = Path("/root/cannot_write/eval_report.json")

    with (
        patch("rag_condominios.api.routes.evaluate.GOLDEN_SET_PATH") as mock_path,
        patch(
            "rag_condominios.api.routes.evaluate.evaluable_cases",
            return_value=_SAMPLE_CASES,
        ),
        patch(
            "rag_condominios.api.routes.evaluate.evaluate_golden_set",
            return_value=report,
        ),
        patch("rag_condominios.api.routes.evaluate._REPORT_PATH", unwritable),
        TestClient(app) as client,
    ):
        mock_path.exists.return_value = True
        app.state.rag = _ready_state()
        resp = client.post("/evaluate")

    assert resp.status_code == 200
