"""Integration tests for GET /health and GET /metrics."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from rag_condominios.api.app import create_app
from rag_condominios.api.state import AppState


def test_health_all_none_returns_200() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.rag = AppState()  # all-None regardless of env vars
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["qdrant"] == "error"
    assert body["openai_key_present"] is False
    assert body["groq_key_present"] is False
    assert body["bm25_loaded"] is False


def test_health_qdrant_ok_when_collection_exists() -> None:
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.return_value = MagicMock()
    state = AppState(
        openai_client=MagicMock(),
        groq_client=MagicMock(),
        qdrant_client=mock_qdrant,
        bm25_retriever=MagicMock(),
    )
    app = create_app()
    with TestClient(app) as client:
        app.state.rag = state
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["qdrant"] == "ok"
    assert body["openai_key_present"] is True
    assert body["groq_key_present"] is True
    assert body["bm25_loaded"] is True


def test_health_qdrant_error_when_collection_raises() -> None:
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.side_effect = Exception("connection refused")
    state = AppState(qdrant_client=mock_qdrant)
    app = create_app()
    with TestClient(app) as client:
        app.state.rag = state
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["qdrant"] == "error"


def test_metrics_returns_empty_on_start() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["recent_queries"] == []
    assert body["total_count"] == 0
