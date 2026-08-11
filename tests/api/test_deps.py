"""Tests for extract_clients — ensures no assert-based guards remain."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from rag_condominios.api.deps import extract_clients
from rag_condominios.api.state import AppState


def _full_state() -> AppState:
    return AppState(
        openai_client=MagicMock(),
        groq_client=MagicMock(),
        qdrant_client=MagicMock(),
    )


def test_extract_clients_returns_tuple_when_all_present() -> None:
    state = _full_state()
    openai, groq, qdrant = extract_clients(state)
    assert openai is state.openai_client
    assert groq is state.groq_client
    assert qdrant is state.qdrant_client


def test_extract_clients_raises_503_when_openai_missing() -> None:
    state = _full_state()
    state.openai_client = None
    with pytest.raises(HTTPException) as exc:
        extract_clients(state)
    assert exc.value.status_code == 503


def test_extract_clients_raises_503_when_groq_missing() -> None:
    state = _full_state()
    state.groq_client = None
    with pytest.raises(HTTPException) as exc:
        extract_clients(state)
    assert exc.value.status_code == 503


def test_extract_clients_raises_503_when_qdrant_missing() -> None:
    state = _full_state()
    state.qdrant_client = None
    with pytest.raises(HTTPException) as exc:
        extract_clients(state)
    assert exc.value.status_code == 503


def test_extract_clients_raises_503_when_all_missing() -> None:
    state = AppState()
    with pytest.raises(HTTPException) as exc:
        extract_clients(state)
    assert exc.value.status_code == 503
