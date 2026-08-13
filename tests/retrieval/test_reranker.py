"""Tests for MsMarcoReranker, BgeReranker and create_reranker factory.

CrossEncoder is mocked to avoid model download in CI.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_condominios.core.config import RERANKER_MODEL_BGE, RERANKER_MODEL_MSMARCO
from rag_condominios.retrieval.pipeline import RetrievalResult
from rag_condominios.retrieval.reranker import (
    BgeReranker,
    MsMarcoReranker,
    RankedResult,
    create_reranker,
)


def _make_result(chunk_id: str, text: str = "texto") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, rrf_score=0.5, text=text, lei="Lei 4591", artigo="Art. 1")


# ---------------------------------------------------------------------------
# MsMarcoReranker
# ---------------------------------------------------------------------------

@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_reranker_returns_ranked_results_with_scores(mock_ce_class: MagicMock) -> None:
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.9, 0.5])
    mock_ce_class.return_value = mock_ce

    reranker = MsMarcoReranker()
    results = [_make_result("c1"), _make_result("c2")]
    ranked = reranker.rerank("query", results)

    assert len(ranked) == 2
    assert all(isinstance(r, RankedResult) for r in ranked)
    assert all(hasattr(r, "rerank_score") for r in ranked)


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_reranker_orders_by_score_descending(mock_ce_class: MagicMock) -> None:
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.3, 0.9, 0.6])
    mock_ce_class.return_value = mock_ce

    reranker = MsMarcoReranker()
    results = [_make_result("c1"), _make_result("c2"), _make_result("c3")]
    ranked = reranker.rerank("query", results)

    scores = [r.rerank_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_reranker_empty_input_returns_empty(mock_ce_class: MagicMock) -> None:
    mock_ce_class.return_value = MagicMock()
    reranker = MsMarcoReranker()
    assert reranker.rerank("query", []) == []


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_ranked_result_preserves_original_fields(mock_ce_class: MagicMock) -> None:
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.8])
    mock_ce_class.return_value = mock_ce

    original = _make_result("chunk-42", text="Artigo 22 da lei")
    original.lei = "Lei 4591"
    original.artigo = "Art. 22"

    reranker = MsMarcoReranker()
    ranked = reranker.rerank("quorum", [original])

    assert ranked[0].chunk_id == "chunk-42"
    assert ranked[0].text == "Artigo 22 da lei"
    assert ranked[0].lei == "Lei 4591"
    assert ranked[0].artigo == "Art. 22"
    assert ranked[0].rrf_score == pytest.approx(0.5)


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_msmarco_loads_correct_model(mock_ce_class: MagicMock) -> None:
    mock_ce_class.return_value = MagicMock()
    MsMarcoReranker()
    mock_ce_class.assert_called_once_with(RERANKER_MODEL_MSMARCO)


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_reranker_degrades_to_rrf_score_on_predict_failure(mock_ce_class: MagicMock) -> None:
    mock_ce = MagicMock()
    mock_ce.predict.side_effect = RuntimeError("OOM")
    mock_ce_class.return_value = mock_ce

    reranker = MsMarcoReranker()
    results = [_make_result("c1"), _make_result("c2")]
    ranked = reranker.rerank("query", results)

    assert len(ranked) == 2
    for r in ranked:
        assert r.rerank_score == r.rrf_score


# ---------------------------------------------------------------------------
# BgeReranker — DC-01 multilingual replacement
# ---------------------------------------------------------------------------

@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_bge_reranker_loads_correct_model(mock_ce_class: MagicMock) -> None:
    mock_ce_class.return_value = MagicMock()
    BgeReranker()
    mock_ce_class.assert_called_once_with(RERANKER_MODEL_BGE)


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_bge_reranker_returns_ranked_results(mock_ce_class: MagicMock) -> None:
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.85, 0.40])
    mock_ce_class.return_value = mock_ce

    reranker = BgeReranker()
    results = [_make_result("c1"), _make_result("c2")]
    ranked = reranker.rerank("síndico convocar assembleia", results)

    assert len(ranked) == 2
    assert ranked[0].rerank_score == pytest.approx(0.85)
    assert ranked[1].rerank_score == pytest.approx(0.40)


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_bge_reranker_orders_by_score_descending(mock_ce_class: MagicMock) -> None:
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.2, 0.95, 0.55])
    mock_ce_class.return_value = mock_ce

    reranker = BgeReranker()
    results = [_make_result("c1"), _make_result("c2"), _make_result("c3")]
    ranked = reranker.rerank("query", results)

    scores = [r.rerank_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_bge_reranker_empty_input_returns_empty(mock_ce_class: MagicMock) -> None:
    mock_ce_class.return_value = MagicMock()
    assert BgeReranker().rerank("query", []) == []


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_bge_reranker_degrades_to_rrf_score_on_failure(mock_ce_class: MagicMock) -> None:
    mock_ce = MagicMock()
    mock_ce.predict.side_effect = RuntimeError("CUDA OOM")
    mock_ce_class.return_value = mock_ce

    reranker = BgeReranker()
    results = [_make_result("c1")]
    ranked = reranker.rerank("query", results)

    assert len(ranked) == 1
    assert ranked[0].rerank_score == ranked[0].rrf_score


# ---------------------------------------------------------------------------
# create_reranker factory
# ---------------------------------------------------------------------------

@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_factory_returns_msmarco_for_msmarco_model(mock_ce_class: MagicMock) -> None:
    mock_ce_class.return_value = MagicMock()
    reranker = create_reranker(RERANKER_MODEL_MSMARCO)
    assert isinstance(reranker, MsMarcoReranker)


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_factory_returns_bge_for_bge_model(mock_ce_class: MagicMock) -> None:
    mock_ce_class.return_value = MagicMock()
    reranker = create_reranker(RERANKER_MODEL_BGE)
    assert isinstance(reranker, BgeReranker)


@patch("rag_condominios.retrieval.reranker.CrossEncoder")
def test_factory_returns_msmarco_for_unknown_model(mock_ce_class: MagicMock) -> None:
    mock_ce_class.return_value = MagicMock()
    reranker = create_reranker("some/unknown-model")
    assert isinstance(reranker, MsMarcoReranker)
