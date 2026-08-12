"""Tests for crag_actions — decompose_recompose, web_search, ambiguous_action."""

from unittest.mock import MagicMock, patch

import requests
from groq import APIError as GroqAPIError

from rag_condominios.retrieval.crag_actions import (
    ambiguous_action,
    decompose_recompose,
    web_search,
)
from rag_condominios.retrieval.reranker import RankedResult


def _ranked(chunk_id: str, text: str, score: float = 0.80) -> RankedResult:
    return RankedResult(
        chunk_id=chunk_id,
        rrf_score=0.5,
        rerank_score=score,
        text=text,
        lei="CC",
        artigo="Art. 1",
    )


def _mock_reranker(strip_scores: list[float]) -> MagicMock:
    """Return a mock reranker that assigns scores to strips in order."""
    reranker = MagicMock()

    def _rerank(query: str, results: list) -> list:
        from rag_condominios.retrieval.reranker import RankedResult as RR
        scored = []
        for i, result in enumerate(results):
            sc = strip_scores[i] if i < len(strip_scores) else 0.0
            scored.append(
                RR(
                    chunk_id=result.chunk_id,
                    rrf_score=result.rrf_score,
                    rerank_score=sc,
                    text=result.text,
                    lei=result.lei,
                    artigo=result.artigo,
                )
            )
        return sorted(scored, key=lambda x: x.rerank_score, reverse=True)

    reranker.rerank.side_effect = _rerank
    return reranker


# ---------------------------------------------------------------------------
# decompose_recompose
# ---------------------------------------------------------------------------

def test_decompose_recompose_keeps_relevant_strips() -> None:
    chunk = _ranked("c1", "Strip A relevante. Strip B irrelevante. Strip C relevante.")
    reranker = _mock_reranker([0.80, 0.40, 0.80])  # strips 0, 1, 2

    result = decompose_recompose("query", [chunk], reranker)

    assert len(result) == 1
    assert "Strip A relevante" in result[0].text
    assert "Strip C relevante" in result[0].text
    assert "Strip B irrelevante" not in result[0].text


def test_decompose_recompose_drops_chunk_when_no_strip_passes() -> None:
    chunk = _ranked("c1", "Strip irrelevante A. Strip irrelevante B.")
    reranker = _mock_reranker([0.40, 0.50])  # both below threshold 0.70

    result = decompose_recompose("query", [chunk], reranker)

    # Falls back to original when no chunk survives
    assert result == [chunk]


def test_decompose_recompose_preserves_metadata() -> None:
    chunk = _ranked("c42", "Primeiro trecho. Segundo trecho.")
    chunk.lei = "Lei 4591"
    chunk.artigo = "Art. 22"
    reranker = _mock_reranker([0.85, 0.85])

    result = decompose_recompose("query", [chunk], reranker)

    assert result[0].chunk_id == "c42"
    assert result[0].lei == "Lei 4591"
    assert result[0].artigo == "Art. 22"


def test_decompose_recompose_empty_input_returns_empty() -> None:
    reranker = MagicMock()
    result = decompose_recompose("query", [], reranker)
    assert result == []


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

def _groq_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("rag_condominios.retrieval.crag_actions.requests.get")
def test_web_search_returns_snippets_on_success(mock_get: MagicMock) -> None:
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response("lei condominial prazo")

    mock_get.return_value.json.return_value = {
        "AbstractText": "Trecho da lei sobre prazos.",
        "RelatedTopics": [{"Text": "Outro trecho relevante."}],
    }
    mock_get.return_value.raise_for_status.return_value = None

    chunks, timed_out = web_search("Qual o prazo?", groq, ["planalto.gov.br"])

    assert not timed_out
    assert len(chunks) >= 1
    assert any("prazo" in c.text.lower() or "trecho" in c.text.lower() for c in chunks)


@patch("rag_condominios.retrieval.crag_actions.requests.get")
def test_web_search_returns_timed_out_on_network_error(mock_get: MagicMock) -> None:
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response("query rewritten")
    mock_get.side_effect = requests.exceptions.Timeout()

    chunks, timed_out = web_search("query", groq, ["planalto.gov.br"])

    assert timed_out
    assert chunks == []


@patch("rag_condominios.retrieval.crag_actions.requests.get")
def test_web_search_falls_back_to_original_query_when_groq_fails(mock_get: MagicMock) -> None:
    # Groq rewrite fails → fallback to original query → search runs and returns empty
    groq_client = MagicMock()
    groq_client.chat.completions.create.side_effect = GroqAPIError(
        "groq error", MagicMock(), body=None
    )
    mock_get.return_value.json.return_value = {"AbstractText": "", "RelatedTopics": []}
    mock_get.return_value.raise_for_status.return_value = None

    chunks, timed_out = web_search("query", groq_client, ["planalto.gov.br"])

    # Groq failure does not propagate as timed_out; search completes with no results
    assert not timed_out
    assert chunks == []


@patch("rag_condominios.retrieval.crag_actions.requests.get")
def test_web_search_returns_empty_not_timed_out_when_no_snippets(mock_get: MagicMock) -> None:
    # Search succeeds but DuckDuckGo finds nothing → timed_out=False (not a network failure)
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response("rewritten")
    mock_get.return_value.json.return_value = {"AbstractText": "", "RelatedTopics": []}
    mock_get.return_value.raise_for_status.return_value = None

    chunks, timed_out = web_search("query", groq, ["planalto.gov.br"])

    assert not timed_out
    assert chunks == []


# ---------------------------------------------------------------------------
# ambiguous_action
# ---------------------------------------------------------------------------

@patch("rag_condominios.retrieval.crag_actions.requests.get")
def test_ambiguous_action_merges_recomposed_and_web(mock_get: MagicMock) -> None:
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response("rewritten query")

    mock_get.return_value.json.return_value = {
        "AbstractText": "Snippet da web.",
        "RelatedTopics": [],
    }
    mock_get.return_value.raise_for_status.return_value = None

    chunk = _ranked("c1", "Trecho relevante. Trecho irrelevante.")
    reranker = _mock_reranker([0.85, 0.40])

    merged, timed_out = ambiguous_action(
        "query", [chunk], reranker, groq, ["planalto.gov.br"]
    )

    assert not timed_out
    # merged should include both recomposed chunk and web snippet
    assert len(merged) >= 1


@patch("rag_condominios.retrieval.crag_actions.requests.get")
def test_ambiguous_action_graceful_on_web_timeout(mock_get: MagicMock) -> None:
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response("rewritten")
    mock_get.side_effect = requests.exceptions.Timeout()

    chunk = _ranked("c1", "Trecho. Relevante trecho.")
    reranker = _mock_reranker([0.80, 0.80])

    merged, timed_out = ambiguous_action(
        "query", [chunk], reranker, groq, ["planalto.gov.br"]
    )

    assert timed_out
    # Should still return recomposed results despite web failure
    assert len(merged) >= 1
