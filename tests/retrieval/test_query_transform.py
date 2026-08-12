"""Tests for query_transform — HyDE and multi-query, both with mocked LLM calls."""

from unittest.mock import MagicMock

from rag_condominios.retrieval.query_transform import (
    generate_hyde_embedding,
    generate_multi_queries,
    transform_parallel,
)


def _groq_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _openai_embed_response(vector: list[float]) -> MagicMock:
    datum = MagicMock()
    datum.embedding = vector
    resp = MagicMock()
    resp.data = [datum]
    return resp


def test_generate_hyde_embedding_returns_vector() -> None:
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response("Doc hipotético aqui.")

    openai = MagicMock()
    openai.embeddings.create.return_value = _openai_embed_response([0.1, 0.2, 0.3])

    result = generate_hyde_embedding("Qual o prazo?", groq, openai)

    assert result == [0.1, 0.2, 0.3]
    groq.chat.completions.create.assert_called_once()
    openai.embeddings.create.assert_called_once()


def test_generate_hyde_embedding_uses_hypothetical_doc_as_input() -> None:
    """The OpenAI embed call must receive the HyDE doc text, not the raw query."""
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response("Texto hipotético gerado.")

    openai = MagicMock()
    openai.embeddings.create.return_value = _openai_embed_response([0.0])

    generate_hyde_embedding("pergunta original", groq, openai)

    call_kwargs = openai.embeddings.create.call_args
    input_text = call_kwargs[1].get("input") or call_kwargs[0][1]
    assert input_text == "Texto hipotético gerado."


def test_generate_multi_queries_returns_list_of_strings() -> None:
    raw = "1. Pergunta A\n2. Pergunta B\n3. Pergunta C"
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response(raw)

    result = generate_multi_queries("Qual o prazo?", groq)

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(q, str) for q in result)


def test_generate_multi_queries_strips_numbering() -> None:
    raw = "1. Pergunta A\n2. Pergunta B\n3. Pergunta C"
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response(raw)

    result = generate_multi_queries("q", groq)

    assert result[0] == "Pergunta A"
    assert result[1] == "Pergunta B"
    assert result[2] == "Pergunta C"


def test_generate_multi_queries_fallback_on_empty_response() -> None:
    groq = MagicMock()
    groq.chat.completions.create.return_value = _groq_response("")

    result = generate_multi_queries("original query", groq)

    assert result == ["original query"]


def test_transform_parallel_returns_embedding_and_queries() -> None:
    groq = MagicMock()
    # First call → HyDE doc; second call → multi-query
    groq.chat.completions.create.side_effect = [
        _groq_response("Doc hipotético."),
        _groq_response("1. Alt A\n2. Alt B\n3. Alt C"),
    ]

    openai = MagicMock()
    openai.embeddings.create.return_value = _openai_embed_response([1.0, 2.0])

    embedding, queries = transform_parallel("query?", groq, openai)

    assert embedding == [1.0, 2.0]
    assert len(queries) == 3
