"""Unit tests for generator — mocks Groq client."""

from unittest.mock import MagicMock

from rag_condominios.retrieval.generator import build_context, generate
from rag_condominios.retrieval.pipeline import RetrievalResult


def _make_chunk(text: str, lei: str = "lei_4591", artigo: str = "Art. 1") -> RetrievalResult:
    return RetrievalResult(chunk_id="test", rrf_score=1.0, text=text, lei=lei, artigo=artigo)


def test_build_context_includes_text() -> None:
    chunks = [_make_chunk("O condômino deve pagar.")]
    context = build_context(chunks)
    assert "O condômino deve pagar." in context


def test_build_context_includes_article_header() -> None:
    chunks = [_make_chunk("Texto.", lei="codigo_civil", artigo="Art. 1.336")]
    context = build_context(chunks)
    assert "Art. 1.336" in context


def test_build_context_limits_to_max_chunks() -> None:
    chunks = [_make_chunk(f"Texto {i}", artigo=f"Art. {i}") for i in range(10)]
    context = build_context(chunks)
    # Only first 5 chunks should appear
    assert "Texto 0" in context
    assert "Texto 4" in context
    assert "Texto 5" not in context


def test_build_context_empty_chunks_returns_empty_string() -> None:
    assert build_context([]) == ""


def test_generate_calls_groq_and_returns_content() -> None:
    groq_client = MagicMock()
    groq_client.chat.completions.create.return_value.choices[0].message.content = (
        "O condômino deve contribuir."
    )
    chunks = [_make_chunk("Art. 1336: contribuir para despesas.")]

    result = generate("Quais são os deveres do condômino?", chunks, groq_client)

    assert result == "O condômino deve contribuir."
    groq_client.chat.completions.create.assert_called_once()


def test_generate_passes_temperature_zero() -> None:
    groq_client = MagicMock()
    groq_client.chat.completions.create.return_value.choices[0].message.content = "Resposta."
    chunks = [_make_chunk("contexto")]

    generate("pergunta", chunks, groq_client)

    call_kwargs = groq_client.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("temperature") == 0


def test_generate_empty_response_returns_empty_string() -> None:
    groq_client = MagicMock()
    groq_client.chat.completions.create.return_value.choices[0].message.content = None
    chunks = [_make_chunk("contexto")]

    result = generate("pergunta", chunks, groq_client)
    assert result == ""
