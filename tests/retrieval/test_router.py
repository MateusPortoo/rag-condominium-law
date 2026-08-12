"""Tests for classify_query — no mocking needed, pure function."""

from rag_condominios.retrieval.router import classify_query


def test_short_query_is_simple() -> None:
    assert classify_query("Qual o prazo?") == "simple"


def test_long_query_without_keywords_is_simple() -> None:
    assert classify_query("Qual o prazo para convocar a assembleia ordinária anual") == "simple"


def test_query_with_compare_keyword_is_complex() -> None:
    assert classify_query("compare as obrigações do síndico e do subsíndico") == "complex"


def test_query_with_diferenca_keyword_is_complex() -> None:
    assert classify_query("qual a diferença entre assembleia ordinária e extraordinária") == "complex"


def test_query_with_todos_os_casos_is_complex() -> None:
    assert classify_query("quais são todos os casos em que o síndico pode ser destituído") == "complex"


def test_query_with_explique_detalhadamente_is_complex() -> None:
    assert classify_query("explique detalhadamente as regras de uso do salão de festas") == "complex"


def test_exactly_eight_tokens_is_simple() -> None:
    # 8 tokens exactly → len < 8 is False, no keyword → simple
    assert classify_query("um dois três quatro cinco seis sete oito") == "simple"


def test_seven_tokens_is_simple() -> None:
    # 7 tokens → len < 8 → simple regardless of content
    assert classify_query("compare um dois três quatro cinco seis") == "simple"
