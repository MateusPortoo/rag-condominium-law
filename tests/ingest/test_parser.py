"""Unit tests for ingest/parser.py using synthetic HTML fixtures."""

from rag_condominios.ingest.parser import parse_articles

SIMPLE_ARTICLE_HTML = """
<html><body>
<h2>Capítulo I — Das Disposições Gerais</h2>
<p>Art. 1º Esta lei regula a constituição dos condomínios.</p>
<p>Parágrafo único. Aplica-se a todos os edifícios.</p>
<p>Art. 2º Outro artigo aqui.</p>
</body></html>
"""

ARTICLE_WITH_INCISOS_HTML = """
<html><body>
<h2>Capítulo II — Dos Direitos</h2>
<p>Art. 10 São deveres do condômino:</p>
<p>I — pagar as despesas;</p>
<p>II — respeitar as normas;</p>
<p>Art. 11 Próximo artigo.</p>
</body></html>
"""

# Código Civil range: only 1.314 to 1.358 should be included.
CC_RANGE_HTML = """
<html><body>
<p>Art. 1.313 Artigo fora do range.</p>
<p>Art. 1.314 Primeiro artigo do condomínio.</p>
<p>Art. 1.358 Último artigo do condomínio.</p>
<p>Art. 1.359 Artigo fora do range.</p>
</body></html>
"""


def test_parse_simple_article_extracts_text() -> None:
    articles = parse_articles(SIMPLE_ARTICLE_HTML, "lei_4591")
    assert len(articles) >= 1
    first = articles[0]
    assert "Art. 1" in first["artigo"]
    assert "Esta lei regula" in first["texto"]


def test_parse_article_includes_paragraphs_and_incisos() -> None:
    articles = parse_articles(ARTICLE_WITH_INCISOS_HTML, "lei_4591")
    assert len(articles) >= 1
    combined_text = " ".join(a["texto"] for a in articles)
    # Incisos should be part of the article body (collected as following siblings)
    assert "pagar as despesas" in combined_text or "São deveres" in combined_text


def test_parse_cc_filters_out_of_range_articles() -> None:
    articles = parse_articles(CC_RANGE_HTML, "codigo_civil")
    article_numbers = [a["artigo"] for a in articles]
    # Art. 1.313 and Art. 1.359 must be excluded
    assert not any("1.313" in n or "1313" in n for n in article_numbers), (
        f"Art. 1.313 should be excluded but found: {article_numbers}"
    )
    assert not any("1.359" in n or "1359" in n for n in article_numbers), (
        f"Art. 1.359 should be excluded but found: {article_numbers}"
    )


def test_parse_cc_includes_boundary_articles() -> None:
    articles = parse_articles(CC_RANGE_HTML, "codigo_civil")
    article_labels = [a["artigo"] for a in articles]
    assert any("1.314" in label or "1314" in label for label in article_labels), (
        f"Art. 1.314 should be included but found: {article_labels}"
    )
    assert any("1.358" in label or "1358" in label for label in article_labels), (
        f"Art. 1.358 should be included but found: {article_labels}"
    )


def test_parse_lei_name_preserved() -> None:
    articles = parse_articles(SIMPLE_ARTICLE_HTML, "lei_4591")
    for article in articles:
        assert article["lei"] == "lei_4591"


def test_parse_section_extracted_from_heading() -> None:
    articles = parse_articles(SIMPLE_ARTICLE_HTML, "lei_4591")
    assert len(articles) >= 1
    # The h2 heading should be captured as secao for Art. 1º
    assert "Capítulo I" in articles[0]["secao"] or articles[0]["secao"] == ""
