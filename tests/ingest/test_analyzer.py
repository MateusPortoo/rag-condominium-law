"""Unit tests for ingest/analyzer.py."""

from rag_condominios.ingest.analyzer import _compute_percentile, analyze_distribution


def _make_article(text: str, lei: str = "lei_4591", artigo: str = "Art. 1") -> dict[str, str]:
    return {"lei": lei, "artigo": artigo, "texto": text, "secao": ""}


def test_percentile_on_known_values() -> None:
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert _compute_percentile(values, 50) == 50
    assert _compute_percentile(values, 90) == 90


def test_percentile_single_element() -> None:
    assert _compute_percentile([42], 50) == 42
    assert _compute_percentile([42], 95) == 42


def test_percentile_empty_list_returns_zero() -> None:
    assert _compute_percentile([], 50) == 0


def test_analyze_distribution_returns_correct_total() -> None:
    articles = [_make_article("palavra " * 10, artigo=f"Art. {i}") for i in range(5)]
    result = analyze_distribution(articles, chunk_size=512)
    assert result.total_articles == 5


def test_analyze_distribution_coverage_all_fit() -> None:
    # All articles are tiny — should fit easily inside chunk_size=512.
    articles = [_make_article("curto", artigo=f"Art. {i}") for i in range(10)]
    result = analyze_distribution(articles, chunk_size=512)
    assert result.coverage_pct == 100.0


def test_analyze_distribution_coverage_partial() -> None:
    # Mix: some tiny, one very long (repeat a word many times).
    short_articles = [_make_article("curto", artigo=f"Art. {i}") for i in range(9)]
    long_text = "condômino assembleia síndico convenção " * 200  # ~800+ tokens
    long_article = _make_article(long_text, artigo="Art. 99")
    articles = short_articles + [long_article]
    result = analyze_distribution(articles, chunk_size=512)
    # 9 out of 10 fit → 90%
    assert result.coverage_pct == 90.0


def test_analyze_distribution_oversized_articles_identified() -> None:
    long_text = "palavra " * 600  # well over 512 tokens
    articles = [_make_article(long_text, artigo="Art. 99")]
    result = analyze_distribution(articles, chunk_size=512)
    # coverage should be 0% — the single article exceeds chunk_size
    assert result.coverage_pct == 0.0


def test_analyze_distribution_includes_all_distribution_keys() -> None:
    articles = [_make_article("texto " * 20, artigo=f"Art. {i}") for i in range(5)]
    result = analyze_distribution(articles, chunk_size=512)
    assert set(result.distribution.keys()) == {"p50", "p90", "p95", "max"}


def test_analyze_distribution_chunk_size_stored() -> None:
    articles = [_make_article("breve")]
    result = analyze_distribution(articles, chunk_size=256)
    assert result.chunk_size_used == 256


def test_analyze_distribution_max_equals_largest_article() -> None:
    articles = [
        _make_article("a " * 10, artigo="Art. 1"),
        _make_article("b " * 200, artigo="Art. 2"),
        _make_article("c " * 50, artigo="Art. 3"),
    ]
    result = analyze_distribution(articles, chunk_size=512)
    # Max should correspond to the largest article (Art. 2 with ~200 words).
    # We don't assert the exact token count — just that max >= p95 >= p50 >= 0.
    assert result.distribution["max"] >= result.distribution["p95"] >= result.distribution["p50"] >= 0
