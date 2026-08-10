"""Analyze token distribution of parsed articles to guide chunking decisions."""

import tiktoken

from rag_condominios.core.models import IngestResult

TIKTOKEN_ENCODING = "cl100k_base"


def _count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def _compute_percentile(sorted_values: list[int], percentile: float) -> int:
    """Return the value at a given percentile (0–100) from a sorted list."""
    if not sorted_values:
        return 0
    index = max(0, int(len(sorted_values) * percentile / 100) - 1)
    return sorted_values[index]


def _build_distribution(token_counts: list[int]) -> dict[str, int]:
    sorted_counts = sorted(token_counts)
    return {
        "p50": _compute_percentile(sorted_counts, 50),
        "p90": _compute_percentile(sorted_counts, 90),
        "p95": _compute_percentile(sorted_counts, 95),
        "max": sorted_counts[-1] if sorted_counts else 0,
    }


def _find_oversized_articles(articles: list[dict[str, str]], token_counts: list[int], chunk_size: int) -> list[str]:
    oversized = []
    for article, count in zip(articles, token_counts):
        if count > chunk_size:
            oversized.append(f"  {article['artigo']}, {article['lei']} — {count} tokens")
    return oversized


def _print_report(
    total: int,
    distribution: dict[str, int],
    chunk_size: int,
    covered: int,
    oversized_lines: list[str],
) -> None:
    print("\n=== Distribuição de Tamanho dos Artigos ===")
    print(f"Total de artigos: {total}\n")
    print("Percentis (tokens):")
    print(f"  p50: {distribution['p50']:4d}  |  p90: {distribution['p90']:4d}  |  p95: {distribution['p95']:4d}  |  max: {distribution['max']:4d}")
    coverage_pct = 100 * covered / total if total else 0.0
    print(f"\nArtigos ≤ {chunk_size} tokens: {covered}/{total} ({coverage_pct:.1f}%)")
    print(f"Artigos > {chunk_size} tokens:   {total - covered}/{total}  ({100 - coverage_pct:.1f}%)")
    if oversized_lines:
        print("\nArtigos que excedem o chunk size:")
        print("\n".join(oversized_lines))
    print(f"\n→ Chunk size {chunk_size} cobre {coverage_pct:.1f}% dos artigos sem truncar.")
    print(f"  Recomendação: MANTER {chunk_size} / AJUSTAR PARA X (preencher após análise real)\n")


def analyze_distribution(articles: list[dict[str, str]], chunk_size: int) -> IngestResult:
    """Compute token distribution stats and print a formatted report."""
    encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    token_counts = [_count_tokens(a["texto"], encoder) for a in articles]
    distribution = _build_distribution(token_counts)
    covered = sum(1 for c in token_counts if c <= chunk_size)
    oversized_lines = _find_oversized_articles(articles, token_counts, chunk_size)
    _print_report(len(articles), distribution, chunk_size, covered, oversized_lines)
    coverage_pct = 100.0 * covered / len(articles) if articles else 0.0
    return IngestResult(
        total_articles=len(articles),
        total_chunks=0,
        distribution=distribution,
        chunk_size_used=chunk_size,
        coverage_pct=round(coverage_pct, 2),
    )
