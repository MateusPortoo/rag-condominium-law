#!/usr/bin/env python
"""CLI script that runs the full ingest pipeline end to end."""

import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rag_condominios.core.config import settings
from rag_condominios.ingest import downloader, parser, analyzer, chunker, embedder, indexer

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 77
BM25_INDEX_PATH = "data/bm25_index.pkl"


def _ask_confirm_chunk_size(chunk_size: int) -> int:
    answer = input(f"Manter chunk_size={chunk_size}? (s/n): ").strip().lower()
    if answer == "s":
        return chunk_size
    try:
        new_size = int(input("Novo chunk_size (tokens): ").strip())
        return new_size
    except ValueError:
        print("Entrada inválida. Mantendo chunk_size original.")
        return chunk_size


def main() -> None:
    start_time = time.perf_counter()

    print("\n[1/7] Baixando corpus...")
    corpus = downloader.download_corpus()

    print("\n[2/7] Parseando artigos...")
    all_articles: list[dict[str, str]] = []
    for lei_name, html in corpus.items():
        articles = parser.parse_articles(html, lei_name)
        print(f"  {lei_name}: {len(articles)} artigos")
        all_articles.extend(articles)

    print(f"\n[3/7] Analisando distribuição ({len(all_articles)} artigos)...")
    analysis = analyzer.analyze_distribution(all_articles, DEFAULT_CHUNK_SIZE)
    chunk_size = _ask_confirm_chunk_size(analysis.chunk_size_used)

    print(f"\n[4/7] Chunking com chunk_size={chunk_size}, overlap={DEFAULT_OVERLAP}...")
    chunks = chunker.chunk_articles(all_articles, chunk_size=chunk_size, overlap=DEFAULT_OVERLAP)
    print(f"  Gerados {len(chunks)} chunks.")

    print("\n[5/7] Gerando embeddings...")
    embed = embedder.Embedder(api_key=settings.openai_api_key)
    texts = [c.text for c in chunks]
    embeddings = embed.embed_batch(texts)
    print(f"  Embeddings gerados: {len(embeddings)}")

    print("\n[6/7] Indexando no Qdrant + BM25...")
    idx = indexer.Indexer(qdrant_url=settings.qdrant_url, qdrant_api_key=settings.qdrant_api_key)
    idx.index(chunks, embeddings)

    print(f"\n[7/7] Salvando índice BM25 em {BM25_INDEX_PATH}...")
    idx.save_bm25(BM25_INDEX_PATH)

    elapsed = time.perf_counter() - start_time
    print(f"\n✓ Ingest concluído em {elapsed:.1f}s")
    print(f"  Chunks indexados : {len(chunks)}")
    print(f"  Artigos parseados: {len(all_articles)}")


if __name__ == "__main__":
    main()
