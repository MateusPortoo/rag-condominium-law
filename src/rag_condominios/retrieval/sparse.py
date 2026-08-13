"""Sparse retrieval via BM25 (in-memory, bm25s library)."""

import json
import logging
from pathlib import Path
from typing import Any

import bm25s

from rag_condominios.core.config import TOP_K_DEFAULT

_log = logging.getLogger(__name__)


def load_index(index_path: str | Path) -> tuple[Any, list[str]]:
    """Load the BM25 index and chunk ID list from a directory saved by bm25s.

    The index directory contains numpy arrays + JSON files written by
    bm25s.BM25.save(). chunk_ids.json holds the ordered list of chunk IDs
    that maps BM25 result positions back to Qdrant chunk identifiers.
    """
    index_path = Path(index_path)
    try:
        retriever = bm25s.BM25.load(str(index_path), load_corpus=False, show_progress=False)
        chunk_ids: list[str] = json.loads(
            (index_path / "chunk_ids.json").read_text(encoding="utf-8")
        )
        return retriever, chunk_ids
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"BM25 index at {index_path!r} is corrupt or invalid: {exc}") from exc


def search_sparse(
    query: str,
    retriever: Any,
    chunk_ids: list[str],
    top_k: int = TOP_K_DEFAULT,
) -> list[tuple[str, float]]:
    """Return (chunk_id, score) pairs sorted by BM25 relevance."""
    tokenized = bm25s.tokenize(query)
    results, scores = retriever.retrieve(tokenized, k=min(top_k, len(chunk_ids)))

    # results shape: (1, k) â€” we queried a single string
    ids = results[0].tolist()
    raw_scores = scores[0].tolist()

    pairs: list[tuple[str, float]] = []
    for idx, score in zip(ids, raw_scores):
        i = int(idx)
        if i >= len(chunk_ids):
            _log.warning(
                "BM25 returned out-of-range index %d (chunk_ids len=%d) â€” skipping stale entry",
                i, len(chunk_ids),
            )
            continue
        pairs.append((chunk_ids[i], float(score)))
    return pairs

