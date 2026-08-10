"""Sparse retrieval via BM25 (in-memory, bm25s library)."""

import pickle
from pathlib import Path
from typing import Any

import bm25s  # type: ignore[import-untyped]

TOP_K = 20


def load_index(index_path: str | Path) -> tuple[Any, list[str]]:
    """Load the BM25 index and the corresponding chunk ID list from disk."""
    with open(index_path, "rb") as f:
        payload: dict[str, Any] = pickle.load(f)
    return payload["retriever"], payload["chunk_ids"]


def search_sparse(
    query: str,
    retriever: Any,
    chunk_ids: list[str],
    top_k: int = TOP_K,
) -> list[tuple[str, float]]:
    """Return (chunk_id, score) pairs sorted by BM25 relevance."""
    tokenized = bm25s.tokenize(query)
    results, scores = retriever.retrieve(tokenized, k=min(top_k, len(chunk_ids)))

    # results shape: (1, k) — we queried a single string
    ids = results[0].tolist()
    raw_scores = scores[0].tolist()

    # ids are integer indices into chunk_ids list
    return [(chunk_ids[int(idx)], float(score)) for idx, score in zip(ids, raw_scores)]
