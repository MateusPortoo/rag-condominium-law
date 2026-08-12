"""Sparse retrieval via BM25 (in-memory, bm25s library)."""

import logging
import pickle
from pathlib import Path
from typing import Any

import bm25s  # type: ignore[import-untyped]

from rag_condominios.core.config import TOP_K_DEFAULT

_log = logging.getLogger(__name__)


def load_index(index_path: str | Path) -> tuple[Any, list[str]]:
    """Load the BM25 index and the corresponding chunk ID list from disk."""
    try:
        with open(index_path, "rb") as f:
            payload: dict[str, Any] = pickle.load(f)
        return payload["retriever"], payload["chunk_ids"]
    except (pickle.UnpicklingError, KeyError, EOFError, OSError) as exc:
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

    # results shape: (1, k) — we queried a single string
    ids = results[0].tolist()
    raw_scores = scores[0].tolist()

    pairs: list[tuple[str, float]] = []
    for idx, score in zip(ids, raw_scores):
        i = int(idx)
        if i >= len(chunk_ids):
            _log.warning(
                "BM25 returned out-of-range index %d (chunk_ids len=%d) — skipping stale entry",
                i, len(chunk_ids),
            )
            continue
        pairs.append((chunk_ids[i], float(score)))
    return pairs
