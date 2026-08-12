"""Semantic cache backed by Qdrant — SPEC section 8.

Single responsibility: cache query embeddings and responses so near-duplicate
questions are served instantly without re-running the full pipeline.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

_log = logging.getLogger(__name__)
_VALID_VERDICTS = frozenset({"correct", "ambiguous", "incorrect"})

from rag_condominios.core.config import SEMANTIC_CACHE_COLLECTION

_EMBEDDING_DIM = 1536
_CACHE_LIMIT = 1


@dataclass
class CachedResponse:
    """Payload stored and returned by the semantic cache."""

    question: str
    answer: str
    crag_verdict: str
    sources: list[dict[str, Any]]


def ensure_cache_collection(qdrant: QdrantClient) -> None:
    """Create the query_cache collection if it does not already exist."""
    existing = {c.name for c in qdrant.get_collections().collections}
    if SEMANTIC_CACHE_COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
        )


class QdrantSemanticCache:
    """Semantic cache backed by a dedicated Qdrant collection.

    Lookup uses cosine similarity; responses above the threshold are returned
    immediately. Store inserts the embedding + payload as a new point.
    """

    def __init__(self, qdrant: QdrantClient, threshold: float) -> None:
        self._qdrant = qdrant
        self._threshold = threshold

    def lookup(self, query_embedding: list[float]) -> CachedResponse | None:
        """Search the cache; return a CachedResponse if a match is found."""
        results = self._qdrant.query_points(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            query=query_embedding,
            limit=_CACHE_LIMIT,
            with_payload=True,
        )
        if not results.points:
            return None

        hit = results.points[0]
        if (hit.score or 0.0) < self._threshold:
            return None

        payload = hit.payload or {}
        answer = payload.get("answer", "")
        verdict = payload.get("crag_verdict", "")
        if not answer or verdict not in _VALID_VERDICTS:
            _log.warning(
                "Malformed cache entry (answer=%r, verdict=%r) — discarding",
                answer[:30] if answer else "",
                verdict,
            )
            return None
        return CachedResponse(
            question=payload.get("question", ""),
            answer=answer,
            crag_verdict=verdict,
            sources=payload.get("sources", []),
        )

    def store(self, query_embedding: list[float], response: CachedResponse) -> None:
        """Insert the response into the cache collection."""
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=query_embedding,
            payload={
                "question": response.question,
                "answer": response.answer,
                "crag_verdict": response.crag_verdict,
                "sources": response.sources,
            },
        )
        self._qdrant.upsert(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            points=[point],
        )
