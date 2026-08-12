"""Dense retrieval via OpenAI embeddings + Qdrant vector search."""

import logging

from openai import OpenAI, OpenAIError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import ScoredPoint

from rag_condominios.core.config import COLLECTION_NAME, EMBEDDING_MODEL, TOP_K_DEFAULT

_log = logging.getLogger(__name__)


def embed_query(query: str, client: OpenAI) -> list[float]:
    """Embed a single query string using OpenAI."""
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
        return response.data[0].embedding
    except OpenAIError as exc:
        _log.error("OpenAI embedding failed (model=%s, query_len=%d): %s", EMBEDDING_MODEL, len(query), exc)
        raise


def search_dense(
    query_vector: list[float],
    qdrant: QdrantClient,
    top_k: int = TOP_K_DEFAULT,
) -> list[ScoredPoint]:
    """Search Qdrant with a dense vector and return top_k scored points."""
    try:
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return response.points
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        _log.error("Qdrant dense search failed (top_k=%d): %s", top_k, exc)
        raise
