"""Dense retrieval via OpenAI embeddings + Qdrant vector search."""

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from rag_condominios.core.config import COLLECTION_NAME, EMBEDDING_MODEL, TOP_K_DEFAULT


def embed_query(query: str, client: OpenAI) -> list[float]:
    """Embed a single query string using OpenAI."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


def search_dense(
    query_vector: list[float],
    qdrant: QdrantClient,
    top_k: int = TOP_K_DEFAULT,
) -> list[ScoredPoint]:
    """Search Qdrant with a dense vector and return top_k scored points."""
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return response.points
