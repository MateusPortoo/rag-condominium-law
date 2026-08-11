"""Generate embeddings via OpenAI text-embedding-3-small."""

from openai import OpenAI

from rag_condominios.core.config import EMBEDDING_MODEL

BATCH_SIZE = 100


class Embedder:
    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of BATCH_SIZE. Returns one vector per text."""
        all_embeddings: list[list[float]] = []
        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            response = self._client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings
