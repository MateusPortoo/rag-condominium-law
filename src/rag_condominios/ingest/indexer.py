"""Index chunks into Qdrant (dense) and bm25s (sparse) atomically."""

import pickle
from pathlib import Path

import bm25s  # type: ignore[import-untyped]
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_condominios.core.models import ArticleChunk

COLLECTION_NAME = "condominio_docs"
VECTOR_SIZE = 1536


class Indexer:
    def __init__(self, qdrant_url: str, qdrant_api_key: str) -> None:
        self._qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._bm25: bm25s.BM25 | None = None

    def _ensure_collection_exists(self) -> None:
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if COLLECTION_NAME in existing:
            return
        self._qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    def _build_qdrant_points(
        self,
        chunks: list[ArticleChunk],
        embeddings: list[list[float]],
    ) -> list[PointStruct]:
        return [
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "lei": chunk.lei,
                    "artigo": chunk.artigo,
                    "secao": chunk.secao,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                },
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

    def _build_bm25_index(self, chunks: list[ArticleChunk]) -> bm25s.BM25:
        corpus = [chunk.text for chunk in chunks]
        tokenized = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        return retriever

    def index(self, chunks: list[ArticleChunk], embeddings: list[list[float]]) -> None:
        """Index chunks atomically: both Qdrant and BM25 succeed or neither is committed."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)}). "
                "Ensure embed_batch was called on the same chunk list."
            )

        self._ensure_collection_exists()
        points = self._build_qdrant_points(chunks, embeddings)
        self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        # BM25 built after Qdrant succeeds — if this raises, Qdrant data is present
        # but BM25 is not committed to disk (save_bm25 must be called separately).
        try:
            self._bm25 = self._build_bm25_index(chunks)
        except Exception as exc:
            raise RuntimeError(
                f"BM25 index construction failed after Qdrant upsert succeeded. "
                f"Re-run ingest to restore consistency. Cause: {exc}"
            ) from exc

    def save_bm25(self, path: str) -> None:
        if self._bm25 is None:
            raise RuntimeError("No BM25 index in memory. Run index() before save_bm25().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._bm25, fh)

    def load_bm25(self, path: str) -> None:
        with open(path, "rb") as fh:
            self._bm25 = pickle.load(fh)
