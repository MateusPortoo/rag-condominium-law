"""Index chunks into Qdrant (dense) and bm25s (sparse) atomically."""

import json
import shutil
from pathlib import Path

import bm25s
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_condominios.core.config import COLLECTION_NAME
from rag_condominios.core.models import ArticleChunk

VECTOR_SIZE = 1536


class Indexer:
    def __init__(self, qdrant_url: str, qdrant_api_key: str) -> None:
        self._qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._bm25: bm25s.BM25 | None = None
        self._chunk_ids: list[str] = []

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
        valid = [(chunk.id, chunk.text.strip()) for chunk in chunks if chunk.text.strip()]
        ids, corpus = (list(x) for x in zip(*valid)) if valid else ([], [])
        tokenized = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        self._chunk_ids = ids
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

        # BM25 built after Qdrant succeeds â€” if this raises, Qdrant data is present
        # but BM25 is not committed to disk (save_bm25 must be called separately).
        try:
            self._bm25 = self._build_bm25_index(chunks)
        except Exception as exc:
            raise RuntimeError(
                f"BM25 index construction failed after Qdrant upsert succeeded. "
                f"Re-run ingest to restore consistency. Cause: {exc}"
            ) from exc

    def save_bm25(self, path: str) -> None:
        """Save BM25 index atomically: write to a temp dir, then rename."""
        if self._bm25 is None:
            raise RuntimeError("No BM25 index in memory. Run index() before save_bm25().")
        save_dir = Path(path)
        tmp_dir = save_dir.parent / (save_dir.name + ".tmp")
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            self._bm25.save(str(tmp_dir), show_progress=False)
            (tmp_dir / "chunk_ids.json").write_text(
                json.dumps(self._chunk_ids), encoding="utf-8"
            )
            if save_dir.exists():
                shutil.rmtree(save_dir)
            tmp_dir.rename(save_dir)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

