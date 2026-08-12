"""Mutable application state shared across requests via app.state.rag."""

import threading
from dataclasses import dataclass, field
from typing import Any

from groq import Groq
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_condominios.api.schemas import MetricsEntry

_MAX_METRICS = 100


@dataclass
class AppState:
    openai_client: OpenAI | None = None
    groq_client: Groq | None = None
    qdrant_client: QdrantClient | None = None
    bm25_retriever: Any = None
    bm25_chunk_ids: list[str] = field(default_factory=list)
    recent_queries: list[MetricsEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def is_ready(self) -> bool:
        return (
            self.openai_client is not None
            and self.groq_client is not None
            and self.qdrant_client is not None
            and self.bm25_retriever is not None
        )

    def record_query(self, entry: MetricsEntry) -> None:
        with self._lock:
            self.recent_queries.append(entry)
            if len(self.recent_queries) > _MAX_METRICS:
                self.recent_queries.pop(0)
