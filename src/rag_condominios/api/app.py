"""FastAPI application factory with lifespan initialization."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from groq import Groq
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_condominios.api.routes import evaluate, health, ingest, metrics, query
from rag_condominios.api.state import AppState
from rag_condominios.core.config import settings
from rag_condominios.retrieval.pipeline import DEFAULT_BM25_PATH
from rag_condominios.retrieval.sparse import load_index

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    missing = [
        name
        for name, val in [
            ("OPENAI_API_KEY", settings.openai_api_key),
            ("GROQ_API_KEY", settings.groq_api_key),
            ("QDRANT_URL", settings.qdrant_url),
        ]
        if not val
    ]
    if missing:
        _log.critical(
            "STARTUP: required env vars not set: %s — all API requests will return HTTP 503",
            ", ".join(missing),
        )

    state = AppState(
        openai_client=OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None,
        groq_client=Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None,
        qdrant_client=(
            QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            if settings.qdrant_url
            else None
        ),
    )
    if DEFAULT_BM25_PATH.exists():
        retriever, chunk_ids = load_index(DEFAULT_BM25_PATH)
        state.bm25_retriever = retriever
        state.bm25_chunk_ids = chunk_ids

    app.state.rag = state
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Condomínios API",
        description="Retrieval-Augmented Generation sobre legislação condominial brasileira.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(query.router)
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(metrics.router)
    app.include_router(evaluate.router)
    return app


app = create_app()
