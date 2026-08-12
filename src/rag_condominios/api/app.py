"""FastAPI application factory with lifespan initialization."""

import logging
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from groq import Groq
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_condominios.api.routes import evaluate, health, ingest, metrics, query
from rag_condominios.api.state import AppState
from rag_condominios.core.config import settings
from rag_condominios.retrieval.pipeline import DEFAULT_BM25_PATH
from rag_condominios.retrieval.sparse import load_index

_log = logging.getLogger(__name__)

# Minimal pattern: at least one dot, no IP-like strings, no localhost.
_DOMAIN_RE = re.compile(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]*[a-z0-9])?)+$")
_PRIVATE_PREFIXES = ("localhost", "127.", "10.", "192.168.", "169.254.", "::1")


def _validate_allowed_domains(domains: list[str]) -> None:
    for domain in domains:
        d = domain.strip().lower()
        if not _DOMAIN_RE.match(d) or any(d.startswith(p) for p in _PRIVATE_PREFIXES):
            raise ValueError(
                f"ALLOWED_DOMAINS contains an invalid or private hostname: {domain!r}. "
                "Only public domain names are permitted."
            )


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

    try:
        _validate_allowed_domains(settings.allowed_domains_list)
    except ValueError as exc:
        _log.critical("STARTUP: %s", exc)

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
        try:
            retriever, chunk_ids = load_index(DEFAULT_BM25_PATH)
            state.bm25_retriever = retriever
            state.bm25_chunk_ids = chunk_ids
        except RuntimeError as exc:
            _log.error("BM25 index load failed — sparse retrieval disabled: %s", exc)

    if settings.qdrant_url and not settings.qdrant_api_key:
        _log.warning("STARTUP: QDRANT_URL is set but QDRANT_API_KEY is empty — may fail on authenticated clusters")

    # Initialize reranker once at startup to avoid per-request model loading.
    try:
        from rag_condominios.retrieval.reranker import MsMarcoReranker
        state.reranker = MsMarcoReranker()
    except Exception as exc:  # noqa: BLE001 — any model-load failure degrades gracefully
        _log.error("Reranker init failed — reranking disabled (pipeline will use rrf_score): %s", exc)

    # Pre-create the semantic cache collection so per-request calls skip this step.
    if state.qdrant_client is not None:
        try:
            from rag_condominios.retrieval.semantic_cache import ensure_cache_collection
            ensure_cache_collection(state.qdrant_client)
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            _log.warning("STARTUP: semantic cache collection init failed — cache disabled: %s", exc)

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
