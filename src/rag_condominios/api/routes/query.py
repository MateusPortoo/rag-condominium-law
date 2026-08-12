"""POST /query — sync and SSE streaming variants."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from groq import APIError as GroqAPIError
from groq import Groq
from openai import OpenAI, OpenAIError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_condominios.api.deps import extract_clients
from rag_condominios.api.schemas import (
    MetricsEntry,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from rag_condominios.api.security import detect_injection
from rag_condominios.api.state import AppState
from rag_condominios.core.config import settings
from rag_condominios.retrieval.crag_actions import (
    ambiguous_action,
    decompose_recompose,
    web_search,
)
from rag_condominios.retrieval.generator import GROQ_MODEL, generate, generate_stream
from rag_condominios.retrieval.pipeline import (
    RetrievalResult,
    rerank_and_evaluate,
    retrieve,
)
from rag_condominios.retrieval.query_transform import transform_parallel
from rag_condominios.retrieval.reranker import RankedResult
from rag_condominios.retrieval.router import ModelTier, classify_query
from rag_condominios.retrieval.semantic_cache import (
    CachedResponse,
    QdrantSemanticCache,
)

_log = logging.getLogger(__name__)

router = APIRouter()


def _sources(ranked: list[RankedResult]) -> list[SourceItem]:
    return [
        SourceItem(chunk=r.text, score=round(r.rerank_score, 4), artigo=r.artigo)
        for r in ranked[:5]
    ]


def _sources_dicts(ranked: list[RankedResult]) -> list[dict[str, object]]:
    return [
        {"chunk": r.text, "score": round(r.rerank_score, 4), "artigo": r.artigo}
        for r in ranked[:5]
    ]


def _pick_model(
    tier: ModelTier, verdict: str, openai_client: OpenAI, groq_client: Groq
) -> tuple[Any, str]:
    """Return (llm_client, model_name) based on tier and CRAG verdict."""
    if verdict in ("ambiguous", "incorrect") or tier == "complex":
        frontier = settings.openai_frontier_model or GROQ_MODEL
        if settings.openai_frontier_model:
            return openai_client, frontier
        return groq_client, GROQ_MODEL
    return groq_client, GROQ_MODEL


def _retrieve_with_transforms(
    question: str,
    state: AppState,
    openai_client: OpenAI,
    groq_client: Groq,
    qdrant_client: QdrantClient,
) -> list[RetrievalResult]:
    """Run original + HyDE + multi-query retrieval and fuse via RRF."""
    from rag_condominios.retrieval.dense import embed_query, search_dense
    from rag_condominios.retrieval.fusion import reciprocal_rank_fusion
    from rag_condominios.retrieval.sparse import search_sparse

    hyde_emb, alt_queries = transform_parallel(question, groq_client, openai_client)

    # Original dense + sparse
    orig_vector = embed_query(question, openai_client)
    orig_dense = search_dense(orig_vector, qdrant_client)
    orig_sparse = search_sparse(question, state.bm25_retriever, state.bm25_chunk_ids)

    # HyDE dense (skip if embedding is empty — HyDE failed gracefully)
    hyde_dense = search_dense(hyde_emb, qdrant_client) if hyde_emb else []

    # Multi-query dense results
    multi_dense_all = []
    for alt in alt_queries:
        alt_vec = embed_query(alt, openai_client)
        multi_dense_all.extend(search_dense(alt_vec, qdrant_client))

    # RRF: fuse all dense sources + original sparse in one pass
    all_dense = orig_dense + hyde_dense + multi_dense_all
    final_fused = reciprocal_rank_fusion(all_dense, orig_sparse)

    payload_by_id: dict[str, dict[str, Any]] = {}
    for point in all_dense:
        if point.payload:
            cid = str(point.payload.get("chunk_id", point.id))
            payload_by_id[cid] = point.payload

    results: list[RetrievalResult] = []
    for chunk_id, rrf_score in final_fused:
        payload = payload_by_id.get(chunk_id)
        if payload is None:
            _log.warning("No payload for chunk_id=%s (BM25-only hit) — text will be empty", chunk_id)
            payload = {}
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                rrf_score=rrf_score,
                text=payload.get("text", ""),
                lei=payload.get("lei", ""),
                artigo=payload.get("artigo", ""),
            )
        )
    return results


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: Request,
    body: QueryRequest,
    stream: bool = Query(default=False),
) -> QueryResponse | StreamingResponse:
    if detect_injection(body.question):
        _log.warning("injection attempt detected (question_len=%d)", len(body.question))
        raise HTTPException(status_code=400, detail="Query inválida.")

    state: AppState = request.app.state.rag
    openai_client, groq_client, qdrant_client = extract_clients(state)

    args = (body.question, state, openai_client, groq_client, qdrant_client)
    if stream:
        return _build_stream(*args)
    return _sync_answer(*args)


# ---------------------------------------------------------------------------
# Sync path
# ---------------------------------------------------------------------------

def _sync_answer(
    question: str,
    state: AppState,
    openai_client: OpenAI,
    groq_client: Groq,
    qdrant_client: QdrantClient,
) -> QueryResponse:
    t0 = time.monotonic()

    # 1. Semantic cache lookup
    from rag_condominios.retrieval.dense import embed_query
    try:
        query_embedding = embed_query(question, openai_client)
    except OpenAIError as exc:
        _log.error("embedding failed for sync query: %s", exc)
        raise HTTPException(status_code=503, detail="Embedding service unavailable.")

    cache: QdrantSemanticCache | None = None
    cached: CachedResponse | None = None
    try:
        cache = QdrantSemanticCache(qdrant_client, settings.semantic_cache_threshold)
        cached = cache.lookup(query_embedding)
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        _log.warning("semantic cache unavailable: %s", exc)

    if cached is not None:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return QueryResponse(
            answer=cached.answer,
            crag_verdict=cached.crag_verdict,  # type: ignore[arg-type]
            sources=[
                SourceItem(
                    chunk=s.get("chunk", ""),
                    score=s.get("score", 0.0),
                    artigo=s.get("artigo", ""),
                )
                for s in cached.sources
            ],
            model_used="cached",
            cached=True,
            web_search_timeout=False,
            latency_ms=latency_ms,
            query_transformed=None,
        )

    # 2. Query transformations + retrieval
    query_transformed: str | None = None
    try:
        results = _retrieve_with_transforms(
            question, state, openai_client, groq_client, qdrant_client
        )
        query_transformed = "hyde+multi"
    except (GroqAPIError, OpenAIError, UnexpectedResponse, ResponseHandlingException) as exc:
        _log.warning("query transforms failed, falling back to base retrieval: %s", exc)
        results = retrieve(
            query=question,
            openai_client=openai_client,
            qdrant_client=qdrant_client,
            bm25_retriever=state.bm25_retriever,
            bm25_chunk_ids=state.bm25_chunk_ids,
        )

    # 3. Rerank + CRAG evaluate
    ranked, verdict = rerank_and_evaluate(question, results, reranker=state.reranker)

    # 4. CRAG action based on verdict
    web_search_timed_out = False
    final_chunks: list[Any] = list(ranked)
    reranker = state.reranker

    if verdict == "correct" and reranker is not None:
        final_chunks = list(decompose_recompose(question, ranked, reranker))

    elif verdict == "incorrect":
        web_chunks, web_search_timed_out = web_search(
            question, groq_client, settings.allowed_domains_list
        )
        if web_chunks:
            final_chunks = list(web_chunks)

    elif verdict == "ambiguous" and reranker is not None:
        merged, web_search_timed_out = ambiguous_action(
            question, ranked, reranker, groq_client, settings.allowed_domains_list
        )
        final_chunks = list(merged)

    # 5. Model routing
    tier = classify_query(question)
    llm_client, model_name = _pick_model(tier, verdict, openai_client, groq_client)

    # 6. Generate answer
    try:
        answer = generate(question, final_chunks, llm_client, model_name)
    except (GroqAPIError, OpenAIError) as exc:
        _log.error("answer generation failed (model=%s): %s", model_name, exc)
        raise HTTPException(status_code=503, detail="LLM service unavailable.")

    if not answer:
        _log.error("generate() returned empty content (model=%s)", model_name)
        raise HTTPException(status_code=503, detail="LLM returned empty response.")

    latency_ms = int((time.monotonic() - t0) * 1000)

    sources_list = _sources(ranked)
    sources_dicts = _sources_dicts(ranked)

    # 7. Store in semantic cache
    if cache is not None:
        try:
            cache.store(
                query_embedding,
                CachedResponse(
                    question=question,
                    answer=answer,
                    crag_verdict=verdict,
                    sources=sources_dicts,
                ),
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            _log.warning("semantic cache store failed: %s", exc)

    state.record_query(
        MetricsEntry(
            question=question,
            crag_verdict=verdict,
            latency_ms=latency_ms,
            model_used=model_name,
            cached=False,
        )
    )

    return QueryResponse(
        answer=answer,
        crag_verdict=verdict,
        sources=sources_list,
        model_used=model_name,
        cached=False,
        web_search_timeout=web_search_timed_out,
        latency_ms=latency_ms,
        query_transformed=query_transformed,
    )


# ---------------------------------------------------------------------------
# SSE streaming path
# ---------------------------------------------------------------------------

def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_events(
    question: str,
    state: AppState,
    openai_client: OpenAI,
    groq_client: Groq,
    qdrant_client: QdrantClient,
) -> Iterator[str]:
    try:
        yield from _stream_events_inner(question, state, openai_client, groq_client, qdrant_client)
    except Exception as exc:  # noqa: BLE001 — safety net: any unhandled error must close stream cleanly
        _log.error("unhandled error in stream: %s", exc, exc_info=True)
        yield _sse("error", {"message": "Internal error."})


def _stream_events_inner(
    question: str,
    state: AppState,
    openai_client: OpenAI,
    groq_client: Groq,
    qdrant_client: QdrantClient,
) -> Iterator[str]:
    t0 = time.monotonic()

    # 1. Semantic cache check
    from rag_condominios.retrieval.dense import embed_query
    try:
        query_embedding = embed_query(question, openai_client)
    except OpenAIError as exc:
        _log.error("embedding failed for stream query: %s", exc)
        yield _sse("error", {"message": "Embedding service unavailable."})
        return

    cache: QdrantSemanticCache | None = None
    cached: CachedResponse | None = None
    try:
        cache = QdrantSemanticCache(qdrant_client, settings.semantic_cache_threshold)
        cached = cache.lookup(query_embedding)
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        _log.warning("semantic cache unavailable: %s", exc)

    if cached is not None:
        yield _sse("token", {"content": cached.answer})
        yield _sse("metadata", {
            "crag_verdict": cached.crag_verdict,
            "model_used": "cached",
            "cached": True,
            "web_search_timeout": False,
        })
        yield _sse("sources", {"chunks": cached.sources})
        latency_ms = int((time.monotonic() - t0) * 1000)
        yield _sse("done", {"latency_ms": latency_ms})
        return

    # 2. Query transform + retrieval
    yield _sse("status", {"message": "transformando query..."})
    query_transformed: str | None = None
    try:
        results = _retrieve_with_transforms(
            question, state, openai_client, groq_client, qdrant_client
        )
        query_transformed = "hyde+multi"
    except (GroqAPIError, OpenAIError, UnexpectedResponse, ResponseHandlingException) as exc:
        _log.warning("query transforms failed, falling back to base retrieval: %s", exc)
        results = retrieve(
            query=question,
            openai_client=openai_client,
            qdrant_client=qdrant_client,
            bm25_retriever=state.bm25_retriever,
            bm25_chunk_ids=state.bm25_chunk_ids,
        )

    yield _sse("status", {"message": "reranqueando documentos..."})
    ranked, verdict = rerank_and_evaluate(question, results, reranker=state.reranker)

    # 3. CRAG action
    web_search_timed_out = False
    final_chunks: list[Any] = list(ranked)
    reranker = state.reranker

    if verdict == "correct" and reranker is not None:
        final_chunks = list(decompose_recompose(question, ranked, reranker))

    elif verdict == "incorrect":
        yield _sse("status", {"message": "buscando fontes externas..."})
        web_chunks, web_search_timed_out = web_search(
            question, groq_client, settings.allowed_domains_list
        )
        if web_chunks:
            final_chunks = list(web_chunks)

    elif verdict == "ambiguous" and reranker is not None:
        yield _sse("status", {"message": "buscando fontes externas..."})
        merged, web_search_timed_out = ambiguous_action(
            question, ranked, reranker, groq_client, settings.allowed_domains_list
        )
        final_chunks = list(merged)

    # 4. Model routing
    tier = classify_query(question)
    llm_client, model_name = _pick_model(tier, verdict, openai_client, groq_client)

    # 5. Stream generation (only with Groq; OpenAI path falls back to non-stream)
    full_answer_parts: list[str] = []
    if llm_client is groq_client:
        try:
            for token in generate_stream(question, final_chunks, groq_client, model_name):
                full_answer_parts.append(token)
                yield _sse("token", {"content": token})
        except GroqAPIError as exc:
            _log.error("stream generation failed mid-response: %s", exc)
            yield _sse("error", {"message": "Falha na geração da resposta."})
            return
    else:
        try:
            answer = generate(question, final_chunks, llm_client, model_name)
        except (GroqAPIError, OpenAIError) as exc:
            _log.error("stream generation (non-Groq) failed (model=%s): %s", model_name, exc)
            yield _sse("error", {"message": "Falha na geração da resposta."})
            return
        if not answer:
            _log.error("generate() returned empty content in stream (model=%s)", model_name)
            yield _sse("error", {"message": "Falha na geração da resposta."})
            return
        full_answer_parts.append(answer)
        yield _sse("token", {"content": answer})

    full_answer = "".join(full_answer_parts)
    if not full_answer:
        _log.error("stream produced empty answer (model=%s)", model_name)
        yield _sse("error", {"message": "Falha na geração da resposta."})
        return

    # 6. Store in cache
    sources_dicts = _sources_dicts(ranked)
    if cache is not None:
        try:
            cache.store(
                query_embedding,
                CachedResponse(
                    question=question,
                    answer=full_answer,
                    crag_verdict=verdict,
                    sources=sources_dicts,
                ),
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            _log.warning("semantic cache store failed: %s", exc)

    yield _sse("metadata", {
        "crag_verdict": verdict,
        "model_used": model_name,
        "cached": False,
        "web_search_timeout": web_search_timed_out,
        "query_transformed": query_transformed,
    })
    yield _sse("sources", {
        "chunks": [
            {"text": r.text, "score": round(r.rerank_score, 4), "artigo": r.artigo}
            for r in ranked[:5]
        ]
    })
    latency_ms = int((time.monotonic() - t0) * 1000)
    yield _sse("done", {"latency_ms": latency_ms})


def _build_stream(
    question: str,
    state: AppState,
    openai_client: OpenAI,
    groq_client: Groq,
    qdrant_client: QdrantClient,
) -> StreamingResponse:
    return StreamingResponse(
        _stream_events(question, state, openai_client, groq_client, qdrant_client),
        media_type="text/event-stream",
    )
