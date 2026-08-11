"""POST /query — sync and SSE streaming variants."""

import json
import time
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from groq import Groq
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_condominios.api.deps import extract_clients
from rag_condominios.api.schemas import (
    MetricsEntry,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from rag_condominios.api.security import detect_injection
from rag_condominios.api.state import AppState
from rag_condominios.retrieval.generator import (
    GROQ_MODEL,
    SYSTEM_PROMPT,
    build_context,
    build_user_message,
    generate,
)
from rag_condominios.retrieval.pipeline import RetrievalResult, retrieve

router = APIRouter()


def _crag_verdict(chunks: list[RetrievalResult]) -> Literal["correct", "ambiguous", "incorrect"]:
    # Phase 1 placeholder — cross-encoder CRAG arrives in Phase 2 (DC-01).
    return "correct" if chunks else "incorrect"


def _sources(chunks: list[RetrievalResult]) -> list[SourceItem]:
    return [
        SourceItem(chunk=c.text, score=round(c.rrf_score, 4), artigo=c.artigo)
        for c in chunks[:5]
    ]


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: Request,
    body: QueryRequest,
    stream: bool = Query(default=False),
) -> QueryResponse | StreamingResponse:
    if detect_injection(body.question):
        raise HTTPException(status_code=400, detail="Query inválida.")

    state: AppState = request.app.state.rag
    openai_client, groq_client, qdrant_client = extract_clients(state)

    if stream:
        return _build_stream(body.question, state, openai_client, groq_client, qdrant_client)
    return _sync_answer(body.question, state, openai_client, groq_client, qdrant_client)


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
    chunks = retrieve(
        query=question,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        bm25_retriever=state.bm25_retriever,
        bm25_chunk_ids=state.bm25_chunk_ids,
    )
    answer = generate(question, chunks, groq_client)
    verdict = _crag_verdict(chunks)
    latency_ms = int((time.monotonic() - t0) * 1000)

    state.record_query(
        MetricsEntry(
            question=question,
            crag_verdict=verdict,
            latency_ms=latency_ms,
            model_used=GROQ_MODEL,
            cached=False,
        )
    )

    return QueryResponse(
        answer=answer,
        crag_verdict=verdict,
        sources=_sources(chunks),
        model_used=GROQ_MODEL,
        cached=False,
        web_search_timeout=False,
        latency_ms=latency_ms,
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
    t0 = time.monotonic()
    yield _sse("status", {"message": "recuperando documentos..."})

    chunks = retrieve(
        query=question,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        bm25_retriever=state.bm25_retriever,
        bm25_chunk_ids=state.bm25_chunk_ids,
    )
    verdict = _crag_verdict(chunks)
    user_message = build_user_message(build_context(chunks), question)

    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        stream=True,
    )
    for delta in stream:
        content = delta.choices[0].delta.content or ""
        if content:
            yield _sse("token", {"content": content})

    yield _sse("metadata", {
        "crag_verdict": verdict,
        "model_used": GROQ_MODEL,
        "cached": False,
        "web_search_timeout": False,
    })
    yield _sse("sources", {
        "chunks": [
            {"text": c.text, "score": round(c.rrf_score, 4), "artigo": c.artigo}
            for c in chunks[:5]
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
