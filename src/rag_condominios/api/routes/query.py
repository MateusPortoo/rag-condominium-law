"""POST /query — sync and SSE streaming variants."""

import json
import time
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

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
    generate,
)
from rag_condominios.retrieval.pipeline import RetrievalResult, retrieve

router = APIRouter()


def _crag_verdict(chunks: list[RetrievalResult]) -> Literal["correct", "ambiguous", "incorrect"]:
    # Phase 1 placeholder — cross-encoder CRAG arrives in Phase 2 (DC-01).
    # When chunks exist we assume retrieval succeeded; "incorrect" when nothing was found.
    return "correct" if chunks else "incorrect"


def _sources(chunks: list[RetrievalResult]) -> list[SourceItem]:
    return [
        SourceItem(chunk=c.text, score=round(c.rrf_score, 4), artigo=c.artigo)
        for c in chunks[:5]
    ]


def _require_ready(state: AppState) -> None:
    if not state.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Serviço não inicializado. Verifique as variáveis de ambiente.",
        )


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: Request,
    body: QueryRequest,
    stream: bool = Query(default=False),
) -> QueryResponse | StreamingResponse:
    if detect_injection(body.question):
        raise HTTPException(status_code=400, detail="Query inválida.")

    state: AppState = request.app.state.rag
    _require_ready(state)

    if stream:
        return _build_stream(body.question, state)
    return _sync_answer(body.question, state)


# ---------------------------------------------------------------------------
# Sync path
# ---------------------------------------------------------------------------

def _sync_answer(question: str, state: AppState) -> QueryResponse:
    assert state.openai_client is not None
    assert state.groq_client is not None
    assert state.qdrant_client is not None

    t0 = time.monotonic()
    chunks = retrieve(
        query=question,
        openai_client=state.openai_client,
        qdrant_client=state.qdrant_client,
        bm25_retriever=state.bm25_retriever,
        bm25_chunk_ids=state.bm25_chunk_ids,
    )
    answer = generate(question, chunks, state.groq_client)
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


def _stream_events(question: str, state: AppState) -> Iterator[str]:
    assert state.openai_client is not None
    assert state.groq_client is not None
    assert state.qdrant_client is not None

    t0 = time.monotonic()
    yield _sse("status", {"message": "recuperando documentos..."})

    chunks = retrieve(
        query=question,
        openai_client=state.openai_client,
        qdrant_client=state.qdrant_client,
        bm25_retriever=state.bm25_retriever,
        bm25_chunk_ids=state.bm25_chunk_ids,
    )
    verdict = _crag_verdict(chunks)
    context = build_context(chunks)
    user_message = f"Trechos relevantes da legislação:\n\n{context}\n\nPergunta: {question}"

    stream = state.groq_client.chat.completions.create(
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


def _build_stream(question: str, state: AppState) -> StreamingResponse:
    return StreamingResponse(
        _stream_events(question, state),
        media_type="text/event-stream",
    )
