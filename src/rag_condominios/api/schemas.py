"""Pydantic request/response schemas for the API."""

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceItem(BaseModel):
    chunk: str
    score: float
    artigo: str


class QueryResponse(BaseModel):
    answer: str
    crag_verdict: Literal["correct", "ambiguous", "incorrect"]
    sources: list[SourceItem]
    model_used: str
    query_transformed: str | None = None
    cached: bool = False
    web_search_timeout: bool = False
    latency_ms: int


class HealthResponse(BaseModel):
    qdrant: Literal["ok", "error"]
    openai_key_present: bool
    groq_key_present: bool
    bm25_loaded: bool


class MetricsEntry(BaseModel):
    question: str
    crag_verdict: str
    latency_ms: int
    model_used: str
    cached: bool


class MetricsResponse(BaseModel):
    recent_queries: list[MetricsEntry]
    total_count: int


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int


class EvaluateResponse(BaseModel):
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    n_cases: int
    passed: bool
    report_path: str
