"""Domain models shared across the pipeline."""

from pydantic import BaseModel


class ArticleChunk(BaseModel):
    id: str
    text: str
    token_count: int
    lei: str
    artigo: str
    secao: str
    chunk_index: int


class IngestResult(BaseModel):
    total_articles: int
    total_chunks: int
    # Keys: p50, p90, p95, max — values in tokens
    distribution: dict[str, int]
    chunk_size_used: int
    # Percentage of articles whose token count fits within chunk_size_used
    coverage_pct: float
