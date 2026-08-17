"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Module-level constants — single source of truth for the whole pipeline.
# Importing from here prevents silent mismatches between indexing and search.
# ---------------------------------------------------------------------------

COLLECTION_NAME = "condominio_docs"
TIKTOKEN_ENCODING = "cl100k_base"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_DEFAULT = 20
RRF_K = 60

# Reranker models (DC-01)
# Baseline: English-only, used to measure context_precision before swap.
RERANKER_MODEL_MSMARCO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# DC-01 replacement: multilingual, expected to outperform on Portuguese corpus.
RERANKER_MODEL_BGE = "BAAI/bge-reranker-v2-m3"
# Default kept as ms-marco for backward compatibility; override via RERANKER_MODEL env var.
RERANKER_MODEL = RERANKER_MODEL_MSMARCO

# CRAG thresholds — see SPEC section 7.2
CRAG_CORRECT_THRESHOLD = 0.75
CRAG_AMBIGUOUS_THRESHOLD = 0.70

# Semantic cache — see SPEC section 8
SEMANTIC_CACHE_COLLECTION = "query_cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    groq_api_key: str = ""
    openai_frontier_model: str = ""
    semantic_cache_threshold: float = 0.95
    crag_correct_threshold: float = CRAG_CORRECT_THRESHOLD
    crag_ambiguous_threshold: float = CRAG_AMBIGUOUS_THRESHOLD
    allowed_domains: str = "planalto.gov.br,jusbrasil.com.br,stj.jus.br"
    reranker_model: str = RERANKER_MODEL
    ingest_api_key: str = ""  # set INGEST_API_KEY env var to protect POST /ingest

    @property
    def allowed_domains_list(self) -> list[str]:
        return [d.strip() for d in self.allowed_domains.split(",")]


settings = Settings()
