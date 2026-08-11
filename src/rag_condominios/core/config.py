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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    groq_api_key: str = ""
    openai_frontier_model: str = ""
    semantic_cache_threshold: float = 0.95
    crag_correct_threshold: float = 0.75
    crag_incorrect_threshold: float = 0.70
    allowed_domains: str = "planalto.gov.br,jusbrasil.com.br,stj.jus.br"

    @property
    def allowed_domains_list(self) -> list[str]:
        return [d.strip() for d in self.allowed_domains.split(",")]


settings = Settings()
