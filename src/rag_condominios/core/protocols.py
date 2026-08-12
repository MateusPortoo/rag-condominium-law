"""Structural typing protocols for the RAG pipeline.

Protocols enable Liskov Substitution and Dependency Inversion without
requiring subclassing. Any object that implements the required methods
automatically satisfies the protocol (structural typing).

Adding a new retriever or LLM provider means implementing these protocols,
not modifying the pipeline internals — Open/Closed in practice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rag_condominios.retrieval.semantic_cache import CachedResponse


@runtime_checkable
class BM25Retriever(Protocol):
    """Structural interface satisfied by bm25s.BM25.

    Using this instead of `Any` allows mypy to catch type mismatches
    and makes the expected contract explicit to future contributors.
    """

    def retrieve(self, query_tokens: Any, k: int) -> tuple[Any, Any]:
        """Return (results_array, scores_array) for the given tokenized query."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Structural interface for any OpenAI-compatible chat completion client.

    Both `groq.Groq` and `openai.OpenAI` satisfy this protocol, enabling
    model routing without modifying the generator (Open/Closed).
    """

    @property
    def chat(self) -> Any:
        """Access to chat.completions.create(...)."""
        ...


@runtime_checkable
class BaseReranker(Protocol):
    """Structural interface for cross-encoder rerankers.

    Implementations receive (query, results) and return results sorted by
    rerank_score descending. Satisfying this Protocol enables DC-01:
    swap ms-marco for bge-reranker without touching the pipeline.
    """

    def rerank(self, query: str, results: list[Any]) -> list[Any]:
        """Score and sort results by relevance to query."""
        ...


@runtime_checkable
class BaseCRAGEvaluator(Protocol):
    """Structural interface for CRAG verdict evaluators.

    Decouples threshold logic from the pipeline — evaluator can be
    swapped or mocked in tests without modifying query.py.
    """

    def evaluate(self, ranked: list[Any]) -> Literal["correct", "ambiguous", "incorrect"]:
        """Return CRAG verdict based on rerank scores."""
        ...


@runtime_checkable
class BaseQueryTransformer(Protocol):
    """Structural interface for query transformation strategies (SPEC section 6).

    Implementations may use HyDE, multi-query, or other techniques.
    Returns a list of transformed query strings to be retrieved in parallel.
    """

    def transform(self, query: str) -> list[str]:
        """Return one or more transformed queries derived from the original."""
        ...


@runtime_checkable
class BaseSemanticCache(Protocol):
    """Structural interface for semantic query caching (SPEC section 8).

    Cache lookups use embedding similarity so near-duplicate questions
    are served from cache without repeating the full retrieval pipeline.
    """

    def lookup(self, query_embedding: list[float]) -> CachedResponse | None:
        """Return a cached response if one exists above the similarity threshold."""
        ...

    def store(self, query_embedding: list[float], response: CachedResponse) -> None:
        """Persist a query embedding and its response for future lookups."""
        ...
