"""Structural typing protocols for the RAG pipeline.

Protocols enable Liskov Substitution and Dependency Inversion without
requiring subclassing. Any object that implements the required methods
automatically satisfies the protocol (structural typing).

Adding a new retriever or LLM provider means implementing these protocols,
not modifying the pipeline internals — Open/Closed in practice.
"""

from typing import Any, Protocol, runtime_checkable


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
