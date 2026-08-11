"""Cross-encoder reranker — Phase 2 / DC-01.

MsMarcoReranker wraps sentence-transformers CrossEncoder and satisfies
the BaseReranker Protocol. Swap for BgeReranker in DC-01 by implementing
the same Protocol and injecting at the call site — no pipeline changes.
"""

from dataclasses import dataclass

from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

from rag_condominios.core.config import RERANKER_MODEL
from rag_condominios.retrieval.pipeline import RetrievalResult


@dataclass
class RankedResult:
    """RetrievalResult enriched with a cross-encoder rerank_score."""

    chunk_id: str
    rrf_score: float
    rerank_score: float
    text: str = ""
    lei: str = ""
    artigo: str = ""


class MsMarcoReranker:
    """Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

    Known limitation (DC-01): model trained on English, corpus in Portuguese.
    Baseline for measuring context_precision before switching to bge-reranker-v2-m3.
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self._model: CrossEncoder = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RankedResult]:
        """Score each (query, chunk) pair and return results sorted by score."""
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        raw_scores: list[float] = self._model.predict(pairs).tolist()

        ranked = [
            RankedResult(
                chunk_id=r.chunk_id,
                rrf_score=r.rrf_score,
                rerank_score=score,
                text=r.text,
                lei=r.lei,
                artigo=r.artigo,
            )
            for r, score in zip(results, raw_scores)
        ]
        return sorted(ranked, key=lambda x: x.rerank_score, reverse=True)
