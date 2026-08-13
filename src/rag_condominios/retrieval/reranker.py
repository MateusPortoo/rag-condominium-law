"""Cross-encoder rerankers — Phase 2 baseline (MsMarco) and DC-01 swap (Bge).

Both classes satisfy BaseReranker. Select at startup via RERANKER_MODEL env var;
no pipeline code changes required when swapping.
"""

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from rag_condominios.core.config import RERANKER_MODEL_BGE, RERANKER_MODEL_MSMARCO
from rag_condominios.retrieval.pipeline import RetrievalResult

_log = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """RetrievalResult enriched with a cross-encoder rerank_score."""

    chunk_id: str
    rrf_score: float
    rerank_score: float
    text: str = ""
    lei: str = ""
    artigo: str = ""


class _CrossEncoderReranker:
    """Base logic shared by all CrossEncoder-based rerankers."""

    def __init__(self, model_name: str) -> None:
        self._model: CrossEncoder = CrossEncoder(model_name)
        self._model_name = model_name

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RankedResult]:
        """Score each (query, chunk) pair and return results sorted by score desc."""
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        try:
            raw_scores: list[float] = self._model.predict(pairs).tolist()  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 — OOM/CUDA/shape errors degrade to rrf_score
            _log.error(
                "CrossEncoder.predict failed (model=%s n_pairs=%d): %s",
                self._model_name, len(pairs), exc,
            )
            return [
                RankedResult(
                    chunk_id=r.chunk_id,
                    rrf_score=r.rrf_score,
                    rerank_score=r.rrf_score,
                    text=r.text,
                    lei=r.lei,
                    artigo=r.artigo,
                )
                for r in results
            ]

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


class MsMarcoReranker(_CrossEncoderReranker):
    """Baseline reranker — English-only, used to measure DC-01 before delta.

    Known limitation: ms-marco-MiniLM-L-6-v2 trained on English; corpus is
    Portuguese. Expected to underperform BgeReranker on context_precision.
    """

    def __init__(self) -> None:
        super().__init__(RERANKER_MODEL_MSMARCO)


class BgeReranker(_CrossEncoderReranker):
    """DC-01 replacement — multilingual reranker for Portuguese corpus.

    BAAI/bge-reranker-v2-m3 is trained on 100+ languages. Expected to
    outperform MsMarcoReranker on context_precision for this corpus.
    Select via RERANKER_MODEL=BAAI/bge-reranker-v2-m3 env var.
    """

    def __init__(self) -> None:
        super().__init__(RERANKER_MODEL_BGE)


def create_reranker(model_name: str) -> "MsMarcoReranker | BgeReranker":
    """Factory: return the right reranker class for the given model name."""
    if model_name == RERANKER_MODEL_BGE:
        return BgeReranker()
    return MsMarcoReranker()
