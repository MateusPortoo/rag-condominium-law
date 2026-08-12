"""CRAG evaluator — decides verdict from cross-encoder scores.

Single responsibility: apply thresholds to rerank scores and return a verdict.
No retrieval, no generation — only the decision logic (SPEC section 7.2).
"""

from typing import Literal

from rag_condominios.core.config import CRAG_AMBIGUOUS_THRESHOLD, CRAG_CORRECT_THRESHOLD
from rag_condominios.retrieval.reranker import RankedResult

Verdict = Literal["correct", "ambiguous", "incorrect"]


class CRAGEvaluator:
    """Applies SPEC thresholds to the best rerank score in the result set.

    Thresholds (SPEC 7.2):
        correct   : best_score >  CRAG_CORRECT_THRESHOLD  (0.75)
        ambiguous : best_score >= CRAG_AMBIGUOUS_THRESHOLD (0.70)
        incorrect : best_score <  CRAG_AMBIGUOUS_THRESHOLD or empty list
    """

    def __init__(
        self,
        correct_threshold: float = CRAG_CORRECT_THRESHOLD,
        ambiguous_threshold: float = CRAG_AMBIGUOUS_THRESHOLD,
    ) -> None:
        self._correct_threshold = correct_threshold
        self._ambiguous_threshold = ambiguous_threshold

    def evaluate(self, ranked: list[RankedResult]) -> Verdict:
        """Return verdict based on the best rerank score."""
        if not ranked:
            return "incorrect"

        best_score = max(r.rerank_score for r in ranked)

        if best_score > self._correct_threshold:
            return "correct"
        if best_score >= self._ambiguous_threshold:
            return "ambiguous"
        return "incorrect"
