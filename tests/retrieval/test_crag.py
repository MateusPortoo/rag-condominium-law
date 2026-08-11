"""Tests for CRAGEvaluator — threshold logic from SPEC section 7.2."""


from rag_condominios.retrieval.crag import CRAGEvaluator
from rag_condominios.retrieval.reranker import RankedResult


def _ranked(score: float) -> RankedResult:
    return RankedResult(chunk_id="c1", rrf_score=0.5, rerank_score=score)


def _evaluator() -> CRAGEvaluator:
    return CRAGEvaluator()


def test_verdict_correct_when_best_score_above_threshold() -> None:
    assert _evaluator().evaluate([_ranked(0.80)]) == "correct"


def test_verdict_correct_when_one_high_one_low() -> None:
    assert _evaluator().evaluate([_ranked(0.80), _ranked(0.40)]) == "correct"


def test_verdict_ambiguous_when_best_score_between_thresholds() -> None:
    assert _evaluator().evaluate([_ranked(0.72)]) == "ambiguous"


def test_verdict_incorrect_when_best_score_below_threshold() -> None:
    assert _evaluator().evaluate([_ranked(0.65)]) == "incorrect"


def test_verdict_incorrect_when_empty() -> None:
    assert _evaluator().evaluate([]) == "incorrect"


def test_verdict_boundary_correct_threshold() -> None:
    # SPEC: correct = best_score > 0.75 (strict greater than)
    assert _evaluator().evaluate([_ranked(0.75)]) == "ambiguous"


def test_verdict_boundary_ambiguous_threshold() -> None:
    # SPEC: ambiguous = best_score >= 0.70 (inclusive)
    assert _evaluator().evaluate([_ranked(0.70)]) == "ambiguous"


def test_verdict_boundary_just_above_correct() -> None:
    assert _evaluator().evaluate([_ranked(0.7501)]) == "correct"


def test_custom_thresholds_respected() -> None:
    ev = CRAGEvaluator(correct_threshold=0.90, ambiguous_threshold=0.80)
    assert ev.evaluate([_ranked(0.85)]) == "ambiguous"
    assert ev.evaluate([_ranked(0.91)]) == "correct"
    assert ev.evaluate([_ranked(0.79)]) == "incorrect"
