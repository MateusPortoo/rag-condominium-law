"""Run RAGAS evaluation against the golden set."""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from groq import Groq
from openai import OpenAI, OpenAIError
from qdrant_client import QdrantClient

from rag_condominios.core.protocols import BaseReranker, BM25Retriever
from rag_condominios.retrieval.generator import GROQ_MODEL, generate
from rag_condominios.retrieval.pipeline import rerank_and_evaluate, retrieve

# ragas and datasets are imported lazily inside evaluate_golden_set() to avoid
# a broken top-level import in ragas that pulls in langchain_community.vertexai.
# Unit tests that only import EvalReport or EvalCase are not affected.

_log = logging.getLogger(__name__)

CRAGVerdict = Literal["correct", "ambiguous", "incorrect"]


@dataclass
class EvalCase:
    question: str
    reference_answer: str
    contexts: list[str]
    answer: str
    category: str
    crag_verdict: CRAGVerdict = "correct"


@dataclass
class EvalReport:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    n_cases: int
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    verdict_distribution: dict[str, int] = field(default_factory=dict)

    def passed(self, threshold: float = 0.70) -> bool:
        """Return True if all four metrics are above the threshold."""
        return all([
            self.context_precision >= threshold,
            self.context_recall >= threshold,
            self.faithfulness >= threshold,
            self.answer_relevancy >= threshold,
        ])


def _run_pipeline_for_case(
    case: dict[str, Any],
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    groq_client: Groq,
    bm25_retriever: BM25Retriever,
    bm25_chunk_ids: list[str],
    reranker: BaseReranker | None = None,
) -> "EvalCase | None":
    """Run retrieval + reranking + generation for a single golden set case.

    Web search (CRAG incorrect/ambiguous) is intentionally skipped to keep
    evaluation deterministic and reproducible. The CRAG verdict is recorded
    for analysis but does not trigger external calls.

    Returns None if the case cannot be processed (network/API failure).
    """
    question = case["question"]
    reference_answer = case.get("reference_answer") or ""
    category = case.get("category", "unknown")

    try:
        chunks = retrieve(
            query=question,
            openai_client=openai_client,
            qdrant_client=qdrant_client,
            bm25_retriever=bm25_retriever,
            bm25_chunk_ids=bm25_chunk_ids,
        )
    except (OpenAIError, Exception) as exc:  # noqa: BLE001
        _log.warning("retrieval failed for question=%r: %s — skipping case", question[:60], exc)
        return None

    ranked, verdict = rerank_and_evaluate(question, chunks, reranker=reranker)

    try:
        answer = generate(question, ranked, groq_client, model=GROQ_MODEL)
    except Exception as exc:  # noqa: BLE001
        _log.warning("generation failed for question=%r: %s — skipping case", question[:60], exc)
        return None

    if not answer:
        _log.warning("empty answer for question=%r — skipping case", question[:60])
        return None

    contexts = [r.text for r in ranked if r.text]

    return EvalCase(
        question=question,
        reference_answer=reference_answer,
        contexts=contexts,
        answer=answer,
        category=category,
        crag_verdict=verdict,
    )


def evaluate_golden_set(
    cases: list[dict[str, Any]],
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    groq_client: Groq,
    bm25_retriever: BM25Retriever,
    bm25_chunk_ids: list[str],
    reranker: BaseReranker | None = None,
) -> EvalReport:
    """Run the full retrieval+reranking+generation pipeline on every evaluable
    case, then score with RAGAS.

    RAGAS expects a HuggingFace Dataset with columns:
      question, answer, contexts (list[str]), ground_truth
    """
    eval_cases: list[EvalCase] = []
    for i, case in enumerate(cases):
        _log.info("evaluating case %d/%d: %s", i + 1, len(cases), case.get("id", "?"))
        result = _run_pipeline_for_case(
            case, openai_client, qdrant_client, groq_client,
            bm25_retriever, bm25_chunk_ids, reranker=reranker,
        )
        if result is not None:
            eval_cases.append(result)

    if not eval_cases:
        _log.error("no cases could be evaluated — returning zero report")
        return EvalReport(
            context_precision=0.0,
            context_recall=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            n_cases=0,
        )

    # Lazy imports — ragas has a broken top-level import on langchain_community.vertexai
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    ragas_metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

    dataset = Dataset.from_dict({
        "question": [c.question for c in eval_cases],
        "answer": [c.answer for c in eval_cases],
        "contexts": [c.contexts for c in eval_cases],
        "ground_truth": [c.reference_answer for c in eval_cases],
    })

    # RAGAS uses OpenAI internally as LLM judge â€” uses the key from environment.
    # Typed as Any: ragas 0.4.x stubs are incomplete and EvaluationResult is not
    # explicitly exported, making the union type unresolvable under mypy strict.
    scores: Any = ragas_evaluate(dataset, metrics=ragas_metrics)
    scores_dict: dict[str, Any] = scores.to_pandas().mean().to_dict()

    # Per-category breakdown
    by_category: dict[str, dict[str, float]] = {}
    for cat in sorted({c.category for c in eval_cases}):
        cat_indices = [i for i, c in enumerate(eval_cases) if c.category == cat]
        cat_df = scores.to_pandas().iloc[cat_indices]
        by_category[cat] = {k: float(v) for k, v in cat_df.mean().to_dict().items()}

    # CRAG verdict distribution for observability
    verdict_distribution: dict[str, int] = {}
    for c in eval_cases:
        verdict_distribution[c.crag_verdict] = verdict_distribution.get(c.crag_verdict, 0) + 1

    return EvalReport(
        context_precision=float(scores_dict.get("context_precision", 0.0)),
        context_recall=float(scores_dict.get("context_recall", 0.0)),
        faithfulness=float(scores_dict.get("faithfulness", 0.0)),
        answer_relevancy=float(scores_dict.get("answer_relevancy", 0.0)),
        n_cases=len(eval_cases),
        by_category=by_category,
        verdict_distribution=verdict_distribution,
    )

