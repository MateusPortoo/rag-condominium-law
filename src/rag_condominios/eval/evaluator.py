"""Run RAGAS evaluation against the golden set."""

from dataclasses import dataclass, field
from typing import Any

from groq import Groq
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_condominios.retrieval.generator import generate
from rag_condominios.retrieval.pipeline import RetrievalResult, retrieve

# ragas and datasets are imported lazily inside evaluate_golden_set() to avoid
# a broken top-level import in ragas that pulls in langchain_community.vertexai.
# Unit tests that only import EvalReport or EvalCase are not affected.


@dataclass
class EvalCase:
    question: str
    reference_answer: str
    contexts: list[str]
    answer: str
    category: str


@dataclass
class EvalReport:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    n_cases: int
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)

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
    bm25_retriever: Any,
    bm25_chunk_ids: list[str],
) -> EvalCase:
    """Run retrieval + generation for a single golden set case."""
    question = case["question"]
    reference_answer = case["reference_answer"]

    chunks: list[RetrievalResult] = retrieve(
        query=question,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        bm25_retriever=bm25_retriever,
        bm25_chunk_ids=bm25_chunk_ids,
    )

    answer = generate(question, chunks, groq_client)
    contexts = [c.text for c in chunks if c.text]

    return EvalCase(
        question=question,
        reference_answer=reference_answer,
        contexts=contexts,
        answer=answer,
        category=case["category"],
    )


def evaluate_golden_set(
    cases: list[dict[str, Any]],
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    groq_client: Groq,
    bm25_retriever: Any,
    bm25_chunk_ids: list[str],
) -> EvalReport:
    """
    Run the full pipeline on every evaluable case, then score with RAGAS.

    RAGAS expects a HuggingFace Dataset with columns:
      question, answer, contexts (list[str]), ground_truth
    """
    eval_cases: list[EvalCase] = []
    for case in cases:
        result = _run_pipeline_for_case(
            case, openai_client, qdrant_client, groq_client, bm25_retriever, bm25_chunk_ids
        )
        eval_cases.append(result)

    # Lazy imports — ragas has a broken top-level import on langchain_community.vertexai
    from datasets import Dataset  # type: ignore[import-untyped]
    from ragas import evaluate as ragas_evaluate
    from ragas.evaluation import EvaluationResult
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

    # RAGAS uses OpenAI internally as LLM judge — uses the key from environment.
    # evaluate() is typed as EvaluationResult | Executor; synchronous call always
    # returns EvaluationResult, so we assert to satisfy mypy.
    scores = ragas_evaluate(dataset, metrics=ragas_metrics)
    assert isinstance(scores, EvaluationResult)
    scores_dict = scores.to_pandas().mean().to_dict()

    # Per-category breakdown
    by_category: dict[str, dict[str, float]] = {}
    categories = list({c.category for c in eval_cases})
    for cat in categories:
        cat_indices = [i for i, c in enumerate(eval_cases) if c.category == cat]
        cat_df = scores.to_pandas().iloc[cat_indices]
        by_category[cat] = cat_df.mean().to_dict()

    return EvalReport(
        context_precision=float(scores_dict.get("context_precision", 0.0)),
        context_recall=float(scores_dict.get("context_recall", 0.0)),
        faithfulness=float(scores_dict.get("faithfulness", 0.0)),
        answer_relevancy=float(scores_dict.get("answer_relevancy", 0.0)),
        n_cases=len(eval_cases),
        by_category=by_category,
    )
