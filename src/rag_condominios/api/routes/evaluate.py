"""POST /evaluate — run RAGAS evaluation over the golden set."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from rag_condominios.api.deps import extract_clients
from rag_condominios.api.schemas import EvaluateResponse
from rag_condominios.api.state import AppState
from rag_condominios.eval.evaluator import evaluate_golden_set
from rag_condominios.eval.loader import GOLDEN_SET_PATH, evaluable_cases

router = APIRouter()

_log = logging.getLogger(__name__)

# Resolved relative to this file so the path is CWD-independent.
_REPORT_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "eval_report.json"

_PASS_THRESHOLD = 0.70


@router.post("/evaluate", response_model=EvaluateResponse)
def run_evaluate(request: Request) -> EvaluateResponse:
    state: AppState = request.app.state.rag
    openai_client, groq_client, qdrant_client = extract_clients(state)

    if not GOLDEN_SET_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Golden set not found at {GOLDEN_SET_PATH}. Run data preparation first.",
        )

    cases = evaluable_cases()
    if not cases:
        raise HTTPException(status_code=422, detail="No evaluable cases found in golden set.")

    report = evaluate_golden_set(
        cases=cases,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        groq_client=groq_client,
        bm25_retriever=state.bm25_retriever,
        bm25_chunk_ids=state.bm25_chunk_ids,
        reranker=state.reranker,
    )

    if report.n_cases == 0:
        raise HTTPException(
            status_code=503,
            detail="All evaluation cases failed during pipeline execution.",
        )

    try:
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "context_precision": report.context_precision,
                    "context_recall": report.context_recall,
                    "faithfulness": report.faithfulness,
                    "answer_relevancy": report.answer_relevancy,
                    "n_cases": report.n_cases,
                    "by_category": report.by_category,
                    "verdict_distribution": report.verdict_distribution,
                    "passed": report.passed(_PASS_THRESHOLD),
                    "threshold": _PASS_THRESHOLD,
                },
                fh,
                indent=2,
            )
    except OSError as exc:
        _log.warning("could not save eval report to %s: %s", _REPORT_PATH, exc)

    return EvaluateResponse(
        context_precision=report.context_precision,
        context_recall=report.context_recall,
        faithfulness=report.faithfulness,
        answer_relevancy=report.answer_relevancy,
        n_cases=report.n_cases,
        passed=report.passed(_PASS_THRESHOLD),
        report_path=str(_REPORT_PATH),
    )
