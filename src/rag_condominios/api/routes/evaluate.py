"""POST /evaluate — run RAGAS evaluation over the golden set."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from rag_condominios.api.schemas import EvaluateResponse
from rag_condominios.api.state import AppState
from rag_condominios.eval.evaluator import evaluate_golden_set
from rag_condominios.eval.loader import evaluable_cases

router = APIRouter()

_REPORT_PATH = Path("data/eval_report.json")


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: Request) -> EvaluateResponse:
    state: AppState = request.app.state.rag
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    assert state.openai_client is not None
    assert state.groq_client is not None
    assert state.qdrant_client is not None

    cases = evaluable_cases()
    report = evaluate_golden_set(
        cases=cases,
        openai_client=state.openai_client,
        qdrant_client=state.qdrant_client,
        groq_client=state.groq_client,
        bm25_retriever=state.bm25_retriever,
        bm25_chunk_ids=state.bm25_chunk_ids,
    )

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
            },
            fh,
            indent=2,
        )

    return EvaluateResponse(
        context_precision=report.context_precision,
        context_recall=report.context_recall,
        faithfulness=report.faithfulness,
        answer_relevancy=report.answer_relevancy,
        n_cases=report.n_cases,
        passed=report.passed(),
        report_path=str(_REPORT_PATH),
    )
