"""CLI script to run RAGAS evaluation against the golden set."""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pickle

from groq import Groq
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_condominios.core.config import settings
from rag_condominios.eval.evaluator import EvalReport, evaluate_golden_set
from rag_condominios.eval.loader import evaluable_cases

BM25_INDEX_PATH = Path("data/bm25_index.pkl")
PASS_THRESHOLD = 0.70


def _load_bm25(path: Path) -> tuple[Any, list[str]]:
    with open(path, "rb") as f:
        payload: dict[str, Any] = pickle.load(f)
    return payload["retriever"], payload["chunk_ids"]


def _print_report(report: EvalReport) -> None:
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION REPORT")
    print("=" * 50)
    print(f"Cases evaluated : {report.n_cases}")
    print(f"context_precision  : {report.context_precision:.3f}")
    print(f"context_recall     : {report.context_recall:.3f}")
    print(f"faithfulness       : {report.faithfulness:.3f}")
    print(f"answer_relevancy   : {report.answer_relevancy:.3f}")

    if report.by_category:
        print("\nBy category:")
        for cat, scores in sorted(report.by_category.items()):
            cp = scores.get("context_precision", 0)
            cr = scores.get("context_recall", 0)
            f = scores.get("faithfulness", 0)
            ar = scores.get("answer_relevancy", 0)
            print(f"  {cat:<25} cp={cp:.2f} cr={cr:.2f} f={f:.2f} ar={ar:.2f}")

    status = "PASS" if report.passed(PASS_THRESHOLD) else "FAIL"
    print(f"\nResult: {status} (threshold={PASS_THRESHOLD})")
    print("=" * 50)


def main(fail_below_threshold: bool = False) -> None:
    if not BM25_INDEX_PATH.exists():
        print(f"[eval] BM25 index not found at {BM25_INDEX_PATH}. Run ingest first.")
        sys.exit(1)

    cases = evaluable_cases()
    print(f"[eval] Loaded {len(cases)} evaluable cases from golden set.")

    bm25_retriever, bm25_chunk_ids = _load_bm25(BM25_INDEX_PATH)

    openai_client = OpenAI(api_key=settings.openai_api_key)
    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    groq_client = Groq(api_key=settings.groq_api_key)

    print("[eval] Running pipeline on all cases...")
    report = evaluate_golden_set(
        cases=cases,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        groq_client=groq_client,
        bm25_retriever=bm25_retriever,
        bm25_chunk_ids=bm25_chunk_ids,
    )

    _print_report(report)

    # Save JSON report alongside the script output for CI artifacts
    report_path = Path("data/eval_report.json")
    report_path.write_text(
        json.dumps({
            "context_precision": report.context_precision,
            "context_recall": report.context_recall,
            "faithfulness": report.faithfulness,
            "answer_relevancy": report.answer_relevancy,
            "n_cases": report.n_cases,
            "by_category": report.by_category,
            "passed": report.passed(PASS_THRESHOLD),
            "threshold": PASS_THRESHOLD,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[eval] Report saved to {report_path}")

    if fail_below_threshold and not report.passed(PASS_THRESHOLD):
        sys.exit(1)


if __name__ == "__main__":
    fail_below_threshold = "--fail-below-threshold" in sys.argv
    main(fail_below_threshold=fail_below_threshold)
