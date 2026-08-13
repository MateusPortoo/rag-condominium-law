"""CLI script to run RAGAS evaluation against the golden set."""

import json
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groq import Groq
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_condominios.core.config import settings
from rag_condominios.eval.evaluator import EvalReport, evaluate_golden_set
from rag_condominios.eval.loader import GOLDEN_SET_PATH, evaluable_cases
from rag_condominios.retrieval.pipeline import DEFAULT_BM25_PATH
from rag_condominios.retrieval.sparse import load_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
_log = logging.getLogger(__name__)

_REPORT_PATH = DEFAULT_BM25_PATH.parent / "eval_report.json"
PASS_THRESHOLD = 0.70


def _print_report(report: EvalReport) -> None:
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION REPORT")
    print("=" * 50)
    print(f"Cases evaluated : {report.n_cases}")
    print(f"context_precision  : {report.context_precision:.3f}")
    print(f"context_recall     : {report.context_recall:.3f}")
    print(f"faithfulness       : {report.faithfulness:.3f}")
    print(f"answer_relevancy   : {report.answer_relevancy:.3f}")

    if report.verdict_distribution:
        print("\nCRAG verdict distribution:")
        for verdict, count in sorted(report.verdict_distribution.items()):
            print(f"  {verdict:<12} {count}")

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
    if not DEFAULT_BM25_PATH.exists():
        _log.error("BM25 index not found at %s. Run ingest first.", DEFAULT_BM25_PATH)
        sys.exit(1)

    if not GOLDEN_SET_PATH.exists():
        _log.error("Golden set not found at %s. Run data preparation first.", GOLDEN_SET_PATH)
        sys.exit(1)

    cases = evaluable_cases()
    _log.info("Loaded %d evaluable cases from golden set.", len(cases))

    try:
        bm25_retriever, bm25_chunk_ids = load_index(DEFAULT_BM25_PATH)
    except RuntimeError as exc:
        _log.error("BM25 index is corrupt or invalid: %s", exc)
        sys.exit(1)

    openai_client = OpenAI(api_key=settings.openai_api_key)
    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    groq_client = Groq(api_key=settings.groq_api_key)

    _log.info("Running pipeline on all cases...")
    report = evaluate_golden_set(
        cases=cases,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        groq_client=groq_client,
        bm25_retriever=bm25_retriever,
        bm25_chunk_ids=bm25_chunk_ids,
    )

    _print_report(report)

    try:
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text(
            json.dumps(
                {
                    "context_precision": report.context_precision,
                    "context_recall": report.context_recall,
                    "faithfulness": report.faithfulness,
                    "answer_relevancy": report.answer_relevancy,
                    "n_cases": report.n_cases,
                    "by_category": report.by_category,
                    "verdict_distribution": report.verdict_distribution,
                    "passed": report.passed(PASS_THRESHOLD),
                    "threshold": PASS_THRESHOLD,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _log.info("Report saved to %s", _REPORT_PATH)
    except OSError as exc:
        _log.warning("Could not save report: %s", exc)

    if fail_below_threshold and not report.passed(PASS_THRESHOLD):
        sys.exit(1)


if __name__ == "__main__":
    fail_below_threshold = "--fail-below-threshold" in sys.argv
    main(fail_below_threshold=fail_below_threshold)
