"""Load and filter the golden set from disk."""

import json
from pathlib import Path
from typing import Any

# Resolved relative to this file so the path is CWD-independent.
# loader.py is at src/rag_condominios/eval/loader.py → 4 parents up → repo root
GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "golden_set.json"

# Categories that require real retrieved context to evaluate.
# Adversarial cases (prompt injection, out-of-scope) are excluded from RAGAS
# because they have no reference_answer and the system is expected to refuse.
EVALUABLE_CATEGORIES = {"simple", "multi_chunk", "different_vocabulary", "ambiguous"}


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict[str, Any]]:
    """Return all golden set cases from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def evaluable_cases(path: Path = GOLDEN_SET_PATH) -> list[dict[str, Any]]:
    """
    Return only cases suitable for RAGAS evaluation.

    Excluded:
    - no_answer: system is expected to say "I don't know", not retrieve context
    - adversarial: no reference_answer, system should refuse
    """
    cases = load_golden_set(path)
    return [c for c in cases if c["category"] in EVALUABLE_CATEGORIES]
