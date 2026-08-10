"""Unit tests for golden set loader."""

import json
import tempfile
from pathlib import Path

from rag_condominios.eval.loader import (
    EVALUABLE_CATEGORIES,
    evaluable_cases,
    load_golden_set,
)

SAMPLE_CASES = [
    {"id": "gs-001", "category": "simple", "question": "Q1", "reference_answer": "A1", "reference_articles": [], "expected_crag_verdict": "correct"},
    {"id": "gs-002", "category": "multi_chunk", "question": "Q2", "reference_answer": "A2", "reference_articles": [], "expected_crag_verdict": "correct"},
    {"id": "gs-003", "category": "no_answer", "question": "Q3", "reference_answer": "A3", "reference_articles": [], "expected_crag_verdict": "incorrect"},
    {"id": "gs-004", "category": "adversarial", "question": "Q4", "reference_answer": None, "reference_articles": [], "expected_crag_verdict": "incorrect"},
    {"id": "gs-005", "category": "ambiguous", "question": "Q5", "reference_answer": "A5", "reference_articles": [], "expected_crag_verdict": "ambiguous"},
]


def _write_temp_golden_set(cases: list[dict]) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(cases, tmp)
        return Path(tmp.name)


def test_load_golden_set_returns_all_cases() -> None:
    path = _write_temp_golden_set(SAMPLE_CASES)
    cases = load_golden_set(path)
    assert len(cases) == len(SAMPLE_CASES)


def test_load_golden_set_preserves_fields() -> None:
    path = _write_temp_golden_set(SAMPLE_CASES)
    cases = load_golden_set(path)
    assert cases[0]["id"] == "gs-001"
    assert cases[0]["category"] == "simple"


def test_evaluable_cases_excludes_no_answer() -> None:
    path = _write_temp_golden_set(SAMPLE_CASES)
    cases = evaluable_cases(path)
    categories = {c["category"] for c in cases}
    assert "no_answer" not in categories


def test_evaluable_cases_excludes_adversarial() -> None:
    path = _write_temp_golden_set(SAMPLE_CASES)
    cases = evaluable_cases(path)
    categories = {c["category"] for c in cases}
    assert "adversarial" not in categories


def test_evaluable_cases_keeps_evaluable_categories() -> None:
    path = _write_temp_golden_set(SAMPLE_CASES)
    cases = evaluable_cases(path)
    categories = {c["category"] for c in cases}
    assert categories.issubset(EVALUABLE_CATEGORIES)


def test_evaluable_cases_count() -> None:
    path = _write_temp_golden_set(SAMPLE_CASES)
    cases = evaluable_cases(path)
    # simple + multi_chunk + ambiguous = 3 evaluable, no_answer + adversarial excluded
    assert len(cases) == 3
