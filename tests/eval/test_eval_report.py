"""Unit tests for EvalReport logic — no RAGAS or API calls needed."""

from rag_condominios.eval.evaluator import EvalReport


def _make_report(**kwargs: float) -> EvalReport:
    defaults = {
        "context_precision": 0.80,
        "context_recall": 0.80,
        "faithfulness": 0.80,
        "answer_relevancy": 0.80,
    }
    defaults.update(kwargs)
    return EvalReport(n_cases=10, **defaults)  # type: ignore[arg-type]


def test_report_passes_when_all_metrics_above_threshold() -> None:
    report = _make_report()
    assert report.passed(threshold=0.70) is True


def test_report_fails_when_context_precision_below_threshold() -> None:
    report = _make_report(context_precision=0.60)
    assert report.passed(threshold=0.70) is False


def test_report_fails_when_context_recall_below_threshold() -> None:
    report = _make_report(context_recall=0.65)
    assert report.passed(threshold=0.70) is False


def test_report_fails_when_faithfulness_below_threshold() -> None:
    report = _make_report(faithfulness=0.50)
    assert report.passed(threshold=0.70) is False


def test_report_fails_when_answer_relevancy_below_threshold() -> None:
    report = _make_report(answer_relevancy=0.69)
    assert report.passed(threshold=0.70) is False


def test_report_passes_exactly_at_threshold() -> None:
    report = _make_report(
        context_precision=0.70,
        context_recall=0.70,
        faithfulness=0.70,
        answer_relevancy=0.70,
    )
    assert report.passed(threshold=0.70) is True


def test_report_custom_threshold() -> None:
    report = _make_report(
        context_precision=0.85,
        context_recall=0.85,
        faithfulness=0.85,
        answer_relevancy=0.85,
    )
    assert report.passed(threshold=0.90) is False
    assert report.passed(threshold=0.80) is True
