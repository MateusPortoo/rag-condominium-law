"""Unit tests for evaluator — _run_pipeline_for_case and evaluate_golden_set."""

import sys
from unittest.mock import MagicMock, patch

from rag_condominios.eval.evaluator import (
    EvalCase,
    _run_pipeline_for_case,
    evaluate_golden_set,
)
from rag_condominios.retrieval.pipeline import RetrievalResult
from rag_condominios.retrieval.reranker import RankedResult


def _fake_chunk(chunk_id: str = "c-001", text: str = "Condomínio deve...") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        rrf_score=0.02,
        text=text,
        lei="CC",
        artigo="Art. 1.350",
    )


def _fake_ranked(chunk: RetrievalResult, score: float = 0.85) -> RankedResult:
    return RankedResult(
        chunk_id=chunk.chunk_id,
        rrf_score=chunk.rrf_score,
        rerank_score=score,
        text=chunk.text,
        lei=chunk.lei,
        artigo=chunk.artigo,
    )


def _clients() -> tuple[MagicMock, MagicMock, MagicMock]:
    return MagicMock(), MagicMock(), MagicMock()


def _bm25() -> tuple[MagicMock, list[str]]:
    return MagicMock(), ["chunk-001"]


_CASE: dict = {
    "id": "gs-001",
    "category": "simple",
    "question": "Qual o prazo para assembleia?",
    "reference_answer": "60 dias.",
    "reference_articles": [],
    "expected_crag_verdict": "correct",
}


class TestRunPipelineForCase:
    def test_returns_eval_case_on_success(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()
        chunk = _fake_chunk()
        ranked = [_fake_ranked(chunk)]

        with (
            patch("rag_condominios.eval.evaluator.retrieve", return_value=[chunk]),
            patch(
                "rag_condominios.eval.evaluator.rerank_and_evaluate",
                return_value=(ranked, "correct"),
            ),
            patch("rag_condominios.eval.evaluator.generate", return_value="60 dias conforme CC."),
        ):
            result = _run_pipeline_for_case(
                _CASE, openai, qdrant, groq, bm25, chunk_ids,
            )

        assert isinstance(result, EvalCase)
        assert result.question == _CASE["question"]
        assert result.reference_answer == _CASE["reference_answer"]
        assert result.crag_verdict == "correct"
        assert result.category == "simple"
        assert "Condomínio deve..." in result.contexts

    def test_returns_none_when_retrieval_fails(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()

        with patch(
            "rag_condominios.eval.evaluator.retrieve",
            side_effect=RuntimeError("qdrant down"),
        ):
            result = _run_pipeline_for_case(_CASE, openai, qdrant, groq, bm25, chunk_ids)

        assert result is None

    def test_returns_none_when_generation_fails(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()
        chunk = _fake_chunk()
        ranked = [_fake_ranked(chunk)]

        with (
            patch("rag_condominios.eval.evaluator.retrieve", return_value=[chunk]),
            patch(
                "rag_condominios.eval.evaluator.rerank_and_evaluate",
                return_value=(ranked, "correct"),
            ),
            patch(
                "rag_condominios.eval.evaluator.generate",
                side_effect=RuntimeError("groq down"),
            ),
        ):
            result = _run_pipeline_for_case(_CASE, openai, qdrant, groq, bm25, chunk_ids)

        assert result is None

    def test_returns_none_when_answer_is_empty(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()
        chunk = _fake_chunk()
        ranked = [_fake_ranked(chunk)]

        with (
            patch("rag_condominios.eval.evaluator.retrieve", return_value=[chunk]),
            patch(
                "rag_condominios.eval.evaluator.rerank_and_evaluate",
                return_value=(ranked, "correct"),
            ),
            patch("rag_condominios.eval.evaluator.generate", return_value=""),
        ):
            result = _run_pipeline_for_case(_CASE, openai, qdrant, groq, bm25, chunk_ids)

        assert result is None

    def test_passes_reranker_to_pipeline(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()
        chunk = _fake_chunk()
        ranked = [_fake_ranked(chunk)]
        fake_reranker = MagicMock()

        mock_rerank = MagicMock(return_value=(ranked, "ambiguous"))

        with (
            patch("rag_condominios.eval.evaluator.retrieve", return_value=[chunk]),
            patch("rag_condominios.eval.evaluator.rerank_and_evaluate", mock_rerank),
            patch("rag_condominios.eval.evaluator.generate", return_value="Ambíguo."),
        ):
            result = _run_pipeline_for_case(
                _CASE, openai, qdrant, groq, bm25, chunk_ids, reranker=fake_reranker,
            )

        _, kwargs = mock_rerank.call_args
        assert kwargs.get("reranker") is fake_reranker
        assert result is not None
        assert result.crag_verdict == "ambiguous"

    def test_falls_back_category_to_unknown_when_missing(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()
        chunk = _fake_chunk()
        ranked = [_fake_ranked(chunk)]
        case_without_category = {k: v for k, v in _CASE.items() if k != "category"}

        with (
            patch("rag_condominios.eval.evaluator.retrieve", return_value=[chunk]),
            patch(
                "rag_condominios.eval.evaluator.rerank_and_evaluate",
                return_value=(ranked, "correct"),
            ),
            patch("rag_condominios.eval.evaluator.generate", return_value="Resposta."),
        ):
            result = _run_pipeline_for_case(
                case_without_category, openai, qdrant, groq, bm25, chunk_ids,
            )

        assert result is not None
        assert result.category == "unknown"


class TestEvaluateGoldenSet:
    def _cases(self) -> list[dict]:
        return [
            {**_CASE, "id": f"gs-00{i}", "category": "simple" if i % 2 == 0 else "multi_chunk"}
            for i in range(1, 4)
        ]

    def test_returns_zero_report_when_all_cases_fail(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()

        with patch("rag_condominios.eval.evaluator.retrieve", side_effect=RuntimeError("fail")):
            report = evaluate_golden_set(
                cases=self._cases(),
                openai_client=openai,
                qdrant_client=qdrant,
                groq_client=groq,
                bm25_retriever=bm25,
                bm25_chunk_ids=chunk_ids,
            )

        assert report.n_cases == 0
        assert report.context_precision == 0.0
        assert report.context_recall == 0.0

    def test_returns_zero_report_when_cases_list_empty(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()

        report = evaluate_golden_set(
            cases=[],
            openai_client=openai,
            qdrant_client=qdrant,
            groq_client=groq,
            bm25_retriever=bm25,
            bm25_chunk_ids=chunk_ids,
        )

        assert report.n_cases == 0

    def _fake_scores_df(self) -> MagicMock:
        df = MagicMock()
        df.mean.return_value.to_dict.return_value = {
            "context_precision": 0.80,
            "context_recall": 0.75,
            "faithfulness": 0.85,
            "answer_relevancy": 0.78,
        }
        df.iloc.__getitem__.return_value = df
        return df

    def _mock_ragas_modules(self, fake_scores: MagicMock) -> dict:
        mock_ragas_mod = MagicMock()
        mock_ragas_mod.evaluate = MagicMock(return_value=fake_scores)
        # EvaluationDataset and SingleTurnSample are constructors — return a MagicMock.
        mock_ragas_mod.EvaluationDataset = MagicMock(return_value=MagicMock())
        mock_ragas_mod.SingleTurnSample = MagicMock(return_value=MagicMock())

        mock_metrics_mod = MagicMock()

        return {
            "ragas": mock_ragas_mod,
            "ragas.metrics": mock_metrics_mod,
        }

    def test_report_includes_verdict_distribution(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()
        chunk = _fake_chunk()
        ranked = [_fake_ranked(chunk)]

        fake_scores = MagicMock()
        fake_scores.to_pandas.return_value = self._fake_scores_df()

        with (
            patch("rag_condominios.eval.evaluator.retrieve", return_value=[chunk]),
            patch(
                "rag_condominios.eval.evaluator.rerank_and_evaluate",
                return_value=(ranked, "correct"),
            ),
            patch("rag_condominios.eval.evaluator.generate", return_value="Resposta OK."),
            patch.dict(sys.modules, self._mock_ragas_modules(fake_scores)),
        ):
            report = evaluate_golden_set(
                cases=self._cases(),
                openai_client=openai,
                qdrant_client=qdrant,
                groq_client=groq,
                bm25_retriever=bm25,
                bm25_chunk_ids=chunk_ids,
            )

        assert report.n_cases == 3
        assert report.verdict_distribution == {"correct": 3}
        assert report.by_category != {}

    def test_skips_failed_cases_and_evaluates_rest(self) -> None:
        openai, groq, qdrant = _clients()
        bm25, chunk_ids = _bm25()
        chunk = _fake_chunk()
        ranked = [_fake_ranked(chunk)]

        call_count = 0

        def retrieve_side_effect(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first case fails")
            return [chunk]

        fake_scores_df = MagicMock()
        fake_scores_df.mean.return_value.to_dict.return_value = {
            "context_precision": 0.70,
            "context_recall": 0.70,
            "faithfulness": 0.70,
            "answer_relevancy": 0.70,
        }
        fake_scores_df.iloc.__getitem__.return_value = fake_scores_df
        fake_scores = MagicMock()
        fake_scores.to_pandas.return_value = fake_scores_df

        with (
            patch("rag_condominios.eval.evaluator.retrieve", side_effect=retrieve_side_effect),
            patch(
                "rag_condominios.eval.evaluator.rerank_and_evaluate",
                return_value=(ranked, "correct"),
            ),
            patch("rag_condominios.eval.evaluator.generate", return_value="Resp."),
            patch.dict(sys.modules, self._mock_ragas_modules(fake_scores)),
        ):
            report = evaluate_golden_set(
                cases=self._cases(),
                openai_client=openai,
                qdrant_client=qdrant,
                groq_client=groq,
                bm25_retriever=bm25,
                bm25_chunk_ids=chunk_ids,
            )

        assert report.n_cases == 2
