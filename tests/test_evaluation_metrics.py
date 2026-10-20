"""Tests for mullet.knowledge.engine.evaluation — pure metric functions."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mullet.knowledge.engine.evaluation import (
    EvalCase,
    EvalResult,
    EvalReport,
    _chunk_matches,
    mrr,
    hit_at_k,
    precision_at_k,
    ndcg,
    context_recall,
    compute_retrieval_metrics,
    aggregate_metrics,
    load_eval_dataset,
    _llm_judge,
    compute_rag_metrics,
    generate_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(content: str) -> dict:
    return {"content": content}


# ---------------------------------------------------------------------------
# EvalCase / EvalResult / EvalReport dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_test_case_required_fields(self):
        tc = EvalCase(id="t1", query="what is auth", expected_chunks=["auth"])
        assert tc.id == "t1"
        assert tc.query == "what is auth"
        assert tc.expected_chunks == ["auth"]

    def test_test_case_defaults(self):
        tc = EvalCase(id="t2", query="q", expected_chunks=[])
        assert tc.ground_truth is None
        assert tc.tags == []

    def test_eval_result_creation(self):
        tc = EvalCase(id="t1", query="q", expected_chunks=[])
        result = EvalResult(
            test_case=tc,
            retrieved_chunks=[_chunk("hello")],
            metrics={"mrr": 1.0},
        )
        assert result.metrics["mrr"] == 1.0
        assert result.answer is None
        assert result.quality is None

    def test_eval_report_defaults(self):
        report = EvalReport(
            client_name="acme",
            config_label="bm25",
            results=[],
            aggregate={},
        )
        assert report.client_name == "acme"
        assert isinstance(report.timestamp, str)
        assert len(report.timestamp) > 0


# ---------------------------------------------------------------------------
# _chunk_matches
# ---------------------------------------------------------------------------

class TestChunkMatches:
    def test_substring_match(self):
        assert _chunk_matches(_chunk("user authentication flow"), ["auth"])

    def test_case_insensitive(self):
        assert _chunk_matches(_chunk("Authentication required"), ["authentication"])
        assert _chunk_matches(_chunk("authentication"), ["Authentication"])

    def test_multiple_patterns_any_match(self):
        assert _chunk_matches(_chunk("login page"), ["signup", "login"])

    def test_no_match(self):
        assert not _chunk_matches(_chunk("payment flow"), ["auth", "login"])

    def test_empty_patterns(self):
        assert not _chunk_matches(_chunk("content"), [])

    def test_missing_content_key(self):
        assert not _chunk_matches({}, ["auth"])


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------

class TestMrr:
    def test_first_result_relevant(self):
        chunks = [_chunk("auth login"), _chunk("payment")]
        assert mrr(chunks, ["auth"]) == 1.0

    def test_second_result_relevant(self):
        chunks = [_chunk("payment"), _chunk("auth login")]
        assert mrr(chunks, ["auth"]) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        chunks = [_chunk("a"), _chunk("b"), _chunk("auth")]
        assert mrr(chunks, ["auth"]) == pytest.approx(1 / 3)

    def test_no_relevant_result(self):
        chunks = [_chunk("payment"), _chunk("signup")]
        assert mrr(chunks, ["auth"]) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], ["auth"]) == 0.0

    def test_empty_expected(self):
        assert mrr([_chunk("auth")], []) == 0.0


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------

class TestHitAtK:
    def test_hit_within_k(self):
        chunks = [_chunk("a")] * 5 + [_chunk("auth")]
        assert hit_at_k(chunks, ["auth"], k=6) == 1.0

    def test_miss_outside_k(self):
        chunks = [_chunk("a")] * 5 + [_chunk("auth")]
        assert hit_at_k(chunks, ["auth"], k=5) == 0.0

    def test_exact_boundary(self):
        chunks = [_chunk("a")] * 4 + [_chunk("auth")]
        assert hit_at_k(chunks, ["auth"], k=5) == 1.0

    def test_empty_retrieved(self):
        assert hit_at_k([], ["auth"]) == 0.0

    def test_no_match(self):
        chunks = [_chunk("payment")] * 10
        assert hit_at_k(chunks, ["auth"]) == 0.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self):
        chunks = [_chunk("auth")] * 5
        assert precision_at_k(chunks, ["auth"], k=5) == pytest.approx(1.0)

    def test_half_relevant(self):
        chunks = [_chunk("auth"), _chunk("payment"), _chunk("auth"), _chunk("payment")]
        assert precision_at_k(chunks, ["auth"], k=4) == pytest.approx(0.5)

    def test_none_relevant(self):
        chunks = [_chunk("payment")] * 5
        assert precision_at_k(chunks, ["auth"], k=5) == 0.0

    def test_k_zero_returns_zero(self):
        assert precision_at_k([_chunk("auth")], ["auth"], k=0) == 0.0

    def test_fewer_results_than_k(self):
        chunks = [_chunk("auth")]
        # Only 1 chunk, k=5: denominator is still k (5), not len(retrieved)
        assert precision_at_k(chunks, ["auth"], k=5) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# ndcg
# ---------------------------------------------------------------------------

class TestNdcg:
    def test_perfect_ranking(self):
        chunks = [_chunk("auth"), _chunk("login")]
        score = ndcg(chunks, ["auth", "login"], k=2)
        assert score == pytest.approx(1.0)

    def test_no_relevant(self):
        chunks = [_chunk("payment")] * 5
        assert ndcg(chunks, ["auth"], k=5) == 0.0

    def test_empty_retrieved(self):
        assert ndcg([], ["auth"]) == 0.0

    def test_score_between_zero_and_one(self):
        chunks = [_chunk("payment"), _chunk("auth"), _chunk("login")]
        score = ndcg(chunks, ["auth"], k=3)
        assert 0.0 <= score <= 1.0

    def test_relevant_first_beats_relevant_later(self):
        early = [_chunk("auth"), _chunk("payment"), _chunk("payment")]
        late = [_chunk("payment"), _chunk("payment"), _chunk("auth")]
        assert ndcg(early, ["auth"]) > ndcg(late, ["auth"])


# ---------------------------------------------------------------------------
# context_recall
# ---------------------------------------------------------------------------

class TestContextRecall:
    def test_all_patterns_found(self):
        chunks = [_chunk("auth login"), _chunk("payment service")]
        assert context_recall(chunks, ["auth", "payment"]) == pytest.approx(1.0)

    def test_partial_recall(self):
        chunks = [_chunk("auth login")]
        assert context_recall(chunks, ["auth", "payment"]) == pytest.approx(0.5)

    def test_none_found(self):
        chunks = [_chunk("signup")]
        assert context_recall(chunks, ["auth", "payment"]) == 0.0

    def test_empty_expected_returns_zero(self):
        assert context_recall([_chunk("auth")], []) == 0.0

    def test_empty_retrieved(self):
        assert context_recall([], ["auth"]) == 0.0

    def test_case_insensitive(self):
        chunks = [_chunk("Authentication flow")]
        assert context_recall(chunks, ["authentication"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_retrieval_metrics
# ---------------------------------------------------------------------------

class TestComputeRetrievalMetrics:
    def test_returns_all_keys(self):
        chunks = [_chunk("auth")]
        metrics = compute_retrieval_metrics(chunks, ["auth"])
        expected_keys = {"mrr", "hit@10", "hit@5", "hit@1", "precision@10", "precision@5", "ndcg@10", "context_recall"}
        assert set(metrics.keys()) == expected_keys

    def test_all_values_are_floats(self):
        chunks = [_chunk("auth")]
        for val in compute_retrieval_metrics(chunks, ["auth"]).values():
            assert isinstance(val, float)

    def test_perfect_single_result(self):
        chunks = [_chunk("auth")]
        m = compute_retrieval_metrics(chunks, ["auth"])
        assert m["mrr"] == 1.0
        assert m["hit@1"] == 1.0
        assert m["context_recall"] == 1.0

    def test_empty_retrieved_all_zeros(self):
        m = compute_retrieval_metrics([], ["auth"])
        for val in m.values():
            assert val == 0.0


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def _make_result(self, metrics: dict) -> EvalResult:
        tc = EvalCase(id="x", query="q", expected_chunks=[])
        return EvalResult(test_case=tc, retrieved_chunks=[], metrics=metrics)

    def test_empty_returns_empty(self):
        assert aggregate_metrics([]) == {}

    def test_single_result_passthrough(self):
        r = self._make_result({"mrr": 0.8, "hit@10": 1.0})
        agg = aggregate_metrics([r])
        assert agg["mrr"] == pytest.approx(0.8)
        assert agg["hit@10"] == pytest.approx(1.0)

    def test_average_across_results(self):
        r1 = self._make_result({"mrr": 1.0})
        r2 = self._make_result({"mrr": 0.0})
        agg = aggregate_metrics([r1, r2])
        assert agg["mrr"] == pytest.approx(0.5)

    def test_missing_keys_treated_as_zero(self):
        r1 = self._make_result({"mrr": 1.0, "hit@10": 0.8})
        r2 = self._make_result({"hit@10": 0.4})  # no "mrr"
        agg = aggregate_metrics([r1, r2])
        assert agg["mrr"] == pytest.approx(0.5)  # (1.0 + 0) / 2
        assert agg["hit@10"] == pytest.approx(0.6)

    def test_keys_sorted_alphabetically(self):
        r = self._make_result({"z_metric": 0.1, "a_metric": 0.9})
        keys = list(aggregate_metrics([r]).keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# load_eval_dataset
# ---------------------------------------------------------------------------

class TestLoadEvalDataset:
    def _make_client(self, tmp_path: Path):
        client = MagicMock()
        client.path = tmp_path
        return client

    def test_load_eval_dataset_success(self, tmp_path: Path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        dataset = {
            "test_cases": [
                {
                    "id": "tc1",
                    "query": "what is auth",
                    "expected_chunks": ["auth"],
                    "ground_truth": "auth is authentication",
                    "tags": ["auth"],
                },
                {
                    "id": "tc2",
                    "query": "how to login",
                },
            ]
        }
        (tests_dir / "eval_dataset.json").write_text(json.dumps(dataset))

        client = self._make_client(tmp_path)
        cases = load_eval_dataset(client)

        assert len(cases) == 2
        assert cases[0] == EvalCase(
            id="tc1",
            query="what is auth",
            expected_chunks=["auth"],
            ground_truth="auth is authentication",
            tags=["auth"],
        )
        assert cases[1] == EvalCase(
            id="tc2",
            query="how to login",
            expected_chunks=[],
            ground_truth=None,
            tags=[],
        )

    def test_load_eval_dataset_file_not_found(self, tmp_path: Path):
        client = self._make_client(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_eval_dataset(client)

    def test_load_eval_dataset_empty_cases(self, tmp_path: Path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "eval_dataset.json").write_text(json.dumps({"test_cases": []}))

        client = self._make_client(tmp_path)
        cases = load_eval_dataset(client)
        assert cases == []


# ---------------------------------------------------------------------------
# _llm_judge
# ---------------------------------------------------------------------------

class TestLLMJudge:
    def test_llm_judge_success(self):
        provider = MagicMock()
        provider.generate.return_value = '{"score": 0.8, "reason": "good"}'
        result = _llm_judge(provider, "prompt")
        assert result == {"score": 0.8, "reason": "good"}

    def test_llm_judge_with_code_fence(self):
        provider = MagicMock()
        provider.generate.return_value = '```json\n{"score": 0.9, "reason": "great"}\n```'
        result = _llm_judge(provider, "prompt")
        assert result == {"score": 0.9, "reason": "great"}

    def test_llm_judge_all_retries_fail(self):
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("boom")
        result = _llm_judge(provider, "prompt", max_retries=2)
        assert result["score"] == 0.0
        assert "boom" in result["reason"]

    def test_llm_judge_unreachable_line(self):
        # max_retries=-1 makes the for-loop empty, hitting the final return
        provider = MagicMock()
        result = _llm_judge(provider, "prompt", max_retries=-1)
        assert result == {"score": 0.0, "reason": "unreachable"}


# ---------------------------------------------------------------------------
# compute_rag_metrics
# ---------------------------------------------------------------------------

class TestComputeRagMetrics:
    def test_compute_rag_metrics_no_ground_truth(self):
        provider = MagicMock()
        provider.generate.return_value = '{"score": 0.75, "reason": "ok"}'
        chunks = [{"content": "chunk1"}, {"content": "chunk2"}]
        metrics = compute_rag_metrics(provider, "q?", "ans", chunks)
        assert set(metrics.keys()) == {"faithfulness", "answer_relevance"}
        assert metrics["faithfulness"] == 0.75
        assert metrics["answer_relevance"] == 0.75

    def test_compute_rag_metrics_with_ground_truth(self):
        provider = MagicMock()
        provider.generate.return_value = '{"score": 0.6, "reason": "fine"}'
        chunks = [{"content": "chunk1"}, {"content": "chunk2"}]
        metrics = compute_rag_metrics(provider, "q?", "ans", chunks, ground_truth="gt")
        assert set(metrics.keys()) == {"faithfulness", "answer_relevance", "context_precision"}
        assert metrics["context_precision"] == 0.6


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    @patch("mullet.knowledge.engine.llm.build_rag_prompt")
    def test_generate_answer(self, mock_build):
        from mullet.knowledge.engine.llm import DEFAULT_SYSTEM_PROMPT
        mock_build.return_value = [{"role": "user", "content": "hello"}]
        provider = MagicMock()
        provider.generate.return_value = "the answer"
        chunks = [{"content": "chunk"}]
        result = generate_answer(provider, "q?", chunks, max_context_chars=500)
        assert result == "the answer"
        mock_build.assert_called_once_with("q?", chunks, max_context_chars=500)
        provider.generate.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=1000,
        )
