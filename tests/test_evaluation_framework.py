"""Tests for evaluation framework (LLM-as-judge and retrieval metrics)."""

import pytest
from unittest.mock import Mock

from mullet.knowledge.engine import evaluation as eval_mod


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = '{"score": 0.8, "reason": "test"}'):
        self.response = response
        self.call_count = 0

    def generate(self, messages, max_tokens=200, **kwargs):
        self.call_count += 1
        return self.response


# ===========================================================================
# Retrieval Metrics Tests
# ===========================================================================

class TestRetrievalMetrics:
    """Tests for retrieval quality metrics."""

    def test_mrr_first_match(self):
        """MRR = 1.0 when first result matches."""
        retrieved = [{"content": "auth system"}]
        expected = ["auth"]
        assert eval_mod.mrr(retrieved, expected) == 1.0

    def test_mrr_second_match(self):
        """MRR = 0.5 when second result matches."""
        retrieved = [{"content": "user"}, {"content": "auth system"}]
        expected = ["auth"]
        assert eval_mod.mrr(retrieved, expected) == 0.5

    def test_mrr_no_match(self):
        """MRR = 0.0 when no results match."""
        retrieved = [{"content": "user"}, {"content": "login"}]
        expected = ["auth"]
        assert eval_mod.mrr(retrieved, expected) == 0.0

    def test_mrr_empty_results(self):
        """MRR = 0.0 for empty results."""
        assert eval_mod.mrr([], ["auth"]) == 0.0

    def test_mrr_case_insensitive(self):
        """MRR matching is case-insensitive."""
        retrieved = [{"content": "AUTH System"}]
        expected = ["auth"]
        assert eval_mod.mrr(retrieved, expected) == 1.0

    def test_hit_at_k_true(self):
        """Hit@k = 1.0 when result found in top k."""
        retrieved = [
            {"content": "one"},
            {"content": "auth system"},
            {"content": "three"},
        ]
        expected = ["auth"]
        assert eval_mod.hit_at_k(retrieved, expected, k=2) == 1.0

    def test_hit_at_k_false(self):
        """Hit@k = 0.0 when result not in top k."""
        retrieved = [
            {"content": "one"},
            {"content": "two"},
            {"content": "auth system"},
        ]
        expected = ["auth"]
        assert eval_mod.hit_at_k(retrieved, expected, k=2) == 0.0

    def test_hit_at_k_empty_results(self):
        """Hit@k = 0.0 for empty results."""
        assert eval_mod.hit_at_k([], ["auth"], k=10) == 0.0

    def test_precision_at_k_perfect(self):
        """Precision@k = 1.0 when all top k are relevant."""
        retrieved = [
            {"content": "auth one"},
            {"content": "auth two"},
            {"content": "other"},
        ]
        expected = ["auth"]
        assert eval_mod.precision_at_k(retrieved, expected, k=2) == 1.0

    def test_precision_at_k_partial(self):
        """Precision@k = 0.5 when half of top k are relevant."""
        retrieved = [
            {"content": "auth one"},
            {"content": "not relevant"},
        ]
        expected = ["auth"]
        assert eval_mod.precision_at_k(retrieved, expected, k=2) == pytest.approx(0.5)

    def test_precision_at_k_zero(self):
        """Precision@k = 0.0 when no top k are relevant."""
        retrieved = [
            {"content": "user"},
            {"content": "login"},
        ]
        expected = ["auth"]
        assert eval_mod.precision_at_k(retrieved, expected, k=2) == 0.0

    def test_precision_at_k_zero_k(self):
        """Precision@k = 0.0 when k=0."""
        retrieved = [{"content": "auth"}]
        expected = ["auth"]
        assert eval_mod.precision_at_k(retrieved, expected, k=0) == 0.0

    def test_ndcg_perfect(self):
        """nDCG = 1.0 for perfect ranking."""
        retrieved = [
            {"content": "auth one"},
            {"content": "auth two"},
        ]
        expected = ["auth"]
        ndcg = eval_mod.ndcg(retrieved, expected, k=2)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_degraded(self):
        """nDCG < 1.0 when relevant results ranked lower."""
        retrieved = [
            {"content": "not relevant"},
            {"content": "auth system"},
            {"content": "auth two"},
        ]
        expected = ["auth"]
        ndcg = eval_mod.ndcg(retrieved, expected, k=3)
        assert 0 < ndcg < 1.0

    def test_ndcg_no_match(self):
        """nDCG = 0.0 when no results match."""
        retrieved = [
            {"content": "user"},
            {"content": "login"},
        ]
        expected = ["auth"]
        assert eval_mod.ndcg(retrieved, expected, k=2) == 0.0

    def test_context_recall_perfect(self):
        """Context recall = 1.0 when all expected patterns found."""
        retrieved = [
            {"content": "auth system and user management"},
        ]
        expected = ["auth", "user"]
        assert eval_mod.context_recall(retrieved, expected) == pytest.approx(1.0)

    def test_context_recall_partial(self):
        """Context recall = 0.5 when half of expected patterns found."""
        retrieved = [
            {"content": "auth system only"},
        ]
        expected = ["auth", "user"]
        assert eval_mod.context_recall(retrieved, expected) == pytest.approx(0.5)

    def test_context_recall_none(self):
        """Context recall = 0.0 when no patterns found."""
        retrieved = [
            {"content": "login flow"},
        ]
        expected = ["auth", "user"]
        assert eval_mod.context_recall(retrieved, expected) == 0.0

    def test_context_recall_empty_expected(self):
        """Context recall = 0.0 when expected is empty."""
        retrieved = [{"content": "something"}]
        assert eval_mod.context_recall(retrieved, []) == 0.0

    def test_compute_retrieval_metrics_all(self):
        """compute_retrieval_metrics returns all metrics."""
        retrieved = [
            {"content": "auth system"},
            {"content": "user management"},
        ]
        expected = ["auth"]
        metrics = eval_mod.compute_retrieval_metrics(retrieved, expected)

        assert "mrr" in metrics
        assert "hit@10" in metrics
        assert "hit@5" in metrics
        assert "hit@1" in metrics
        assert "precision@10" in metrics
        assert "precision@5" in metrics
        assert "ndcg@10" in metrics
        assert "context_recall" in metrics

        assert metrics["mrr"] == 1.0
        assert metrics["hit@1"] == 1.0


# ===========================================================================
# LLM-as-Judge Tests
# ===========================================================================

class TestLLMJudge:
    """Tests for LLM-as-judge evaluation."""

    def test_llm_judge_valid_response(self):
        """_llm_judge parses valid JSON response."""
        provider = MockProvider('{"score": 0.85, "reason": "good"}')
        result = eval_mod._llm_judge(provider, "test prompt")

        assert result["score"] == 0.85
        assert result["reason"] == "good"
        assert provider.call_count == 1

    def test_llm_judge_score_clamped_high(self):
        """_llm_judge clamps scores > 1.0 to 1.0."""
        provider = MockProvider('{"score": 1.5, "reason": "too high"}')
        result = eval_mod._llm_judge(provider, "test prompt")

        assert result["score"] == 1.0

    def test_llm_judge_score_clamped_low(self):
        """_llm_judge clamps negative scores to 0.0."""
        provider = MockProvider('{"score": -0.5, "reason": "too low"}')
        result = eval_mod._llm_judge(provider, "test prompt")

        assert result["score"] == 0.0

    def test_llm_judge_markdown_code_fence(self):
        """_llm_judge extracts JSON from markdown code fence."""
        response = '```json\n{"score": 0.9, "reason": "extracted"}\n```'
        provider = MockProvider(response)
        result = eval_mod._llm_judge(provider, "test prompt")

        assert result["score"] == 0.9
        assert result["reason"] == "extracted"

    def test_llm_judge_missing_score(self):
        """_llm_judge defaults to 0.0 if score missing."""
        provider = MockProvider('{"reason": "no score"}')
        result = eval_mod._llm_judge(provider, "test prompt")

        assert result["score"] == 0.0
        assert result["reason"] == "no score"

    def test_llm_judge_missing_reason(self):
        """_llm_judge defaults to empty string if reason missing."""
        provider = MockProvider('{"score": 0.7}')
        result = eval_mod._llm_judge(provider, "test prompt")

        assert result["score"] == 0.7
        assert result["reason"] == ""

    def test_llm_judge_invalid_json_retries(self):
        """_llm_judge retries on invalid JSON."""
        responses = [
            "invalid json {",
            '{"score": 0.75, "reason": "retry"}',
        ]
        provider = Mock()
        provider.generate = Mock(side_effect=responses)

        result = eval_mod._llm_judge(provider, "test prompt", max_retries=2)

        assert result["score"] == 0.75
        assert provider.generate.call_count == 2

    def test_llm_judge_max_retries_exhausted(self):
        """_llm_judge returns 0.0 after max retries."""
        provider = MockProvider("always invalid json {{{")
        result = eval_mod._llm_judge(provider, "test prompt", max_retries=1)

        assert result["score"] == 0.0
        assert "error" in result["reason"].lower()
        assert provider.call_count == 2


# ===========================================================================
# RAG Quality Metrics Tests
# ===========================================================================

class TestRAGMetrics:
    """Tests for RAG quality metrics (LLM-as-judge)."""

    def test_compute_rag_metrics_faithfulness(self):
        """compute_rag_metrics includes faithfulness score."""
        provider = MockProvider('{"score": 0.9, "reason": "faithful"}')
        chunks = [
            {"content": "The system uses OAuth 2.0 for authentication."},
        ]

        metrics = eval_mod.compute_rag_metrics(
            provider=provider,
            question="What authentication does the system use?",
            answer="The system uses OAuth 2.0.",
            retrieved_chunks=chunks,
        )

        assert "faithfulness" in metrics
        assert metrics["faithfulness"] == 0.9

    def test_compute_rag_metrics_answer_relevance(self):
        """compute_rag_metrics includes answer relevance score."""
        provider = MockProvider('{"score": 0.85, "reason": "relevant"}')
        chunks = [{"content": "Some context"}]

        metrics = eval_mod.compute_rag_metrics(
            provider=provider,
            question="What is X?",
            answer="X is something.",
            retrieved_chunks=chunks,
        )

        assert "answer_relevance" in metrics
        assert metrics["answer_relevance"] == 0.85

    def test_compute_rag_metrics_context_precision_with_ground_truth(self):
        """compute_rag_metrics includes context_precision when ground_truth provided."""
        provider = MockProvider('{"score": 0.8, "reason": "useful"}')
        chunks = [
            {"content": "Relevant chunk 1"},
            {"content": "Relevant chunk 2"},
        ]

        metrics = eval_mod.compute_rag_metrics(
            provider=provider,
            question="How does auth work?",
            answer="It uses OAuth.",
            retrieved_chunks=chunks,
            ground_truth="OAuth 2.0 protocol",
        )

        assert "context_precision" in metrics
        assert metrics["context_precision"] == 0.8

    def test_compute_rag_metrics_no_context_precision_without_ground_truth(self):
        """compute_rag_metrics skips context_precision when no ground_truth."""
        provider = MockProvider('{"score": 0.9, "reason": "test"}')
        chunks = [{"content": "Some context"}]

        metrics = eval_mod.compute_rag_metrics(
            provider=provider,
            question="What is X?",
            answer="X is Y.",
            retrieved_chunks=chunks,
            ground_truth=None,
        )

        assert "context_precision" not in metrics
        assert "faithfulness" in metrics
        assert "answer_relevance" in metrics

    def test_compute_rag_metrics_truncates_long_chunks(self):
        """compute_rag_metrics truncates very long chunks."""
        provider = MockProvider('{"score": 0.7, "reason": "test"}')
        long_content = "x" * 5000
        chunks = [{"content": long_content}]

        metrics = eval_mod.compute_rag_metrics(
            provider=provider,
            question="What is X?",
            answer="X is something.",
            retrieved_chunks=chunks,
        )

        assert "faithfulness" in metrics

    def test_compute_rag_metrics_handles_missing_content(self):
        """compute_rag_metrics handles chunks without content field."""
        provider = MockProvider('{"score": 0.8, "reason": "test"}')
        chunks = [
            {"text": "No content field here"},
            {},
        ]

        metrics = eval_mod.compute_rag_metrics(
            provider=provider,
            question="What is X?",
            answer="X is something.",
            retrieved_chunks=chunks,
        )

        assert "faithfulness" in metrics


# ===========================================================================
# Aggregation Tests
# ===========================================================================

class TestAggregation:
    """Tests for metric aggregation."""

    def test_aggregate_metrics_empty(self):
        """aggregate_metrics returns empty dict for empty results."""
        result = eval_mod.aggregate_metrics([])
        assert result == {}

    def test_aggregate_metrics_single(self):
        """aggregate_metrics with one result returns same metrics."""
        result = eval_mod.EvalResult(
            test_case=eval_mod.EvalCase("id1", "query", ["chunk1"]),
            retrieved_chunks=[],
            metrics={"mrr": 1.0, "ndcg": 0.95},
        )

        agg = eval_mod.aggregate_metrics([result])

        assert agg["mrr"] == 1.0
        assert agg["ndcg"] == 0.95

    def test_aggregate_metrics_multiple(self):
        """aggregate_metrics averages metrics across results."""
        results = [
            eval_mod.EvalResult(
                test_case=eval_mod.EvalCase("id1", "q1", ["c1"]),
                retrieved_chunks=[],
                metrics={"mrr": 1.0, "ndcg": 1.0, "hit@10": 1.0},
            ),
            eval_mod.EvalResult(
                test_case=eval_mod.EvalCase("id2", "q2", ["c2"]),
                retrieved_chunks=[],
                metrics={"mrr": 0.0, "ndcg": 0.0, "hit@10": 0.0},
            ),
        ]

        agg = eval_mod.aggregate_metrics(results)

        assert agg["mrr"] == 0.5
        assert agg["ndcg"] == 0.5
        assert agg["hit@10"] == 0.5

    def test_aggregate_metrics_different_keys(self):
        """aggregate_metrics handles results with different metric keys."""
        results = [
            eval_mod.EvalResult(
                test_case=eval_mod.EvalCase("id1", "q1", ["c1"]),
                retrieved_chunks=[],
                metrics={"mrr": 0.8, "ndcg": 0.9},
            ),
            eval_mod.EvalResult(
                test_case=eval_mod.EvalCase("id2", "q2", ["c2"]),
                retrieved_chunks=[],
                metrics={"mrr": 0.6, "faithfulness": 0.7},
            ),
        ]

        agg = eval_mod.aggregate_metrics(results)

        assert agg["mrr"] == 0.7
        assert agg["ndcg"] == 0.45
        assert agg["faithfulness"] == 0.35

    def test_aggregate_metrics_sorted(self):
        """aggregate_metrics returns sorted keys."""
        results = [
            eval_mod.EvalResult(
                test_case=eval_mod.EvalCase("id1", "q1", ["c1"]),
                retrieved_chunks=[],
                metrics={"z": 1.0, "a": 0.5, "m": 0.8},
            ),
        ]

        agg = eval_mod.aggregate_metrics(results)
        keys = list(agg.keys())

        assert keys == sorted(keys)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestEvaluationIntegration:
    pass
