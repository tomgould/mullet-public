"""Tests for mullet.knowledge.engine.evaluation."""

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mullet.knowledge.engine.evaluation import (
    EvalCase,
    EvalResult,
    aggregate_metrics,
    compute_rag_metrics,
    compute_retrieval_metrics,
    context_recall,
    generate_answer,
    hit_at_k,
    load_eval_dataset,
    mrr,
    ndcg,
    precision_at_k,
    _chunk_matches,
    _llm_judge,
)


# --- load_eval_dataset -------------------------------------------------------


def test_load_eval_dataset_success(tmp_path):
    dataset_dir = tmp_path / "tests"
    dataset_dir.mkdir()
    data = {
        "test_cases": [
            {
                "id": "tc1",
                "query": "What is RAG?",
                "expected_chunks": ["Retrieval", "Augmented", "Generation"],
                "ground_truth": "RAG is retrieval augmented generation.",
                "tags": ["basic", "rag"],
            }
        ]
    }
    (dataset_dir / "eval_dataset.json").write_text(json.dumps(data))

    client = SimpleNamespace(path=tmp_path)
    cases = load_eval_dataset(client)
    assert len(cases) == 1
    case = cases[0]
    assert case.id == "tc1"
    assert case.query == "What is RAG?"
    assert case.expected_chunks == ["Retrieval", "Augmented", "Generation"]
    assert case.ground_truth == "RAG is retrieval augmented generation."
    assert case.tags == ["basic", "rag"]


def test_load_eval_dataset_file_not_found(tmp_path):
    client = SimpleNamespace(path=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_eval_dataset(client)


def test_load_eval_dataset_empty_test_cases(tmp_path):
    dataset_dir = tmp_path / "tests"
    dataset_dir.mkdir()
    (dataset_dir / "eval_dataset.json").write_text(json.dumps({"test_cases": []}))
    client = SimpleNamespace(path=tmp_path)
    assert load_eval_dataset(client) == []


def test_load_eval_dataset_full_fields(tmp_path):
    dataset_dir = tmp_path / "tests"
    dataset_dir.mkdir()
    data = {
        "test_cases": [
            {"id": "1", "query": "q1", "expected_chunks": ["a"], "ground_truth": "gt1", "tags": ["t1"]},
            {"id": "2", "query": "q2", "expected_chunks": [], "ground_truth": None, "tags": []},
        ]
    }
    (dataset_dir / "eval_dataset.json").write_text(json.dumps(data))
    client = SimpleNamespace(path=tmp_path)
    cases = load_eval_dataset(client)
    assert len(cases) == 2
    assert cases[0].id == "1"
    assert cases[1].expected_chunks == []
    assert cases[1].ground_truth is None
    assert cases[1].tags == []


# --- _chunk_matches ----------------------------------------------------------


def test_chunk_matches_hit():
    assert _chunk_matches({"content": "hello world"}, ["world"]) is True


def test_chunk_matches_miss():
    assert _chunk_matches({"content": "hello world"}, ["foo"]) is False


def test_chunk_matches_case_insensitive():
    assert _chunk_matches({"content": "Hello World"}, ["world"]) is True
    assert _chunk_matches({"content": "hello world"}, ["WORLD"]) is True


# --- mrr ---------------------------------------------------------------------


def test_mrr_first_hit():
    chunks = [{"content": "alpha"}, {"content": "beta"}]
    assert mrr(chunks, ["alpha"]) == 1.0


def test_mrr_second_hit():
    chunks = [{"content": "alpha"}, {"content": "beta"}]
    assert mrr(chunks, ["beta"]) == 0.5


def test_mrr_no_hit():
    chunks = [{"content": "alpha"}]
    assert mrr(chunks, ["gamma"]) == 0.0


# --- hit_at_k ----------------------------------------------------------------


def test_hit_at_k_hit_in_range():
    chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
    assert hit_at_k(chunks, ["b"], k=2) == 1.0


def test_hit_at_k_hit_out_of_range():
    chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
    assert hit_at_k(chunks, ["c"], k=2) == 0.0


def test_hit_at_k_no_hit():
    chunks = [{"content": "a"}]
    assert hit_at_k(chunks, ["z"], k=5) == 0.0


# --- precision_at_k ----------------------------------------------------------


def test_precision_at_k_zero():
    assert precision_at_k([], ["a"], k=0) == 0.0


def test_precision_at_k_all_relevant():
    chunks = [{"content": "a"}, {"content": "b"}]
    assert precision_at_k(chunks, ["a", "b"], k=2) == 1.0


def test_precision_at_k_half_relevant():
    chunks = [{"content": "a"}, {"content": "z"}]
    assert precision_at_k(chunks, ["a"], k=2) == 0.5


# --- ndcg --------------------------------------------------------------------


def test_ndcg_perfect():
    chunks = [{"content": "a"}, {"content": "b"}]
    assert ndcg(chunks, ["a", "b"], k=2) == 1.0


def test_ndcg_partial():
    chunks = [{"content": "z"}, {"content": "a"}]
    dcg = 0.0 / math.log2(2) + 1.0 / math.log2(3)
    idcg = 1.0 / math.log2(2)
    assert ndcg(chunks, ["a"], k=2) == pytest.approx(dcg / idcg)


def test_ndcg_no_relevant():
    chunks = [{"content": "z"}]
    assert ndcg(chunks, ["a"], k=1) == 0.0


# --- context_recall ----------------------------------------------------------


def test_context_recall_empty_expected():
    assert context_recall([{"content": "a"}], []) == 0.0


def test_context_recall_all_found():
    chunks = [{"content": "foo bar baz"}]
    assert context_recall(chunks, ["foo", "bar", "baz"]) == 1.0


def test_context_recall_partial():
    chunks = [{"content": "foo bar"}]
    assert context_recall(chunks, ["foo", "baz"]) == 0.5


# --- compute_retrieval_metrics -----------------------------------------------


def test_compute_retrieval_metrics_has_all_keys():
    chunks = [{"content": "relevant"}]
    metrics = compute_retrieval_metrics(chunks, ["relevant"], k=10)
    expected_keys = {
        "mrr", "hit@10", "hit@5", "hit@1",
        "precision@10", "precision@5", "ndcg@10", "context_recall",
    }
    assert set(metrics.keys()) == expected_keys


# --- _llm_judge --------------------------------------------------------------


def test_llm_judge_success():
    provider = MagicMock()
    provider.generate.return_value = '{"score": 0.75, "reason": "good"}'
    result = _llm_judge(provider, "prompt")
    assert result["score"] == 0.75
    assert result["reason"] == "good"


def test_llm_judge_code_fence_stripping():
    provider = MagicMock()
    provider.generate.return_value = '```json\n{"score": 0.8, "reason": "ok"}\n```'
    result = _llm_judge(provider, "prompt")
    assert result["score"] == 0.8
    assert result["reason"] == "ok"


def test_llm_judge_score_clamp_high():
    provider = MagicMock()
    provider.generate.return_value = '{"score": 1.5, "reason": "high"}'
    result = _llm_judge(provider, "prompt")
    assert result["score"] == 1.0


def test_llm_judge_score_clamp_low():
    provider = MagicMock()
    provider.generate.return_value = '{"score": -0.5, "reason": "low"}'
    result = _llm_judge(provider, "prompt")
    assert result["score"] == 0.0


def test_llm_judge_all_retries_exhausted():
    provider = MagicMock()
    provider.generate.side_effect = Exception("boom")
    result = _llm_judge(provider, "prompt", max_retries=2)
    assert result["score"] == 0.0
    assert "LLM judge error" in result["reason"]


# --- compute_rag_metrics -----------------------------------------------------


def test_compute_rag_metrics_without_ground_truth():
    provider = MagicMock()
    provider.generate.return_value = '{"score": 0.9, "reason": "ok"}'
    chunks = [{"content": "context"}]
    metrics = compute_rag_metrics(provider, "q", "a", chunks)
    assert set(metrics.keys()) == {"faithfulness", "answer_relevance"}
    assert metrics["faithfulness"] == 0.9
    assert metrics["answer_relevance"] == 0.9


def test_compute_rag_metrics_with_ground_truth():
    provider = MagicMock()
    provider.generate.return_value = '{"score": 0.8, "reason": "ok"}'
    chunks = [{"content": "context"}]
    metrics = compute_rag_metrics(provider, "q", "a", chunks, ground_truth="gt")
    assert set(metrics.keys()) == {"faithfulness", "answer_relevance", "context_precision"}
    assert metrics["context_precision"] == 0.8


# --- generate_answer ---------------------------------------------------------


def test_generate_answer():
    from unittest.mock import ANY
    provider = MagicMock()
    provider.generate.return_value = "generated answer"
    chunks = [{"content": "chunk1"}]
    with patch("mullet.knowledge.engine.llm.build_rag_prompt") as mock_build:
        mock_build.return_value = [{"role": "user", "content": "hi"}]
        answer = generate_answer(provider, "question", chunks)
    assert answer == "generated answer"
    provider.generate.assert_called_once_with(
        [{"role": "user", "content": "hi"}],
        system_prompt=ANY,
        max_tokens=1000,
    )


# --- aggregate_metrics -------------------------------------------------------


def test_aggregate_metrics_empty():
    assert aggregate_metrics([]) == {}


def test_aggregate_metrics_single():
    case = EvalCase(id="1", query="q", expected_chunks=[])
    result = EvalResult(test_case=case, retrieved_chunks=[], metrics={"mrr": 0.5})
    assert aggregate_metrics([result]) == {"mrr": 0.5}


def test_aggregate_metrics_multi_different_keys():
    case = EvalCase(id="1", query="q", expected_chunks=[])
    r1 = EvalResult(test_case=case, retrieved_chunks=[], metrics={"mrr": 0.5, "hit": 1.0})
    r2 = EvalResult(test_case=case, retrieved_chunks=[], metrics={"mrr": 1.0, "ndcg": 0.8})
    agg = aggregate_metrics([r1, r2])
    assert agg["mrr"] == 0.75
    assert agg["hit"] == 0.5
    assert agg["ndcg"] == 0.4
    assert list(agg.keys()) == sorted(agg.keys())
