"""Tests for mullet.core.knowledge.evaluation — RAG retrieval metrics."""

import json

import pytest

from mullet.core.knowledge.evaluation import (
    QualityReport,
    RetrievalResult,
    _compute_ndcg,
    aggregate_results,
    load_test_cases,
    measure_retrieval,
)


# --- load_test_cases -------------------------------------------------------


def test_load_test_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"query": "q1", "expected_ids": ["a", "b"]},
        {"query": "q2", "expected_ids": ["c"]},
    ]))
    cases = load_test_cases(path)
    assert len(cases) == 2
    assert cases[0].query == "q1"
    assert cases[0].expected_ids == ["a", "b"]


def test_load_test_cases_with_expected_answer(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"query": "q1", "expected_ids": ["a"], "expected_answer": "yes"},
    ]))
    cases = load_test_cases(path)
    assert cases[0].expected_answer == "yes"


# --- measure_retrieval -----------------------------------------------------


def test_measure_retrieval_perfect_top_hit():
    """Expected ID at position 0 → MRR == 1.0, Hit@k True for all k."""
    results = [{"chunk_id": "a"}, {"chunk_id": "x"}, {"chunk_id": "y"}]
    rv = measure_retrieval(results, ["a"])
    assert rv.mrr == 1.0
    assert rv.hit_at_k[1] is True
    assert rv.hit_at_k[3] is True


def test_measure_retrieval_hit_at_third_position():
    """Expected ID at position 2 (3rd) → MRR == 1/3."""
    results = [{"chunk_id": "x"}, {"chunk_id": "y"}, {"chunk_id": "a"}]
    rv = measure_retrieval(results, ["a"])
    assert rv.mrr == pytest.approx(1/3)
    assert rv.hit_at_k[1] is False
    assert rv.hit_at_k[3] is True


def test_measure_retrieval_no_hit():
    """No expected IDs in results → MRR 0, all Hit@k False."""
    results = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    rv = measure_retrieval(results, ["a"])
    assert rv.mrr == 0.0
    assert all(not hit for hit in rv.hit_at_k.values())


def test_measure_retrieval_supports_id_field():
    """'id' field should work as fallback for 'chunk_id'."""
    results = [{"id": "a"}, {"id": "b"}]
    rv = measure_retrieval(results, ["b"])
    assert rv.mrr == 0.5


def test_measure_retrieval_empty_expected_no_crash():
    """No expected IDs → P@k = 0, Hit@k = False."""
    results = [{"chunk_id": "a"}]
    rv = measure_retrieval(results, [])
    assert rv.mrr == 0.0
    assert rv.precision_at_k[1] == 0.0


def test_measure_retrieval_custom_k_values():
    """k_values parameter controls which k values are computed."""
    results = [{"chunk_id": "a"}]
    rv = measure_retrieval(results, ["a"], k_values=[1, 2])
    assert 1 in rv.hit_at_k
    assert 2 in rv.hit_at_k
    assert 10 not in rv.hit_at_k


# --- aggregate_results -----------------------------------------------------


def test_aggregate_results_empty_returns_empty_report():
    report = aggregate_results([])
    assert report.total_queries == 0
    assert report.mean_mrr == 0.0


def test_aggregate_results_mean_mrr():
    r1 = RetrievalResult(query="q1", mrr=1.0)
    r2 = RetrievalResult(query="q2", mrr=0.5)
    report = aggregate_results([r1, r2])
    assert report.mean_mrr == 0.75
    assert report.total_queries == 2


def test_aggregate_results_hit_rates():
    r1 = RetrievalResult(query="q1", hit_at_k={1: True, 3: True})
    r2 = RetrievalResult(query="q2", hit_at_k={1: False, 3: True})
    report = aggregate_results([r1, r2])
    assert report.hit_rates[1] == 0.5
    assert report.hit_rates[3] == 1.0


# --- _compute_ndcg ---------------------------------------------------------


def test_compute_ndcg_perfect_ranking():
    """All expected IDs at the top → nDCG = 1.0."""
    retrieved = ["a", "b", "c"]
    expected = {"a", "b", "c"}
    assert _compute_ndcg(retrieved, expected) == 1.0


def test_compute_ndcg_no_hits():
    """No expected in retrieved → 0.0."""
    retrieved = ["x", "y", "z"]
    expected = {"a"}
    assert _compute_ndcg(retrieved, expected) == 0.0


def test_compute_ndcg_partial_hit():
    """Partial hits produce values 0 < ndcg < 1."""
    retrieved = ["a", "x", "y"]
    expected = {"a", "b"}
    ndcg = _compute_ndcg(retrieved, expected)
    assert 0.0 < ndcg < 1.0


def test_compute_ndcg_empty_expected():
    """Empty expected set → ndcg 0 (no ideal to divide by)."""
    assert _compute_ndcg(["a", "b"], set()) == 0.0


# --- QualityReport.summary -------------------------------------------------


def test_quality_report_summary():
    report = QualityReport(
        mean_mrr=0.5, mean_ndcg=0.75, hit_rates={1: 0.5, 3: 1.0},
        mean_precision={1: 0.5, 3: 0.75}, total_queries=10,
    )
    s = report.summary()
    assert "10 queries" in s
    assert "MRR:" in s
    assert "0.500" in s
    assert "0.750" in s
