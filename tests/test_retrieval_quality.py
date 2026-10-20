"""Tests for the RAG retrieval quality measurement framework."""

import json
import pytest

# Import from the knowledge quality module
from mullet.core.knowledge import evaluation as quality_mod


def test_perfect_retrieval():
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    r = quality_mod.measure_retrieval(results, expected_ids=["a", "b"])
    assert r.mrr == 1.0
    assert r.hit_at_k[1] is True


def test_no_relevant_results():
    r = quality_mod.measure_retrieval([{"chunk_id": "x"}], expected_ids=["a"])
    assert r.mrr == 0.0
    assert r.ndcg == 0.0


def test_mrr_second():
    r = quality_mod.measure_retrieval([{"chunk_id": "x"}, {"chunk_id": "a"}], expected_ids=["a"])
    assert r.mrr == 0.5


def test_hit_at_k():
    results = [{"chunk_id": f"r{i}"} for i in range(10)]
    results[4] = {"chunk_id": "target"}
    r = quality_mod.measure_retrieval(results, expected_ids=["target"], k_values=[1, 5, 10])
    assert r.hit_at_k[1] is False
    assert r.hit_at_k[5] is True


def test_precision_at_k():
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    r = quality_mod.measure_retrieval(results, expected_ids=["a", "c"], k_values=[3])
    assert r.precision_at_k[3] == pytest.approx(1.0)


def test_empty():
    r = quality_mod.measure_retrieval([], expected_ids=["a"])
    assert r.mrr == 0.0


def test_ndcg_perfect():
    assert quality_mod._compute_ndcg(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_ndcg_none():
    assert quality_mod._compute_ndcg(["x", "y"], {"a"}, k=2) == 0.0


def test_aggregate():
    r1 = quality_mod.RetrievalResult(query="q1", mrr=1.0, ndcg=1.0, hit_at_k={1: True}, precision_at_k={1: 1.0})
    r2 = quality_mod.RetrievalResult(query="q2", mrr=0.0, ndcg=0.0, hit_at_k={1: False}, precision_at_k={1: 0.0})
    report = quality_mod.aggregate_results([r1, r2])
    assert report.mean_mrr == 0.5
    assert report.hit_rates[1] == 0.5
    assert report.total_queries == 2


def test_report_summary():
    report = quality_mod.QualityReport(total_queries=10, mean_mrr=0.75, mean_ndcg=0.82,
                                        hit_rates={1: 0.6}, mean_precision={1: 0.6})
    assert "0.750" in report.summary()


def test_load_cases(tmp_path):
    data = [{"query": "auth", "expected_ids": ["a1"]}]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    cases = quality_mod.load_test_cases(path)
    assert len(cases) == 1
    assert cases[0].query == "auth"
