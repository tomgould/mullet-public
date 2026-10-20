"""Tests for the reranker module."""

import pytest
from mullet.core.knowledge.reranker import (
    SimpleReranker, get_reranker, _tokenize,
)


# ── _tokenize ────────────────────────────────────────────────────────────────

def test_tokenize_basic():
    assert _tokenize("Hello World") == ["hello", "world"]

def test_tokenize_punctuation():
    assert _tokenize("it's a test!") == ["it", "s", "a", "test"]

def test_tokenize_empty():
    assert _tokenize("") == []

def test_tokenize_numbers():
    tokens = _tokenize("Python 3.10")
    assert "python" in tokens
    assert "3" in tokens
    assert "10" in tokens


# ── SimpleReranker ───────────────────────────────────────────────────────────

@pytest.fixture
def reranker():
    return SimpleReranker()


@pytest.fixture
def sample_results():
    return [
        {"content": "Python is a great programming language for data science."},
        {"content": "JavaScript frameworks like React are popular for web development."},
        {"content": "Python data science libraries include pandas, numpy, and scikit-learn."},
        {"content": "Go is used for building scalable backend services."},
        {"content": "Machine learning with Python uses TensorFlow and PyTorch."},
    ]


def test_rerank_basic(reranker, sample_results):
    results = reranker.rerank("Python data science", sample_results, top_k=3)
    assert len(results) == 3
    # Python data science results should rank higher
    names = [r["content"][:10] for r in results]
    assert any("Python" in r["content"] for r in results[:2])


def test_rerank_preserves_content(reranker, sample_results):
    results = reranker.rerank("Python", sample_results, top_k=5)
    for r in results:
        assert "content" in r
        assert "rerank_score" in r


def test_rerank_scores_descending(reranker, sample_results):
    results = reranker.rerank("Python programming", sample_results)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_top_k_limits(reranker, sample_results):
    results = reranker.rerank("Python", sample_results, top_k=2)
    assert len(results) == 2


def test_rerank_empty_query(reranker, sample_results):
    results = reranker.rerank("", sample_results, top_k=3)
    assert len(results) == 3


def test_rerank_empty_results(reranker):
    results = reranker.rerank("query", [], top_k=5)
    assert results == []


def test_rerank_exact_phrase_boost(reranker):
    results = [
        {"content": "Python data science is powerful"},
        {"content": "data and science with Python tools"},
    ]
    ranked = reranker.rerank("Python data science", results)
    # Exact phrase match should score higher
    assert ranked[0]["rerank_score"] >= ranked[1]["rerank_score"]


def test_rerank_preserves_metadata(reranker):
    results = [
        {"content": "Hello world", "source": "test.py", "chunk_id": 1},
    ]
    ranked = reranker.rerank("hello", results)
    assert ranked[0]["source"] == "test.py"
    assert ranked[0]["chunk_id"] == 1


def test_rerank_irrelevant_query(reranker, sample_results):
    results = reranker.rerank("quantum physics nuclear", sample_results)
    # All scores should be low
    assert all(r["rerank_score"] < 1.0 for r in results)


def test_rerank_single_result(reranker):
    results = [{"content": "Only one result here"}]
    ranked = reranker.rerank("one result", results)
    assert len(ranked) == 1
    assert ranked[0]["rerank_score"] > 0


# ── get_reranker ─────────────────────────────────────────────────────────────

def test_get_reranker_default():
    r = get_reranker()
    assert isinstance(r, SimpleReranker)


def test_get_reranker_with_model():
    r = get_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Should return CrossEncoderReranker (lazy-loads model)
    assert hasattr(r, 'model_name')
    assert r.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
