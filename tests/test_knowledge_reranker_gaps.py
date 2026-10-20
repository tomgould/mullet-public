"""Gap tests for mullet.core.knowledge.reranker — targets uncovered lines."""

import pytest
from unittest.mock import MagicMock, patch

from mullet.core.knowledge.reranker import (
    SimpleReranker,
    CrossEncoderReranker,
    get_reranker,
    _tokenize,
)


# ── SimpleReranker uncovered lines ───────────────────────────────────────────

def test_rerank_query_with_no_tokens():
    """Line 54: query that tokenizes to nothing (e.g. punctuation only)."""
    r = SimpleReranker()
    results = [{"content": "hello world"}]
    out = r.rerank("!!!", results, top_k=1)
    assert out == results[:1]


def test_score_empty_content_tokens():
    """Line 88: content that tokenizes to nothing yields 0.0 score."""
    r = SimpleReranker()
    score = r._score("!!!", ["hello"], {"hello": 1}, "hello")
    assert score == 0.0


# ── CrossEncoderReranker uncovered lines ─────────────────────────────────────

def test_cross_encoder_lazy_load_and_import_error():
    """Lines 132-141: lazy model load + ImportError when sentence-transformers missing."""
    ce = CrossEncoderReranker("test-model")
    assert ce._model is None
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(ImportError, match="sentence-transformers required"):
            _ = ce.model


def test_cross_encoder_rerank_empty_query():
    """Line 145: empty query returns early."""
    ce = CrossEncoderReranker("test-model")
    results = [{"content": "doc"}]
    # Early return happens before self.model is accessed; no need to mock model.
    out = ce.rerank("", results, top_k=5)
    assert out == results[:5]


def test_cross_encoder_rerank_empty_results():
    """Line 145: empty results returns early."""
    ce = CrossEncoderReranker("test-model")
    # Early return happens before self.model is accessed; no need to mock model.
    out = ce.rerank("query", [], top_k=5)
    assert out == []


def test_cross_encoder_rerank_success():
    """Lines 148-160: normal rerank path with mocked model."""
    ce = CrossEncoderReranker("test-model")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.2, 0.8]
    with patch.object(ce, "_model", mock_model):
        results = [{"content": "a"}, {"content": "b"}]
        out = ce.rerank("q", results, top_k=2)
    assert len(out) == 2
    # highest score first
    assert out[0]["rerank_score"] == 0.8
    assert out[1]["rerank_score"] == 0.2
    assert "rerank_score" in out[0]


def test_cross_encoder_rerank_top_k_slice():
    """Lines 155-158: top_k slices output correctly."""
    ce = CrossEncoderReranker("test-model")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5]
    with patch.object(ce, "_model", mock_model):
        results = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        out = ce.rerank("q", results, top_k=2)
    assert len(out) == 2
    assert out[0]["rerank_score"] == 0.9


def test_cross_encoder_model_lazy_load_success():
    """Line 135: successful lazy load of the cross-encoder model."""
    ce = CrossEncoderReranker("test-model")
    mock_model = MagicMock()
    with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
        model = ce.model
    assert model is mock_model
    assert ce._model is mock_model


# ── get_reranker ─────────────────────────────────────────────────────────────

def test_get_reranker_cross_encoder_type():
    """Ensure get_reranker with a name returns CrossEncoderReranker."""
    r = get_reranker("cross-encoder/test")
    assert isinstance(r, CrossEncoderReranker)


def test_get_reranker_none_returns_simple():
    """Line 166: get_reranker(None) returns SimpleReranker."""
    r = get_reranker(None)
    assert isinstance(r, SimpleReranker)


# ── SimpleReranker uncovered lines ───────────────────────────────────────────

def test_simple_reranker_empty_query():
    """Line 50: empty query returns early."""
    r = SimpleReranker()
    results = [{"content": "hello"}]
    assert r.rerank("", results, top_k=1) == results[:1]


def test_simple_reranker_empty_results():
    """Line 50: empty results returns early."""
    r = SimpleReranker()
    assert r.rerank("query", [], top_k=5) == []


def test_simple_reranker_basic_scoring():
    """Lines 56-76, 90-115: normal rerank path with scoring and sorting."""
    r = SimpleReranker()
    results = [
        {"content": "python data science"},
        {"content": "javascript frameworks"},
    ]
    out = r.rerank("python data", results, top_k=2)
    assert len(out) == 2
    assert "rerank_score" in out[0]
    scores = [o["rerank_score"] for o in out]
    assert scores == sorted(scores, reverse=True)


def test_simple_reranker_exact_phrase_boost():
    """Line 109-110: exact phrase match gets boosted."""
    r = SimpleReranker(phrase_boost=2.0)
    results = [
        {"content": "python data science is great"},
        {"content": "data and science with python"},
    ]
    out = r.rerank("python data science", results, top_k=2)
    # Exact phrase should win
    assert out[0]["content"] == "python data science is great"


def test_simple_reranker_short_content_penalty():
    """Line 113: very short content gets length penalty."""
    r = SimpleReranker()
    score = r._score("hi", ["hello"], {"hello": 1}, "hello")
    # Short content should still compute without error
    assert isinstance(score, float)


def test_simple_reranker_no_phrase_boost_for_single_token():
    """Line 109: single-token query should not get phrase boost."""
    r = SimpleReranker()
    results = [{"content": "python"}]
    out = r.rerank("python", results, top_k=1)
    assert len(out) == 1
    assert "rerank_score" in out[0]
