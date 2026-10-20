"""Tests for the knowledge engine reranker module.

Tests API reranker and fallback logic with mocks (no real models).
LocalReranker tests require sentence-transformers — skipped if unavailable.
"""

import pytest
from unittest.mock import MagicMock, patch

from mullet.knowledge.engine.reranker import (
    APIReranker,
    FallbackReranker,
    create_reranker,
)


# --- APIReranker ---

def test_api_reranker_init():
    r = APIReranker(endpoint="http://localhost:8080", model="test-model")
    assert r.endpoint == "http://localhost:8080"
    assert r.model == "test-model"
    assert "Content-Type" in r.headers


def test_api_reranker_init_with_key():
    r = APIReranker(endpoint="http://localhost:8080", api_key="sk-test")
    assert r.headers["Authorization"] == "Bearer sk-test"


def test_api_reranker_init_strips_trailing_slash():
    r = APIReranker(endpoint="http://localhost:8080/")
    assert r.endpoint == "http://localhost:8080"


def test_api_reranker_empty_pairs():
    r = APIReranker(endpoint="http://localhost:8080")
    assert r.predict([]) == []


@patch("mullet.knowledge.engine.reranker.requests.post")
def test_api_reranker_predict(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.3},
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    r = APIReranker(endpoint="http://localhost:8080", model="test")
    scores = r.predict([("query", "doc1"), ("query", "doc2")])
    assert scores == [0.9, 0.3]
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "/v1/rerank" in call_args[0][0]


@patch("mullet.knowledge.engine.reranker.requests.post")
def test_api_reranker_sorts_by_index(mock_post):
    """Results returned out of order should be sorted by index."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.3},
            {"index": 0, "relevance_score": 0.9},
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    r = APIReranker(endpoint="http://localhost:8080")
    scores = r.predict([("q", "d1"), ("q", "d2")])
    assert scores == [0.9, 0.3]  # Sorted by index


@patch("mullet.knowledge.engine.reranker.requests.post")
def test_api_reranker_network_error(mock_post):
    mock_post.side_effect = ConnectionError("refused")
    r = APIReranker(endpoint="http://localhost:8080")
    with pytest.raises(ConnectionError):
        r.predict([("query", "doc")])


def test_api_reranker_close():
    r = APIReranker(endpoint="http://localhost:8080")
    r.close()  # Should not raise


# --- FallbackReranker ---

def test_fallback_uses_api_first():
    api = MagicMock()
    api.predict.return_value = [0.8, 0.2]
    api.close = MagicMock()

    fb = FallbackReranker(api, local_model="test")
    scores = fb.predict([("q", "d1"), ("q", "d2")])
    assert scores == [0.8, 0.2]
    api.predict.assert_called_once()


def test_fallback_switches_to_local_on_failure():
    api = MagicMock()
    api.predict.side_effect = ConnectionError("down")
    api.close = MagicMock()

    mock_local = MagicMock()
    mock_local.predict.return_value = [0.5]

    fb = FallbackReranker(api, local_model="test")
    fb._local = mock_local  # Inject mock to avoid real model load

    scores = fb.predict([("q", "d")])
    assert scores == [0.5]
    assert fb._api_failed is True


def test_fallback_stays_on_local_after_failure():
    api = MagicMock()
    api.predict.side_effect = ConnectionError("down")
    api.close = MagicMock()

    mock_local = MagicMock()
    mock_local.predict.return_value = [0.7]

    fb = FallbackReranker(api, local_model="test")
    fb._local = mock_local
    fb._api_failed = True  # Already failed

    scores = fb.predict([("q", "d")])
    assert scores == [0.7]
    api.predict.assert_not_called()  # Should not retry API


def test_fallback_close():
    api = MagicMock()
    mock_local = MagicMock()

    fb = FallbackReranker(api, local_model="test")
    fb._local = mock_local
    fb.close()
    api.close.assert_called_once()
    mock_local.close.assert_called_once()


# --- create_reranker ---

def test_create_reranker_api_requires_endpoint():
    with pytest.raises(ValueError, match="endpoint"):
        create_reranker(backend="api")


@patch("mullet.knowledge.engine.reranker.LocalReranker")
def test_create_reranker_local(mock_local_cls):
    mock_local_cls.return_value = MagicMock()
    r = create_reranker(backend="local", model="test-model")
    mock_local_cls.assert_called_once_with(model="test-model", device=None)


def test_create_reranker_api_returns_fallback():
    r = create_reranker(backend="api", endpoint="http://localhost:8080", model="test")
    assert isinstance(r, FallbackReranker)


# --- LocalReranker (mocked torch + sentence_transformers) ---

from mullet.knowledge.engine.reranker import LocalReranker


def _make_torch_mock(cuda: bool = False):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = cuda
    return mock_torch


def test_local_reranker_init_cpu():
    """Lines 40-56: LocalReranker init on CPU (device auto-detected)."""
    import sys
    mock_torch = _make_torch_mock(cuda=False)
    mock_ce_cls = MagicMock()
    mock_ce_instance = MagicMock()
    mock_ce_cls.return_value = mock_ce_instance

    fake_st = MagicMock()
    fake_st.CrossEncoder = mock_ce_cls
    with patch.dict(sys.modules, {"torch": mock_torch, "sentence_transformers": fake_st}):
        r = LocalReranker(model="test-model")
    assert r._device == "cpu"
    mock_ce_cls.assert_called_once_with("test-model", device="cpu")


def test_local_reranker_init_cuda_fallback():
    """Lines 48-53: LocalReranker CUDA fails → falls back to CPU."""
    import sys
    mock_torch = _make_torch_mock(cuda=True)
    mock_ce_cls = MagicMock()
    cuda_instance = MagicMock()
    cpu_instance = MagicMock()
    mock_ce_cls.side_effect = [RuntimeError("no cuda"), cpu_instance]

    fake_st = MagicMock()
    fake_st.CrossEncoder = mock_ce_cls
    with patch.dict(sys.modules, {"torch": mock_torch, "sentence_transformers": fake_st}):
        r = LocalReranker(model="test-model", device="cuda")
    assert r._model is cpu_instance


def test_local_reranker_init_runtime_error_on_cpu_reraises():
    """Line 53: RuntimeError on non-CUDA device re-raises (not swallowed)."""
    import sys
    mock_torch = _make_torch_mock(cuda=False)
    mock_ce_cls = MagicMock()
    mock_ce_cls.side_effect = RuntimeError("bad model")

    fake_st = MagicMock()
    fake_st.CrossEncoder = mock_ce_cls
    with patch.dict(sys.modules, {"torch": mock_torch, "sentence_transformers": fake_st}):
        with pytest.raises(RuntimeError, match="bad model"):
            LocalReranker(model="test-model", device="cpu")


def test_local_reranker_predict():
    """Lines 60-61: LocalReranker.predict converts scores to float."""
    import sys
    mock_torch = _make_torch_mock(cuda=False)
    mock_ce_instance = MagicMock()
    mock_ce_instance.predict.return_value = [0.9, 0.1]
    mock_ce_cls = MagicMock(return_value=mock_ce_instance)

    fake_st = MagicMock()
    fake_st.CrossEncoder = mock_ce_cls
    with patch.dict(sys.modules, {"torch": mock_torch, "sentence_transformers": fake_st}):
        r = LocalReranker(model="test-model")
    scores = r.predict([("q", "d1"), ("q", "d2")])
    assert scores == [0.9, 0.1]
    assert all(isinstance(s, float) for s in scores)


def test_local_reranker_close():
    """Lines 65-67: LocalReranker.close deletes model and sets to None."""
    import sys
    mock_torch = _make_torch_mock(cuda=False)
    mock_ce_instance = MagicMock()
    mock_ce_cls = MagicMock(return_value=mock_ce_instance)

    fake_st = MagicMock()
    fake_st.CrossEncoder = mock_ce_cls
    with patch.dict(sys.modules, {"torch": mock_torch, "sentence_transformers": fake_st}):
        r = LocalReranker(model="test-model")
    r.close()
    assert r._model is None


def test_fallback_get_local_lazy_init():
    """Lines 147-148: FallbackReranker._get_local initializes LocalReranker on first call."""
    api = MagicMock()
    mock_local_instance = MagicMock()
    with patch("mullet.knowledge.engine.reranker.LocalReranker", return_value=mock_local_instance) as mock_cls:
        fb = FallbackReranker(api, local_model="test-model", device="cpu")
        assert fb._local is None
        local = fb._get_local()
        assert local is mock_local_instance
        mock_cls.assert_called_once_with("test-model", "cpu")
        # Second call returns same instance without re-initialising
        local2 = fb._get_local()
        assert local2 is mock_local_instance
        assert mock_cls.call_count == 1
