"""Tests for mullet.knowledge.engine.model_cache.ModelCache singleton."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from mullet.knowledge.engine.model_cache import ModelCache, get_model_cache


@pytest.fixture
def _reset_singleton(monkeypatch):
    """Reset the module-level _instance between tests."""
    import mullet.knowledge.engine.model_cache as mc
    monkeypatch.setattr(mc, "_instance", None)
    yield
    monkeypatch.setattr(mc, "_instance", None)


# --- get_model_cache singleton --------------------------------------------


def test_get_model_cache_returns_singleton(_reset_singleton):
    cache1 = get_model_cache()
    cache2 = get_model_cache()
    assert cache1 is cache2


def test_get_model_cache_is_modelcache_instance(_reset_singleton):
    cache = get_model_cache()
    assert isinstance(cache, ModelCache)


# --- get_embedder ----------------------------------------------------------


def test_get_embedder_loads_once(_reset_singleton):
    cache = ModelCache()
    fake_model = MagicMock(name="SentenceTransformer")

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_st:
        result1 = cache.get_embedder("model-x", device="cpu")
        result2 = cache.get_embedder("model-x", device="cpu")

    # Same instance both times — loaded once
    assert result1 is fake_model
    assert result2 is fake_model
    assert mock_st.call_count == 1


def test_get_embedder_different_devices_separate_cache_keys(_reset_singleton):
    cache = ModelCache()

    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_st.return_value = MagicMock()
        cache.get_embedder("model-x", device="cpu")
        cache.get_embedder("model-x", device="cuda")

    assert mock_st.call_count == 2


def test_get_embedder_cache_key_format(_reset_singleton):
    """Cache key should be 'model:device' format."""
    cache = ModelCache()
    with patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()):
        cache.get_embedder("my-model", device="cpu")
    assert "my-model:cpu" in cache._embedders


# --- get_reranker ----------------------------------------------------------


def test_get_reranker_loads_once(_reset_singleton):
    cache = ModelCache()
    fake = MagicMock(name="CrossEncoder")

    with patch("sentence_transformers.CrossEncoder", return_value=fake) as mock_ce:
        cache.get_reranker("rerank-model", device="cpu")
        cache.get_reranker("rerank-model", device="cpu")

    assert mock_ce.call_count == 1


def test_get_reranker_separate_from_embedder(_reset_singleton):
    """Embedders and rerankers use separate caches."""
    cache = ModelCache()
    with patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()):
        with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()):
            cache.get_embedder("m", device="cpu")
            cache.get_reranker("m", device="cpu")

    assert "m:cpu" in cache._embedders
    assert "m:cpu" in cache._rerankers


# --- clear -----------------------------------------------------------------


def test_clear_empties_both_caches(_reset_singleton):
    cache = ModelCache()
    with patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()):
        with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()):
            cache.get_embedder("m1", device="cpu")
            cache.get_embedder("m2", device="cpu")
            cache.get_reranker("r1", device="cpu")

    assert len(cache._embedders) == 2
    assert len(cache._rerankers) == 1

    with patch("torch.cuda.is_available", return_value=False):
        cache.clear()

    assert cache._embedders == {}
    assert cache._rerankers == {}


def test_clear_calls_empty_cache_when_cuda_available(_reset_singleton):
    cache = ModelCache()
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty:
        cache.clear()
    mock_empty.assert_called_once()


def test_clear_on_empty_cache_no_log(_reset_singleton, caplog):
    import logging as _lg
    caplog.set_level(_lg.INFO, logger="mullet.knowledge.engine.model_cache")
    cache = ModelCache()
    with patch("torch.cuda.is_available", return_value=False):
        cache.clear()
    # No "cleared N" log when nothing was cached
    cleared_logs = [r for r in caplog.records if "cleared" in r.message]
    assert not cleared_logs


# --- inference_lock ----------------------------------------------------------
# Regression coverage for the hang where concurrent .encode() calls on the
# shared embedder (loading is locked, inference wasn't) would deadlock instead
# of raising, leaving reindex/sync jobs stuck at "100% processed" forever.


def test_inference_lock_returns_lock_instance(_reset_singleton):
    cache = ModelCache()
    lock = cache.inference_lock()
    assert isinstance(lock, type(threading.Lock()))


def test_inference_lock_same_instance_across_calls(_reset_singleton):
    """Every caller must serialize against the same lock, not a fresh one each time."""
    cache = ModelCache()
    assert cache.inference_lock() is cache.inference_lock()


def test_inference_lock_shared_across_model_keys(_reset_singleton):
    """One lock for the whole cache — this deployment has one GPU, so encode()
    calls for different cached models must still serialize against each other."""
    cache = ModelCache()
    with patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()):
        cache.get_embedder("model-a", device="cpu")
        cache.get_embedder("model-b", device="cpu")
    assert cache.inference_lock() is cache.inference_lock()


def test_inference_lock_serializes_concurrent_callers(_reset_singleton):
    """Two threads holding inference_lock() around 'encode' must never overlap."""
    cache = ModelCache()
    lock = cache.inference_lock()
    overlap_detected = threading.Event()
    currently_inside = threading.Event()

    def fake_encode():
        with lock:
            if currently_inside.is_set():
                overlap_detected.set()
            currently_inside.set()
            time.sleep(0.05)
            currently_inside.clear()

    threads = [threading.Thread(target=fake_encode) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not any(t.is_alive() for t in threads), "a thread deadlocked on inference_lock()"
    assert not overlap_detected.is_set(), "two threads held inference_lock() at the same time"
