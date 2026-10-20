"""Tests for HybridSearcher._mmr_diversify CUDA OOM fallback path.

Per docs/TODO.md P1 audit: CUDA OOM fallback to CPU was untested.
These tests mock the embedder to raise torch-style OOM and verify the
graceful fallback (CPU retry, then skip MMR entirely if CPU also fails).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def _searcher_min():
    """Build a minimal HybridSearcher shell without loading real models."""
    from mullet.knowledge.engine.search.searcher import HybridSearcher

    # Bypass __init__ — we only need attributes the method uses
    s = HybridSearcher.__new__(HybridSearcher)
    s.embedder = MagicMock()
    s.query_prefix = ""
    return s


def test_mmr_single_chunk_passthrough(_searcher_min):
    """len(chunks) <= 1 should passthrough without calling embedder."""
    chunks = [{"content": "only one", "_rerank_score": 0.5}]
    out = _searcher_min._mmr_diversify("q", chunks, top_k=5, lambda_param=0.5)
    assert out == chunks
    _searcher_min.embedder.encode.assert_not_called()


def test_mmr_empty_chunks_passthrough(_searcher_min):
    """Empty chunks list should return empty list."""
    out = _searcher_min._mmr_diversify("q", [], top_k=5, lambda_param=0.5)
    assert out == []


def test_mmr_falls_back_to_cpu_on_cuda_oom(_searcher_min):
    """RuntimeError (CUDA OOM proxy) on first encode triggers CPU retry."""

    chunks = [
        {"content": "a", "_rerank_score": 0.5},
        {"content": "b", "_rerank_score": 0.4},
    ]

    # First call raises OOM; second call (device='cpu') succeeds
    fake_embeddings = np.random.rand(3, 8).astype(np.float32)
    call_count = {"n": 0}

    def encode_side_effect(texts, device=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return fake_embeddings

    _searcher_min.embedder.encode.side_effect = encode_side_effect

    with patch("torch.cuda.empty_cache"):
        out = _searcher_min._mmr_diversify("q", chunks, top_k=2, lambda_param=0.5)

    assert call_count["n"] == 2, "expected OOM, then CPU retry"
    assert len(out) == 2


def test_mmr_cpu_fallback_also_fails_returns_original_chunks(_searcher_min):
    """If both GPU and CPU fail, skip MMR and return the input chunks unchanged."""
    chunks = [
        {"content": "a", "_rerank_score": 0.9},
        {"content": "b", "_rerank_score": 0.7},
        {"content": "c", "_rerank_score": 0.5},
    ]

    _searcher_min.embedder.encode.side_effect = RuntimeError("no devices available")

    with patch("torch.cuda.empty_cache"):
        out = _searcher_min._mmr_diversify("q", chunks, top_k=3, lambda_param=0.5)

    # On total failure, method returns chunks as-is
    assert out == chunks
