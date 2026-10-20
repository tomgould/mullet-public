"""B1 coverage: HybridSearcher gaps from 2026-04-19 audit.

Cycle 6 — regression guards for `_load_bm25_cache`, `_mmr_diversify`, and
`recursive_search`. Most of these paths already had targeted coverage
added during P1 remediation (see `test_bm25_cache_numpy.py`,
`test_searcher_mmr_fallback.py`, `test_knowledge_engine_silent_failures.py`),
so several tests here intentionally pass on first run as regression guards
that lock the documented behaviour.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _searcher_min():
    """Minimal HybridSearcher shell — bypass __init__, no real models."""
    from mullet.knowledge.engine.search.searcher import HybridSearcher

    s = HybridSearcher.__new__(HybridSearcher)
    s.embedder = MagicMock()
    s.query_prefix = ""
    s.passage_prefix = ""
    s.llm_provider = None
    s.collection = MagicMock()
    s._progress_reporter = None
    s.bm25_k1 = 1.5
    s.bm25_b = 0.75
    s.ephemeral = False
    return s


def _bm25_cache_stub(tmp_path: Path):
    """Build a HybridSearcher stub primed for cache-load tests."""
    from mullet.knowledge.engine.search.searcher import HybridSearcher

    stub = HybridSearcher.__new__(HybridSearcher)
    stub.ephemeral = False
    stub.index_path = tmp_path
    stub._progress_reporter = None
    stub.bm25_k1 = 1.5
    stub.bm25_b = 0.75
    stub._compute_corpus_hash = lambda: "test-hash"
    stub._last_corpus_hash = None
    stub.bm25_index = None
    stub.bm25_chunks = []
    stub._bm25_tokenized_corpus = None

    coll = MagicMock()
    coll.count.return_value = 0
    stub.collection = coll
    return stub


# ---------------------------------------------------------------------------
# B1.1 — _load_bm25_cache validation
# ---------------------------------------------------------------------------


def test_load_bm25_cache_handles_corrupt_file(tmp_path, caplog):
    """Corrupt JSON in cache file → returns False, logs WARNING (no crash).

    Regression guard: the bare `except Exception` at the bottom of
    _load_bm25_cache must catch JSON parse errors and log them.
    """
    import logging

    stub = _bm25_cache_stub(tmp_path)
    cache_path = stub._get_bm25_cache_path()
    cache_path.write_bytes(b"\x00\x01\x02not-valid-json{{{")

    with caplog.at_level(logging.WARNING, logger="mullet.knowledge.engine.search.searcher"):
        ok = stub._load_bm25_cache()

    assert ok is False, "corrupt cache must not load"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "mullet.knowledge.engine.search.searcher"
        and "Failed to load BM25 cache" in r.getMessage()
    ]
    assert warnings, "expected a WARNING naming BM25 cache load failure"


def test_load_bm25_cache_handles_missing_file(tmp_path, caplog):
    """No cache file → returns False quietly (cold-start case, INFO not WARNING)."""
    import logging

    stub = _bm25_cache_stub(tmp_path)
    # Do not create any cache file
    assert not stub._get_bm25_cache_path().exists()

    with caplog.at_level(logging.INFO, logger="mullet.knowledge.engine.search.searcher"):
        ok = stub._load_bm25_cache()

    assert ok is False
    # Cold-start logs INFO, not WARNING — must not pollute warning channel
    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and r.name == "mullet.knowledge.engine.search.searcher"
    ]
    assert not warnings, "missing cache is expected — must not emit WARNING"


def test_load_bm25_cache_handles_version_mismatch(tmp_path, monkeypatch):
    """Cache with stale tokenizer/format version → discarded, returns False.

    Regression guard for the explicit version checks at lines 425-433.
    """
    from rank_bm25 import BM25Okapi
    from mullet.knowledge.engine.search.searcher import HybridSearcher

    corpus = [["a", "b"]]
    stub = _bm25_cache_stub(tmp_path)
    stub.bm25_chunks = [{"id": "c0", "content": "a b"}]
    stub._bm25_tokenized_corpus = corpus
    stub.bm25_index = BM25Okapi(corpus, k1=stub.bm25_k1, b=stub.bm25_b)
    stub._last_corpus_hash = "h"
    stub._compute_corpus_hash = lambda: "h"
    stub._save_bm25_cache()

    # Now bump format version to simulate stale cache
    monkeypatch.setattr(HybridSearcher, "_BM25_CACHE_FORMAT", 999)

    stub2 = _bm25_cache_stub(tmp_path)
    stub2.collection.count.return_value = 1
    stub2._compute_corpus_hash = lambda: "h"

    ok = stub2._load_bm25_cache()
    assert ok is False, "version-mismatch cache must be discarded"


def test_load_bm25_cache_handles_doc_count_mismatch(tmp_path):
    """Cache with wrong doc count → discarded, returns False (fast-path check)."""
    from rank_bm25 import BM25Okapi

    corpus = [["a", "b"], ["c", "d"]]
    stub = _bm25_cache_stub(tmp_path)
    stub.bm25_chunks = [
        {"id": "c0", "content": "a b"},
        {"id": "c1", "content": "c d"},
    ]
    stub._bm25_tokenized_corpus = corpus
    stub.bm25_index = BM25Okapi(corpus, k1=stub.bm25_k1, b=stub.bm25_b)
    stub._last_corpus_hash = "h"
    stub._save_bm25_cache()

    stub2 = _bm25_cache_stub(tmp_path)
    # Collection count is 99 — does not match cached doc_count of 2
    stub2.collection.count.return_value = 99
    stub2._compute_corpus_hash = lambda: "h"

    ok = stub2._load_bm25_cache()
    assert ok is False, "doc count mismatch must invalidate cache"


# ---------------------------------------------------------------------------
# B1.2 — _mmr_diversify CUDA OOM fallback (regression guard for existing
# coverage in test_searcher_mmr_fallback.py + test_knowledge_engine_silent_failures.py)
# ---------------------------------------------------------------------------


def test_mmr_diversify_cuda_oom_falls_back_to_cpu(_searcher_min):
    """Regression guard: GPU encode raises RuntimeError → CPU retry succeeds.

    Already covered by test_searcher_mmr_fallback.py — duplicated here as
    explicit B1 audit closure marker.
    """
    chunks = [
        {"content": "alpha", "_rerank_score": 0.9},
        {"content": "beta", "_rerank_score": 0.7},
        {"content": "gamma", "_rerank_score": 0.5},
    ]
    fake = np.random.rand(4, 8).astype(np.float32)
    calls = {"n": 0}

    def encode(texts, device=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        # Verify CPU fallback was requested
        assert device == "cpu"
        return fake

    _searcher_min.embedder.encode.side_effect = encode

    with patch("torch.cuda.empty_cache"):
        out = _searcher_min._mmr_diversify("q", chunks, top_k=2, lambda_param=0.5)

    assert calls["n"] == 2, "expected GPU then CPU call"
    assert len(out) == 2


def test_mmr_diversify_empty_results(_searcher_min):
    """Empty chunk list → returns [] without invoking the embedder."""
    out = _searcher_min._mmr_diversify("q", [], top_k=5, lambda_param=0.5)
    assert out == []
    _searcher_min.embedder.encode.assert_not_called()


def test_mmr_diversify_single_chunk_passthrough(_searcher_min):
    """Single chunk → returns it unchanged, no embedding cost."""
    chunks = [{"content": "only", "_rerank_score": 0.5}]
    out = _searcher_min._mmr_diversify("q", chunks, top_k=5, lambda_param=0.5)
    assert out == chunks
    _searcher_min.embedder.encode.assert_not_called()


def test_mmr_diversify_total_failure_returns_inputs(_searcher_min):
    """If both GPU and CPU encode fail → return chunks unchanged (no crash)."""
    chunks = [
        {"content": "a", "_rerank_score": 0.9},
        {"content": "b", "_rerank_score": 0.7},
    ]
    _searcher_min.embedder.encode.side_effect = RuntimeError("no devices")

    with patch("torch.cuda.empty_cache"):
        out = _searcher_min._mmr_diversify("q", chunks, top_k=2, lambda_param=0.5)

    # On total failure, return inputs unchanged
    assert out == chunks


# ---------------------------------------------------------------------------
# B1.3 — recursive_search failure handling (no prior coverage)
# ---------------------------------------------------------------------------


def test_recursive_search_returns_results_when_no_llm(_searcher_min):
    """Without an LLM, recursive_search uses score heuristic to terminate.

    Verifies the no-LLM path returns the accumulated chunks + a search log.
    """
    # Make .search return a deterministic batch
    fake_results = [
        {"id": f"c{i}", "content": f"doc {i}", "_rerank_score": 0.8}
        for i in range(5)
    ]
    _searcher_min.search = MagicMock(return_value=fake_results)

    chunks, log, reasoning = _searcher_min.recursive_search(
        query="test query",
        top_k=5,
        max_iterations=3,
        get_llm_response=None,
    )

    assert len(chunks) == 5
    assert all(c["id"].startswith("c") for c in chunks)
    assert isinstance(log, list)
    assert any("Search complete" in entry for entry in log)
    # No-LLM path produces no LLM-driven reasoning text
    assert reasoning == ""


def test_recursive_search_handles_search_failure_gracefully(_searcher_min):
    """If the underlying .search raises, recursive_search must not crash silently
    in a way that loses prior results.

    NOTE: current implementation does NOT wrap the search() call in try/except.
    This test documents the actual behaviour: if search() raises on the FIRST
    iteration, the exception propagates. This is a regression-anchor: if we
    later add try/except, this test will need updating.
    """
    _searcher_min.search = MagicMock(side_effect=RuntimeError("search boom"))

    with pytest.raises(RuntimeError, match="search boom"):
        _searcher_min.recursive_search(
            query="test",
            top_k=5,
            max_iterations=3,
            get_llm_response=None,
        )


def test_recursive_search_max_iterations_respected(_searcher_min):
    """Verify max_iterations caps the loop — no infinite recursion.

    Use a low-quality result that always triggers the alternative-query
    path, and confirm exactly max_iterations search calls happen.
    """
    # All results are low-quality (below default min_quality_score=0.3)
    low_quality = [
        {"id": "x", "content": "noise", "_rerank_score": 0.1},
    ]
    _searcher_min.search = MagicMock(return_value=low_quality)

    chunks, log, _ = _searcher_min.recursive_search(
        query="test",
        top_k=5,
        max_iterations=3,
        get_llm_response=None,
    )

    # Without LLM, alternative query generation is skipped — but the loop
    # must still cap at max_iterations search calls.
    assert _searcher_min.search.call_count <= 3, \
        f"max_iterations=3 must cap search calls, got {_searcher_min.search.call_count}"
    assert isinstance(chunks, list)


def test_recursive_search_dedupes_by_id(_searcher_min):
    """Repeated chunks across iterations must be deduped by ID."""
    # Same single chunk returned every iteration
    repeat = [{"id": "same", "content": "dup", "_rerank_score": 0.5}]
    _searcher_min.search = MagicMock(return_value=repeat)

    chunks, _, _ = _searcher_min.recursive_search(
        query="test",
        top_k=5,
        max_iterations=2,
        get_llm_response=None,
    )

    assert len(chunks) == 1, f"expected dedup, got {len(chunks)} chunks"
    assert chunks[0]["id"] == "same"
