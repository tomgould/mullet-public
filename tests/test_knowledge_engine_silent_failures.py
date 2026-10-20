"""Regression guards: knowledge-engine silent-failure log assertions (B3 + B4).

Cycle 5 audit (TODO P1 — SILENT FAILURES) called out:
- ``pipeline.py::_flush_buffer`` swallowed ChromaDB upsert failures silently.
- ``searcher.py`` had 12+ bare/generic except clauses across BM25, vector,
  HyDE, MMR, reranker — none logged.

Both files were remediated in earlier P1 commits (notably 759fad5 for the
flush rollback). These tests are intentionally lightweight regression
guards: each one provokes the documented failure mode and asserts that a
WARNING (or ERROR for the upsert raise) is emitted on the right logger
channel. They protect against any future refactor that replaces a
``logger.warning(...)`` with a silent ``pass``.

The pipeline tests in ``tests/test_knowledge_pipeline.py`` already cover
the rollback + re-raise behaviour. These tests are strictly about the
*logging* contract.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures: minimal HybridSearcher / IngestionPipeline shells
# ---------------------------------------------------------------------------


@pytest.fixture
def _searcher_min():
    """Build a minimal HybridSearcher shell without loading real models."""
    from mullet.knowledge.engine.search.searcher import HybridSearcher

    s = HybridSearcher.__new__(HybridSearcher)
    s.embedder = MagicMock()
    s.query_prefix = ""
    s.passage_prefix = ""
    s.llm_provider = None
    s.collection = MagicMock()
    return s


def _make_pipeline(searcher, ingestor=None):
    """Build an IngestionPipeline with minimal mocks."""
    from mullet.knowledge.engine.pipeline import IngestionPipeline

    if ingestor is None:
        ingestor = MagicMock()
        ingestor.create_embeddings.return_value = [[0.0] * 4, [0.0] * 4]
        ingestor.embedder = MagicMock()
    return IngestionPipeline(
        ingestor=ingestor,
        searcher=searcher,
        parse_workers=1,
        embed_batch_size=4,
        contextualize=False,
        resource_manager=None,
    )


def _mock_searcher_for_pipeline():
    s = MagicMock()
    s._sanitize_metadata = lambda m: m
    s.collection = MagicMock()
    return s


# ---------------------------------------------------------------------------
# B3 — pipeline.py::_flush_buffer
# ---------------------------------------------------------------------------


def test_pipeline_flush_buffer_logs_on_upsert_failure(caplog):
    """_flush_buffer must emit an ERROR record naming the failure before re-raising.

    Regression guard for the silent ChromaDB upsert failure mode that
    P1-6 (commit 759fad5) fixed.
    """
    searcher = _mock_searcher_for_pipeline()
    searcher.collection.upsert.side_effect = RuntimeError("chroma boom")

    pipeline = _make_pipeline(searcher)
    pipeline._chunk_buffer = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
        {"id": "b", "content": "beta", "metadata": {"source_path": "/b"}},
    ]
    pipeline._buffer_source_paths = {"/a", "/b"}

    with caplog.at_level(logging.ERROR, logger="mullet.knowledge.engine.pipeline"):
        with pytest.raises(RuntimeError, match="chroma boom"):
            pipeline._flush_buffer()

    error_records = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
        and r.name == "mullet.knowledge.engine.pipeline"
    ]
    assert error_records, "expected an ERROR record from pipeline._flush_buffer"
    msg = error_records[0].getMessage()
    assert "_flush_buffer" in msg, "log must name the failing function"
    assert "chroma boom" in msg, "log must include the exception text"


def test_pipeline_flush_buffer_does_not_save_manifest_on_failure(tmp_path):
    """Regression-safety: upsert failure raises out of run_documents so
    callers' manifest-save step is bypassed (P1-6).
    """
    searcher = _mock_searcher_for_pipeline()
    searcher.collection.upsert.side_effect = RuntimeError("chroma boom")
    searcher.collection.get.return_value = {"metadatas": []}

    ingestor = MagicMock()
    ingestor.create_embeddings.return_value = [[0.0] * 4]
    ingestor.embedder = MagicMock()
    ingestor.parse_document.return_value = ["elem"]
    ingestor.chunk_elements.return_value = [
        {"id": "a", "content": "alpha", "metadata": {}},
    ]

    pipeline = _make_pipeline(searcher, ingestor=ingestor)

    docs = tmp_path / "documents"
    docs.mkdir()
    f = docs / "test.txt"
    f.write_text("hello")

    with pytest.raises(RuntimeError, match="chroma boom"):
        pipeline.run_documents([f], base_directory=docs)

    # Counters did NOT advance — caller will see the exception, not a silent
    # success, so no manifest will be saved on the happy path.
    assert pipeline._total_chunks_indexed == 0
    assert pipeline._flush_count == 0


# ---------------------------------------------------------------------------
# B4 — searcher.py: BM25 / vector / HyDE / MMR / reranker / graph catch sites
# ---------------------------------------------------------------------------


def test_searcher_bm25_failure_logged_in_fast_search(_searcher_min, caplog):
    """When BM25 raises inside fast_search's parallel block, a WARNING must
    be emitted on the searcher logger and the search must fall back to
    vector results only.

    fast_search exercises the same except site as search() (lines 1308-1311
    vs 1571-1574) but does not require the post-rerank quality machinery,
    keeping the test focused on the logging contract.
    """
    s = _searcher_min
    s.merge_strategy = "weighted"
    s.bm25_weight = 0.5
    s.vector_weight = 0.5
    s._sanitize_query = lambda q: q
    s._bm25_search = MagicMock(side_effect=RuntimeError("bm25 boom"))
    s._vector_search = MagicMock(return_value=[
        ({"id": "v1", "content": "vec", "metadata": {}}, 0.5),
    ])
    s._merge_results = MagicMock(return_value=[
        {"id": "v1", "content": "vec", "metadata": {}, "_combined_score": 0.5},
    ])

    with caplog.at_level(logging.WARNING, logger="mullet.knowledge.engine.search.searcher"):
        results = s.fast_search("hello world", top_k=5)

    assert results, "vector-only fallback must still return results"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "mullet.knowledge.engine.search.searcher"
        and "BM25" in r.getMessage()
    ]
    assert warnings, "expected a WARNING naming BM25 failure"
    assert "bm25 boom" in warnings[0].getMessage()


def test_searcher_vector_failure_logged_in_fast_search(_searcher_min, caplog):
    """Vector search failure must log WARNING and fall back to BM25-only."""
    s = _searcher_min
    s.merge_strategy = "weighted"
    s.bm25_weight = 0.5
    s.vector_weight = 0.5
    s._sanitize_query = lambda q: q
    s._bm25_search = MagicMock(return_value=[
        ({"id": "b1", "content": "kw", "metadata": {}}, 0.6),
    ])
    s._vector_search = MagicMock(side_effect=RuntimeError("vec boom"))
    s._merge_results = MagicMock(return_value=[
        {"id": "b1", "content": "kw", "metadata": {}, "_combined_score": 0.6},
    ])

    with caplog.at_level(logging.WARNING, logger="mullet.knowledge.engine.search.searcher"):
        results = s.fast_search("hello world", top_k=5)

    assert results, "BM25-only fallback must still return results"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "mullet.knowledge.engine.search.searcher"
        and "Vector" in r.getMessage()
    ]
    assert warnings, "expected a WARNING naming Vector failure"
    assert "vec boom" in warnings[0].getMessage()


def test_searcher_hyde_generation_failure_logged_falls_back(_searcher_min, caplog):
    """HyDE LLM call raising must log WARNING and return None (caller
    falls back to the original query)."""
    s = _searcher_min
    s.llm_provider = MagicMock()
    s.llm_provider.generate.side_effect = RuntimeError("llm boom")

    with caplog.at_level(logging.WARNING, logger="mullet.knowledge.engine.search.searcher"):
        out = s._generate_hypothetical_document("what is x?")

    assert out is None, "HyDE must return None on LLM failure (caller falls back)"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "mullet.knowledge.engine.search.searcher"
        and "HyDE" in r.getMessage()
    ]
    assert warnings, "expected a WARNING naming HyDE failure"
    assert "llm boom" in warnings[0].getMessage()


def test_searcher_hyde_vector_merge_failure_logged(_searcher_min, caplog):
    """HyDE second vector query failing must log WARNING and use raw query
    results only."""
    s = _searcher_min

    # First call (primary query) succeeds, second (HyDE embedding) raises.
    call_count = {"n": 0}

    def query_side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "ids": [["a"]],
                "documents": [["doc-a"]],
                "metadatas": [[{}]],
                "distances": [[0.1]],
            }
        raise RuntimeError("hyde vec boom")

    s.collection.query.side_effect = query_side_effect
    s.embedder.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])

    with caplog.at_level(logging.WARNING, logger="mullet.knowledge.engine.search.searcher"):
        results = s._vector_search("q", top_n=10, extra_embedding=[0.4, 0.5, 0.6])

    assert results, "primary results must survive HyDE secondary failure"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "mullet.knowledge.engine.search.searcher"
        and "HyDE" in r.getMessage()
    ]
    assert warnings, "expected a WARNING naming HyDE vector failure"


def test_searcher_mmr_diversify_failure_logged(_searcher_min, caplog):
    """MMR embedder failing on both GPU and CPU paths must log WARNING and
    return the input chunks unchanged (no silent crash)."""
    s = _searcher_min
    chunks = [
        {"id": "a", "content": "alpha", "_rerank_score": 0.9},
        {"id": "b", "content": "beta", "_rerank_score": 0.7},
    ]
    s.embedder.encode.side_effect = RuntimeError("mmr boom")

    with patch("torch.cuda.empty_cache"):
        with caplog.at_level(logging.WARNING, logger="mullet.knowledge.engine.search.searcher"):
            out = s._mmr_diversify("q", chunks, top_k=2, lambda_param=0.5)

    assert out == chunks, "MMR must return inputs unchanged on total failure"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "mullet.knowledge.engine.search.searcher"
    ]
    assert warnings, "expected at least one WARNING from MMR fallback path"
    # At least one of the warnings must reference MMR
    assert any("MMR" in r.getMessage() for r in warnings), \
        "expected a WARNING naming MMR"


def test_searcher_reranker_api_failure_logged_falls_back(caplog):
    """Reranker API failure must log WARNING and transparently switch to
    the local reranker for subsequent calls (regression guard for the
    silent-fall-through bug class)."""
    from mullet.knowledge.engine.reranker import FallbackReranker

    api = MagicMock()
    api.predict.side_effect = RuntimeError("rerank api boom")

    fb = FallbackReranker.__new__(FallbackReranker)
    fb._api = api
    fb._api_failed = False
    fb._local = MagicMock()
    fb._local.predict.return_value = [0.5, 0.4]
    fb._local_model = "stub"
    fb._device = "cpu"

    with caplog.at_level(logging.WARNING, logger="mullet.knowledge.engine.reranker"):
        out = fb.predict([("q", "doc1"), ("q", "doc2")])

    assert out == [0.5, 0.4], "must transparently return local reranker scores"
    assert fb._api_failed is True, "must mark API as failed for next call"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "mullet.knowledge.engine.reranker"
    ]
    assert warnings, "expected a WARNING from reranker API fallback"
    assert any("API reranker" in r.getMessage() for r in warnings), \
        "expected a WARNING naming API reranker failure"
    assert any("rerank api boom" in r.getMessage() for r in warnings), \
        "log must include exception text"


def test_searcher_graph_load_failure_logged(_searcher_min, caplog):
    """load_graph() must log WARNING and return None when the graph module
    or file cannot be loaded — never raise out of search."""
    s = _searcher_min
    s.index_path = "/nonexistent/path/that/cannot/exist"
    s._knowledge_graph = None

    with patch(
        "mullet.knowledge.graph.store.KnowledgeGraph",
        side_effect=RuntimeError("graph boom"),
    ):
        with caplog.at_level(logging.WARNING, logger="mullet.knowledge.engine.search.searcher"):
            graph = s.load_graph()

    assert graph is None, "graph load failure must yield None, never raise"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "mullet.knowledge.engine.search.searcher"
        and "graph" in r.getMessage().lower()
    ]
    assert warnings, "expected a WARNING naming graph load failure"
    assert "graph boom" in warnings[0].getMessage()
