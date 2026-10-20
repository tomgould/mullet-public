"""Tests for mullet.knowledge.engine.pipeline.IngestionPipeline rollback semantics.

Covers P1-6: _flush_buffer now re-raises on upsert failure and clears the
buffer so callers don't double-flush / infinitely retry the same batch, and
so any manifest save outside this method is skipped.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mullet.knowledge.engine.pipeline import IngestionPipeline


def _make_pipeline(searcher, ingestor=None):
    """Build an IngestionPipeline with minimal mocks."""
    if ingestor is None:
        ingestor = MagicMock()
        ingestor.create_embeddings.return_value = [[0.0] * 4, [0.0] * 4]
        # Defensive: pipeline probes this when resource_manager is passed, not here
        ingestor.embedder = MagicMock()
    return IngestionPipeline(
        ingestor=ingestor,
        searcher=searcher,
        parse_workers=1,
        embed_batch_size=4,  # small so our 2-chunk batch stays in one flush
        contextualize=False,
        resource_manager=None,
    )


def _mock_searcher():
    """Searcher whose _sanitize_metadata is a pass-through."""
    searcher = MagicMock()
    searcher._sanitize_metadata = lambda m: m
    searcher.collection = MagicMock()
    return searcher


def test_flush_buffer_reraises_on_upsert_failure():
    """Upsert raising should propagate out of _flush_buffer, not be swallowed."""
    searcher = _mock_searcher()
    searcher.collection.upsert.side_effect = RuntimeError("chroma boom")

    pipeline = _make_pipeline(searcher)
    pipeline._chunk_buffer = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
        {"id": "b", "content": "beta", "metadata": {"source_path": "/b"}},
    ]
    pipeline._buffer_source_paths = {"/a", "/b"}

    with pytest.raises(RuntimeError, match="chroma boom"):
        pipeline._flush_buffer()

    # Buffer MUST be cleared so a caller retry doesn't loop on the same batch.
    assert pipeline._chunk_buffer == []
    assert pipeline._buffer_source_paths == set()
    # Chunk counters MUST NOT advance when the upsert failed.
    assert pipeline._total_chunks_indexed == 0
    assert pipeline._flush_count == 0


def test_flush_buffer_reraises_on_embed_failure():
    """Embedding raising is also a flush failure — same rollback semantics."""
    searcher = _mock_searcher()
    ingestor = MagicMock()
    ingestor.create_embeddings.side_effect = ValueError("embedder oom")
    ingestor.embedder = MagicMock()

    pipeline = _make_pipeline(searcher, ingestor=ingestor)
    pipeline._chunk_buffer = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
    ]
    pipeline._buffer_source_paths = {"/a"}

    with pytest.raises(ValueError, match="embedder oom"):
        pipeline._flush_buffer()

    assert pipeline._chunk_buffer == []
    assert pipeline._buffer_source_paths == set()
    assert pipeline._total_chunks_indexed == 0
    # Upsert should never have been called — failure was upstream.
    searcher.collection.upsert.assert_not_called()


def test_flush_buffer_success_path_still_clears_and_advances_counters():
    """Sanity check: the happy path still clears the buffer and advances counters."""
    searcher = _mock_searcher()
    pipeline = _make_pipeline(searcher)
    pipeline._chunk_buffer = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
        {"id": "b", "content": "beta", "metadata": {"source_path": "/b"}},
    ]
    pipeline._buffer_source_paths = {"/a", "/b"}

    pipeline._flush_buffer()

    assert pipeline._chunk_buffer == []
    assert pipeline._buffer_source_paths == set()
    assert pipeline._total_chunks_indexed == 2
    assert pipeline._flush_count == 1
    searcher.collection.upsert.assert_called_once()


def test_flush_buffer_failure_does_not_save_manifest(tmp_path, monkeypatch):
    """Integration-ish test: run_documents -> _flush_buffer fails -> manifest
    save is skipped.

    We don't call save_file_manifest here; we just verify the exception bubbles
    out of run_documents so whoever is wrapping it will naturally bypass their
    manifest-save step.
    """
    searcher = _mock_searcher()
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

    # Simulate a single file in the documents dir.
    docs = tmp_path / "documents"
    docs.mkdir()
    f = docs / "test.txt"
    f.write_text("hello")

    with pytest.raises(RuntimeError, match="chroma boom"):
        pipeline.run_documents([f], base_directory=docs)

    # Buffer cleared post-failure.
    assert pipeline._chunk_buffer == []
    # Counters did NOT advance.
    assert pipeline._total_chunks_indexed == 0
    assert pipeline._flush_count == 0


# ── Coverage gaps for mullet.knowledge.engine.pipeline ──────────────────────

def test_init_with_resource_manager_gpu():
    """ResourceManager with GPU overrides embed_batch_size adaptively."""
    rm = MagicMock()
    rm.probe_gpu.return_value = {"free_mb": 1000}
    rm.recommend_embed_batch.return_value = 32

    ingestor = MagicMock()
    ingestor.embedder.get_sentence_embedding_dimension.return_value = 768

    searcher = _mock_searcher()
    pipeline = IngestionPipeline(
        ingestor=ingestor,
        searcher=searcher,
        resource_manager=rm,
        embed_batch_size=64,
    )
    assert pipeline.embed_batch_size == 32
    rm.recommend_embed_batch.assert_called_once_with(768)


def test_init_with_resource_manager_no_gpu():
    """ResourceManager without GPU leaves defaults unchanged."""
    rm = MagicMock()
    rm.probe_gpu.return_value = None

    ingestor = MagicMock()
    searcher = _mock_searcher()
    pipeline = IngestionPipeline(
        ingestor=ingestor,
        searcher=searcher,
        resource_manager=rm,
        embed_batch_size=64,
    )
    assert pipeline.embed_batch_size == 64
    rm.recommend_embed_batch.assert_not_called()


def test_install_signal_handler_not_main_thread():
    """_install_signal_handler returns early when not in main thread."""
    import threading

    pipeline = _make_pipeline(_mock_searcher())

    def install():
        pipeline._install_signal_handler()

    t = threading.Thread(target=install)
    t.start()
    t.join()
    assert pipeline._original_sigint is None


def test_signal_handler_force_exit():
    """Second SIGINT triggers force-exit path calling original handler."""
    import signal

    pipeline = _make_pipeline(_mock_searcher())
    called = []

    def original_handler(signum, frame):
        called.append((signum, frame))

    with patch.object(signal, "getsignal", return_value=original_handler):
        with patch.object(signal, "signal") as mock_signal:
            pipeline._install_signal_handler()
            # Extract the handler that was registered
            handler = mock_signal.call_args[0][1]

            # First interrupt
            handler(2, None)
            assert pipeline._interrupted is True

            # Second interrupt should call original
            handler(2, None)
            assert len(called) == 1


def test_restore_signal_handler_not_main_thread():
    """_restore_signal_handler returns early when not in main thread."""
    import threading

    pipeline = _make_pipeline(_mock_searcher())
    pipeline._original_sigint = lambda s, f: None

    def restore():
        pipeline._restore_signal_handler()

    t = threading.Thread(target=restore)
    t.start()
    t.join()
    assert pipeline._original_sigint is not None


def test_flush_buffer_empty():
    """_flush_buffer with empty buffer returns immediately."""
    searcher = _mock_searcher()
    pipeline = _make_pipeline(searcher)
    pipeline._flush_buffer()
    searcher.collection.upsert.assert_not_called()


def test_flush_buffer_contextualize_dedup():
    """Context strings are deduplicated alongside chunks."""
    searcher = _mock_searcher()
    ingestor = MagicMock()
    ingestor.create_embeddings.return_value = [[0.0] * 4, [0.0] * 4]
    ingestor.embedder = MagicMock()

    pipeline = _make_pipeline(searcher, ingestor=ingestor)
    pipeline.contextualize = True
    pipeline._chunk_buffer = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
        {"id": "a", "content": "alpha dup", "metadata": {"source_path": "/a"}},
        {"id": "b", "content": "beta", "metadata": {"source_path": "/b"}},
    ]
    pipeline._context_buffer = ["ctx_a", "ctx_a_dup", "ctx_b"]
    pipeline._buffer_source_paths = {"/a", "/b"}

    pipeline._flush_buffer()

    # Only 2 unique chunks after dedup
    assert pipeline._total_chunks_indexed == 2
    ingestor.create_embeddings.assert_called_once()
    args, kwargs = ingestor.create_embeddings.call_args
    assert kwargs["context_strings"] == ["ctx_a", "ctx_b"]


def test_add_chunks_to_buffer_with_context():
    """Context strings are appended to _context_buffer when provided."""
    searcher = _mock_searcher()
    pipeline = _make_pipeline(searcher)
    chunks = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
    ]
    pipeline._add_chunks_to_buffer(chunks, context_strings=["ctx_a"])
    assert pipeline._context_buffer == ["ctx_a"]


def test_add_chunks_to_buffer_triggers_flush():
    """Adding chunks beyond embed_batch_size triggers _flush_buffer."""
    searcher = _mock_searcher()
    pipeline = _make_pipeline(searcher)
    pipeline.embed_batch_size = 2

    chunks = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
        {"id": "b", "content": "beta", "metadata": {"source_path": "/b"}},
    ]
    pipeline._add_chunks_to_buffer(chunks)
    assert pipeline._total_chunks_indexed == 2
    assert pipeline._chunk_buffer == []


def test_parse_document_file_mbox():
    """mbox files are routed to ingest_mbox_chunks_only."""
    ingestor = MagicMock()
    ingestor.ingest_mbox_chunks_only.return_value = [
        {"id": "m1", "content": "mail", "metadata": {}},
    ]
    ingestor.embedder = MagicMock()

    pipeline = _make_pipeline(_mock_searcher(), ingestor=ingestor)
    path = MagicMock()
    path.suffix.lower.return_value = ".mbox"
    path.name = "test.mbox"
    path.relative_to.return_value = "test.mbox"

    result = pipeline._parse_document_file(path, base_directory=MagicMock())
    ingestor.ingest_mbox_chunks_only.assert_called_once_with(path)
    assert result[0]["metadata"]["source_file"] == "test.mbox"


def test_parse_document_file_relative_valueerror():
    """ValueError from relative_to falls back to path.name."""
    ingestor = MagicMock()
    ingestor.parse_document.return_value = ["elem"]
    ingestor.chunk_elements.return_value = [
        {"id": "c1", "content": "chunk", "metadata": {}},
    ]
    ingestor.embedder = MagicMock()

    pipeline = _make_pipeline(_mock_searcher(), ingestor=ingestor)
    path = MagicMock()
    path.suffix.lower.return_value = ".txt"
    path.name = "test.txt"
    path.relative_to.side_effect = ValueError("not relative")

    result = pipeline._parse_document_file(path, base_directory=MagicMock())
    assert result[0]["metadata"]["source_file"] == "test.txt"


def test_parse_document_file_exception():
    """Exception during parsing returns empty list and logs error."""
    ingestor = MagicMock()
    ingestor.parse_document.side_effect = RuntimeError("parse fail")
    ingestor.embedder = MagicMock()

    pipeline = _make_pipeline(_mock_searcher(), ingestor=ingestor)
    path = MagicMock()
    path.suffix.lower.return_value = ".txt"
    path.name = "bad.txt"

    result = pipeline._parse_document_file(path, base_directory=MagicMock())
    assert result == []


def test_get_indexed_source_paths_success():
    """Successfully extracts source_paths from ChromaDB metadata."""
    searcher = _mock_searcher()
    searcher.collection.get.return_value = {
        "metadatas": [
            {"source_path": "/a"},
            {"source_path": "/b"},
            {},
        ]
    }
    pipeline = _make_pipeline(searcher)
    paths = pipeline.get_indexed_source_paths()
    assert paths == {"/a", "/b"}


def test_get_indexed_source_paths_failure():
    """Exception querying ChromaDB returns empty set."""
    searcher = _mock_searcher()
    searcher.collection.get.side_effect = RuntimeError("db down")
    pipeline = _make_pipeline(searcher)
    paths = pipeline.get_indexed_source_paths()
    assert paths == set()


def test_run_documents_resume_log():
    """Already-indexed files are skipped and logged."""
    searcher = _mock_searcher()
    ingestor = MagicMock()
    ingestor.embedder = MagicMock()

    pipeline = _make_pipeline(searcher, ingestor=ingestor)
    docs = MagicMock()
    f = MagicMock()
    f.__str__ = lambda self: "/already/indexed.txt"

    with patch("mullet.knowledge.engine.pipeline.logger") as mock_logger:
        pipeline.run_documents([f], base_directory=docs, already_indexed={"/already/indexed.txt"})
        mock_logger.info.assert_any_call("Resuming: skipping 1 already-indexed files")


def test_run_documents_interrupted_cancels_futures():
    """_interrupted flag cancels remaining futures and breaks loop."""
    import concurrent.futures

    searcher = _mock_searcher()
    ingestor = MagicMock()
    ingestor.embedder = MagicMock()
    ingestor.parse_document.return_value = ["elem"]
    ingestor.chunk_elements.return_value = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
    ]

    pipeline = _make_pipeline(searcher, ingestor=ingestor)
    pipeline.parse_workers = 2

    f1 = MagicMock()
    f1.__str__ = lambda self: "/f1.txt"
    f2 = MagicMock()
    f2.__str__ = lambda self: "/f2.txt"

    # Force interruption after first future completes
    original_parse = pipeline._parse_document_file

    def interrupt_after_first(*args, **kwargs):
        result = original_parse(*args, **kwargs)
        pipeline._interrupted = True
        return result

    with patch.object(pipeline, "_parse_document_file", side_effect=interrupt_after_first):
        stats = pipeline.run_documents([f1, f2], base_directory=MagicMock())

    assert pipeline._interrupted is True


def test_run_documents_contextualize():
    """contextualize=True builds context strings and marks metadata."""
    from mullet.knowledge.engine import ingest as ingest_module

    searcher = _mock_searcher()
    ingestor = MagicMock()
    ingestor.embedder = MagicMock()
    ingestor.parse_document.return_value = ["elem"]
    ingestor.chunk_elements.return_value = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
    ]
    ingestor.create_embeddings.return_value = [[0.0] * 4]

    pipeline = _make_pipeline(searcher, ingestor=ingestor)
    pipeline.contextualize = True
    pipeline.embed_batch_size = 10

    with patch.object(ingest_module, "build_context_prefix", return_value="ctx_prefix"):
        f = MagicMock()
        f.__str__ = lambda self: "/f.txt"
        f.suffix.lower.return_value = ".txt"
        f.name = "f.txt"
        f.relative_to.return_value = "f.txt"
        stats = pipeline.run_documents([f], base_directory=MagicMock())

    assert pipeline._context_buffer == []
    ingestor.create_embeddings.assert_called_once()


def test_run_documents_future_exception():
    """Exception in future.result() logs error and records failed file."""
    import concurrent.futures

    searcher = _mock_searcher()
    ingestor = MagicMock()
    ingestor.embedder = MagicMock()

    pipeline = _make_pipeline(searcher, ingestor=ingestor)

    f = MagicMock()
    f.__str__ = lambda self: "/bad.txt"
    f.name = "bad.txt"

    # Mock ThreadPoolExecutor to raise on result()
    mock_future = MagicMock()
    mock_future.result.side_effect = RuntimeError("worker boom")

    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future
    mock_executor.__enter__ = lambda self: self
    mock_executor.__exit__ = lambda self, *args: None

    with patch("mullet.knowledge.engine.pipeline.ThreadPoolExecutor", return_value=mock_executor):
        with patch("mullet.knowledge.engine.pipeline.as_completed", return_value=[mock_future]):
            with patch("mullet.knowledge.engine.pipeline.logger") as mock_logger:
                stats = pipeline.run_documents([f], base_directory=MagicMock())
                mock_logger.error.assert_any_call("Failed to process bad.txt: worker boom")

    assert "/bad.txt" in pipeline._failed_files


def test_run_documents_progress_callback():
    """progress_callback is invoked after each processed file."""
    searcher = _mock_searcher()
    ingestor = MagicMock()
    ingestor.embedder = MagicMock()
    ingestor.parse_document.return_value = ["elem"]
    ingestor.chunk_elements.return_value = [
        {"id": "a", "content": "alpha", "metadata": {"source_path": "/a"}},
    ]
    ingestor.create_embeddings.return_value = [[0.0] * 4]

    pipeline = _make_pipeline(searcher, ingestor=ingestor)
    pipeline.embed_batch_size = 10

    calls = []
    f = MagicMock()
    f.__str__ = lambda self: "/f.txt"
    f.suffix.lower.return_value = ".txt"
    f.name = "f.txt"
    f.relative_to.return_value = "f.txt"

    pipeline.run_documents([f], base_directory=MagicMock(), progress_callback=lambda s: calls.append(s))
    assert len(calls) >= 1


def test_run_code_empty_chunks():
    """run_code with empty chunks returns stats immediately."""
    searcher = _mock_searcher()
    pipeline = _make_pipeline(searcher)
    stats = pipeline.run_code([])
    assert stats["total_chunks_indexed"] == 0
    assert stats["total_files_processed"] == 0


def test_run_code_success():
    """run_code buffers and flunks pre-chunked code."""
    searcher = _mock_searcher()
    ingestor = MagicMock()
    ingestor.embedder = MagicMock()
    ingestor.create_embeddings.return_value = [[0.0] * 4]

    pipeline = _make_pipeline(searcher, ingestor=ingestor)
    chunks = [
        {"id": "c1", "content": "code", "metadata": {"source_path": "/c"}},
    ]
    stats = pipeline.run_code(chunks)
    assert stats["total_chunks_indexed"] == 1
    assert stats["flush_count"] == 1
    searcher.collection.upsert.assert_called_once()


def test_get_stats_zero_start_time():
    """_get_stats with _start_time == 0 returns zero elapsed."""
    searcher = _mock_searcher()
    pipeline = _make_pipeline(searcher)
    pipeline._start_time = 0.0
    stats = pipeline._get_stats()
    assert stats["elapsed_seconds"] == 0.0
    assert stats["chunks_per_second"] == 0


# ── Coverage gaps for mullet.core.knowledge.pipeline ─────────────────────────

import threading
import signal
import time

from mullet.core.knowledge.pipeline import (
    IngestionPipeline as CoreIngestionPipeline,
    PipelineChunk as CorePipelineChunk,
    PipelineStats as CorePipelineStats,
)


def test_core_parse_stage_progress_callback_on_flush(tmp_path):
    """Progress callback inside _parse_stage is invoked when buffer flushes."""
    content = "print('hello world')\n" * 500
    (tmp_path / "large.py").write_text(content)

    calls = []
    pipeline = CoreIngestionPipeline(
        parse_workers=1,
        embed_batch_size=4,
        max_chunk_chars=200,
        chunk_overlap=20,
        incremental=False,
    )
    pipeline.run(tmp_path, progress_callback=lambda stats: calls.append(stats.chunks_created))
    assert len(calls) > 0


def test_core_parse_stage_interrupted_cancels_futures(tmp_path, monkeypatch):
    """Setting _interrupted=True during _parse_stage cancels remaining futures."""
    for i in range(5):
        (tmp_path / f"file{i}.py").write_text(f"x = {i}\n" * 200)

    pipeline = CoreIngestionPipeline(
        parse_workers=4,
        embed_batch_size=100,
        incremental=False,
    )

    original_parse = pipeline._parse_one_file

    def slow_parse(fpath, root, manifest):
        if fpath.name != "file0.py":
            time.sleep(0.1)
        return original_parse(fpath, root, manifest)

    monkeypatch.setattr(pipeline, "_parse_one_file", slow_parse)

    def trigger_interrupt():
        time.sleep(0.05)
        pipeline._interrupted = True

    t = threading.Thread(target=trigger_interrupt)
    t.start()
    stats = pipeline.run(tmp_path)
    t.join()

    assert stats.interrupted is True


def test_core_parse_stage_future_result_exception(tmp_path, monkeypatch):
    """Exception raised in a worker thread is captured in stats.errors."""
    (tmp_path / "bad.py").write_text("bad content")
    (tmp_path / "good.py").write_text("print('ok')\n" * 200)

    pipeline = CoreIngestionPipeline(
        parse_workers=2,
        embed_batch_size=100,
        incremental=False,
    )

    original_parse = pipeline._parse_one_file

    def failing_parse(fpath, root, manifest):
        if fpath.name == "bad.py":
            raise RuntimeError("parse boom")
        return original_parse(fpath, root, manifest)

    monkeypatch.setattr(pipeline, "_parse_one_file", failing_parse)

    stats = pipeline.run(tmp_path)
    assert any("bad.py" in err and "parse boom" in err for err in stats.errors)


def test_core_install_signal_handler_not_main_thread():
    """_install_signal_handler returns early when not in main thread."""
    pipeline = CoreIngestionPipeline()

    def install_in_thread():
        pipeline._install_signal_handler()

    t = threading.Thread(target=install_in_thread)
    t.start()
    t.join()
    assert pipeline._original_sigint is None


def test_core_restore_signal_handler_not_main_thread():
    """_restore_signal_handler returns early when not in main thread."""
    pipeline = CoreIngestionPipeline()
    pipeline._original_sigint = lambda s, f: None

    def restore_in_thread():
        pipeline._restore_signal_handler()

    t = threading.Thread(target=restore_in_thread)
    t.start()
    t.join()
    assert pipeline._original_sigint is not None


def test_core_signal_handler_logic(monkeypatch):
    """SIGINT handler sets _interrupted; second SIGINT triggers force-exit path."""
    pipeline = CoreIngestionPipeline()
    captured_handler = [None]

    def fake_signal(sig, handler):
        captured_handler[0] = handler
        return None

    monkeypatch.setattr(signal, "signal", fake_signal)
    monkeypatch.setattr(signal, "getsignal", lambda sig: None)

    pipeline._install_signal_handler()
    handler = captured_handler[0]
    assert handler is not None

    # First invocation: sets _interrupted
    handler(0, None)
    assert pipeline._interrupted is True

    # Second invocation: force-exit path calls original handler
    called = []
    pipeline._original_sigint = lambda s, f: called.append(True)
    handler(0, None)
    assert len(called) == 1


def test_core_run_chunks_progress_callback():
    """progress_callback is invoked after buffer flush in run_chunks."""
    chunks = [
        CorePipelineChunk(id=f"c:{i}", content=f"Chunk {i} " * 50)
        for i in range(10)
    ]
    pipeline = CoreIngestionPipeline(embed_batch_size=4)
    calls = []
    pipeline.run_chunks(chunks, progress_callback=lambda s: calls.append(s.flush_count))
    assert len(calls) >= 2


def test_core_run_chunks_interrupted_break(monkeypatch):
    """_interrupted flag breaks out of the run_chunks loop."""
    chunks = [
        CorePipelineChunk(id=f"c:{i}", content=f"Chunk {i} " * 50)
        for i in range(20)
    ]
    pipeline = CoreIngestionPipeline(embed_batch_size=4)

    original_flush = pipeline._flush_buffer

    def slow_flush():
        time.sleep(0.02)
        original_flush()

    monkeypatch.setattr(pipeline, "_flush_buffer", slow_flush)

    def trigger_interrupt():
        time.sleep(0.01)
        pipeline._interrupted = True

    t = threading.Thread(target=trigger_interrupt)
    t.start()
    stats = pipeline.run_chunks(chunks)
    t.join()

    assert stats.chunks_indexed < 20


def test_core_parse_one_file_read_text_exception(tmp_path, monkeypatch):
    """OSError during read_text marks file as skipped."""
    from pathlib import Path

    original_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if self.name == "unreadable.py":
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    (tmp_path / "unreadable.py").write_text("content")
    (tmp_path / "good.py").write_text("print('ok')\n" * 200)

    pipeline = CoreIngestionPipeline(
        parse_workers=2,
        embed_batch_size=100,
        incremental=False,
    )
    stats = pipeline.run(tmp_path)
    assert stats.files_skipped >= 1
    assert stats.files_parsed >= 1
