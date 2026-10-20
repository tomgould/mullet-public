"""B2 coverage: IngestionPipeline gaps from 2026-04-19 audit.

Cycle 6 — regression guards for `_parse_document_file` (path edge cases)
and `_install_signal_handler` (non-main-thread guard). The flush_buffer
upsert failure mode is already covered by
`tests/test_knowledge_engine_silent_failures.py` and
`tests/test_knowledge_pipeline.py` from earlier cycles.
"""
from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path
from unittest.mock import MagicMock



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pipeline(ingestor=None, searcher=None):
    """Build an IngestionPipeline with minimal mocks."""
    from mullet.knowledge.engine.pipeline import IngestionPipeline

    if ingestor is None:
        ingestor = MagicMock()
    if searcher is None:
        searcher = MagicMock()
        searcher._sanitize_metadata = lambda m: m
        searcher.collection = MagicMock()

    return IngestionPipeline(
        ingestor=ingestor,
        searcher=searcher,
        parse_workers=1,
        embed_batch_size=4,
        contextualize=False,
        resource_manager=None,
    )


# ---------------------------------------------------------------------------
# B2.1 — _parse_document_file path edge cases
# ---------------------------------------------------------------------------


def test_parse_document_file_absolute_path(tmp_path):
    """Absolute path under base_directory → relative path used as source_file."""
    base = tmp_path
    f = base / "doc.txt"
    f.write_text("hello world")

    ingestor = MagicMock()
    ingestor.parse_document.return_value = ["element"]
    ingestor.chunk_elements.return_value = [
        {"id": "c0", "content": "hello world", "metadata": {}}
    ]

    pipeline = _make_pipeline(ingestor=ingestor)
    chunks = pipeline._parse_document_file(f, base)

    assert len(chunks) == 1
    # Relative path should be just "doc.txt"
    assert chunks[0]["metadata"]["source_file"] == "doc.txt"
    assert chunks[0]["metadata"]["source_path"] == str(f)


def test_parse_document_file_nested_relative_path(tmp_path):
    """File several levels below base → relative path includes the subdirs."""
    base = tmp_path
    sub = base / "a" / "b" / "c"
    sub.mkdir(parents=True)
    f = sub / "deep.txt"
    f.write_text("deep")

    ingestor = MagicMock()
    ingestor.parse_document.return_value = ["el"]
    ingestor.chunk_elements.return_value = [
        {"id": "c0", "content": "deep", "metadata": {}}
    ]

    pipeline = _make_pipeline(ingestor=ingestor)
    chunks = pipeline._parse_document_file(f, base)

    assert chunks[0]["metadata"]["source_file"] == str(Path("a") / "b" / "c" / "deep.txt")


def test_parse_document_file_path_outside_base_falls_back_to_name(tmp_path):
    """File outside base_directory → ValueError on relative_to → fallback to .name.

    Documents the explicit behaviour at lines 196-198 of pipeline.py.
    """
    base = tmp_path / "base"
    base.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    f = other / "outside.txt"
    f.write_text("outside")

    ingestor = MagicMock()
    ingestor.parse_document.return_value = ["el"]
    ingestor.chunk_elements.return_value = [
        {"id": "c0", "content": "outside", "metadata": {}}
    ]

    pipeline = _make_pipeline(ingestor=ingestor)
    chunks = pipeline._parse_document_file(f, base)

    assert len(chunks) == 1
    # Falls back to file name only, not full path
    assert chunks[0]["metadata"]["source_file"] == "outside.txt"
    assert chunks[0]["metadata"]["source_path"] == str(f)


def test_parse_document_file_unicode_filename(tmp_path):
    """Filename with non-ASCII chars → parses fine, metadata preserved."""
    base = tmp_path
    f = base / "文档-café-naïve.txt"
    f.write_text("unicode")

    ingestor = MagicMock()
    ingestor.parse_document.return_value = ["el"]
    ingestor.chunk_elements.return_value = [
        {"id": "c0", "content": "unicode", "metadata": {}}
    ]

    pipeline = _make_pipeline(ingestor=ingestor)
    chunks = pipeline._parse_document_file(f, base)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source_file"] == "文档-café-naïve.txt"


def test_parse_document_file_missing_file_returns_empty(tmp_path, caplog):
    """Non-existent file → ingestor.parse_document raises → method returns [].

    Regression guard for the broad except at lines 205-207 — the failure
    must be logged at ERROR but not propagate.
    """
    base = tmp_path
    f = base / "does-not-exist.txt"

    ingestor = MagicMock()
    ingestor.parse_document.side_effect = FileNotFoundError(str(f))

    pipeline = _make_pipeline(ingestor=ingestor)

    with caplog.at_level(logging.ERROR, logger="mullet.knowledge.engine.pipeline"):
        chunks = pipeline._parse_document_file(f, base)

    assert chunks == []
    errors = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
        and r.name == "mullet.knowledge.engine.pipeline"
        and "does-not-exist.txt" in r.getMessage()
    ]
    assert errors, "expected ERROR log naming the failed file"


def test_parse_document_file_mbox_dispatch(tmp_path):
    """.mbox files take the mbox-specific code path, not parse_document.

    Documents the suffix-dispatch branch at lines 189-193.
    """
    base = tmp_path
    f = base / "archive.mbox"
    f.write_text("From foo")

    ingestor = MagicMock()
    ingestor.ingest_mbox_chunks_only.return_value = [
        {"id": "m0", "content": "msg", "metadata": {}}
    ]

    pipeline = _make_pipeline(ingestor=ingestor)
    chunks = pipeline._parse_document_file(f, base)

    ingestor.ingest_mbox_chunks_only.assert_called_once_with(f)
    ingestor.parse_document.assert_not_called()
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source_file"] == "archive.mbox"


# ---------------------------------------------------------------------------
# B2.2 — Signal handler non-main-thread guard
# ---------------------------------------------------------------------------


def test_install_signal_handler_in_non_main_thread_no_crash():
    """_install_signal_handler must silently no-op when called from a worker
    thread (Python's signal.signal raises ValueError if not in main thread).

    Regression guard for the threading.main_thread() check at lines 76-78.
    """
    pipeline = _make_pipeline()
    error = {"e": None}

    def worker():
        try:
            pipeline._install_signal_handler()
        except Exception as e:
            error["e"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)

    assert error["e"] is None, f"installer must not raise in worker thread: {error['e']}"
    # And it should NOT have set _original_sigint, since it bailed early
    assert pipeline._original_sigint is None


def test_install_signal_handler_in_main_thread_installs_handler():
    """In the main thread, the handler IS installed and capturable for restore."""
    pipeline = _make_pipeline()
    original_before = signal.getsignal(signal.SIGINT)
    try:
        pipeline._install_signal_handler()
        # The pipeline should have captured the original handler
        assert pipeline._original_sigint == original_before
        # And installed a new one
        new_handler = signal.getsignal(signal.SIGINT)
        assert new_handler != original_before
        assert callable(new_handler)
    finally:
        # Cleanup — restore original
        signal.signal(signal.SIGINT, original_before)
        pipeline._original_sigint = None


def test_restore_signal_handler_in_non_main_thread_no_crash():
    """_restore_signal_handler must also no-op in worker threads."""
    pipeline = _make_pipeline()
    pipeline._original_sigint = signal.getsignal(signal.SIGINT)
    error = {"e": None}

    def worker():
        try:
            pipeline._restore_signal_handler()
        except Exception as e:
            error["e"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)

    assert error["e"] is None, f"restore must not raise in worker thread: {error['e']}"


def test_install_then_restore_signal_handler_round_trip():
    """Install + restore round trip — original handler must be back in place."""
    pipeline = _make_pipeline()
    original = signal.getsignal(signal.SIGINT)
    try:
        pipeline._install_signal_handler()
        assert signal.getsignal(signal.SIGINT) != original
        pipeline._restore_signal_handler()
        assert signal.getsignal(signal.SIGINT) == original
        assert pipeline._original_sigint is None
    finally:
        signal.signal(signal.SIGINT, original)
