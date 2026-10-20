"""Cycle 20 silent-failure round 2: ingest.py + mcp_handlers.py.

Source TODO row (docs/TODO.md, P1 — SILENT FAILURES, 2026-04-19 review):

  - [ ] **`ingest.py` `ingest_directory`** — broad `except` after document
        processing silently counts failure but never propagates.
  - [ ] **`mcp_handlers.py`** — 13 bare except sites; most mid-severity;
        add `logger.warning` to each catch.

Audit findings (Cycle 20):

  ingest.py — 17 catch sites in total. 14 already log; 3 silent
  fall-throughs (two mbox date-parse fall-throughs at L562/L661, plus the
  per-file ``ingest_directory`` catch at L902 which logged at ERROR but
  without traceback context). Cycle 20 promotes the per-file catch to
  WARNING+exc_info and adds WARNING logs to the date-parse fall-throughs.

  mcp_handlers.py — 28 catch sites in total (TODO claim of "13" was
  stale). Of those: 24 already log at WARNING+; 4 sites either logged
  only at DEBUG (silent at default level) or silently swallowed:
    - L42  ``_sanitize_paths`` outer except (silent string return)
    - L403 ``_build_fallback_chain`` Kimi factory (logger.debug)
    - L577 inner JSON char-fallback for prompt-token estimate (bare pass)
    - L1004 ``handle_mullet_budget`` BudgetTracker init/status (logger.debug)
  All four promoted to WARNING with explicit handler context.

Tests below split into three buckets:
  1. ingest_directory per-file logging contract + KeyboardInterrupt safety
  2. mcp_handlers silent-fix sites — assert WARNING contract
  3. Regression guards for category-A sites (already-logging) so a future
     refactor that swaps a logger.warning back to a bare ``pass`` trips
     CI before production.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Bucket 1 — ingest.py::ingest_directory
# ---------------------------------------------------------------------------


def test_ingest_directory_logs_per_file_failure(tmp_path, caplog):
    """ingest_directory: per-file failure logs WARNING + traceback.

    Failure mode (TODO P1, ingest.py): broad except increments
    ``failed_files`` counter without surfacing the error. After Cycle 20
    every failed file MUST emit a WARNING with the file name, the
    exception text, and ``exc_info`` so tracebacks reach the operator.
    Other files in the batch must still complete.
    """
    from mullet.knowledge.engine import ingest as ingest_module

    # Create three text files. The middle one will explode on ingest.
    paths = []
    for i in range(3):
        p = tmp_path / f"file_{i}.txt"
        p.write_text(f"content {i}")
        paths.append(p)

    fake_ingestor = MagicMock()

    def _process(path, *args, **kwargs):
        if path.name == "file_1.txt":
            raise RuntimeError("kaboom-1")
        return [{"id": "x", "content": "ok", "metadata": {}}], [[0.1]]

    fake_ingestor.ingest_document.side_effect = _process

    caplog.set_level(logging.WARNING, logger="mullet.knowledge.engine.ingest")

    chunks, embeddings = ingest_module.ingest_directory(
        directory=tmp_path,
        ingestor=fake_ingestor,
        extensions=(".txt",),
        recursive=False,
    )

    # Counter behaviour preserved: two files succeeded, one is in failed_files
    # (we can't see progress directly, but chunks/embeddings reflect 2 successes).
    assert len(chunks) == 2, f"expected 2 successful files, got {len(chunks)}"
    assert len(embeddings) == 2

    # Logging contract: WARNING message contains file name AND exception text
    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "file_1.txt" in r.getMessage()
    ]
    assert warn_records, (
        "expected WARNING log for failing file_1.txt; "
        f"got: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    assert any("kaboom-1" in r.getMessage() for r in warn_records), (
        "expected exception text in log message"
    )
    # exc_info should be attached for traceability
    assert any(r.exc_info is not None for r in warn_records), (
        "expected exc_info=True so traceback reaches the operator"
    )


def test_ingest_directory_does_not_swallow_keyboard_interrupt(tmp_path):
    """ingest_directory: KeyboardInterrupt MUST propagate, not be caught.

    Failure mode: a future refactor changing ``except Exception:`` to
    ``except:`` would trap Ctrl+C and make the process unkillable
    mid-batch. Lock the contract here.
    """
    from mullet.knowledge.engine import ingest as ingest_module

    p = tmp_path / "file_0.txt"
    p.write_text("content")

    fake_ingestor = MagicMock()
    fake_ingestor.ingest_document.side_effect = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        ingest_module.ingest_directory(
            directory=tmp_path,
            ingestor=fake_ingestor,
            extensions=(".txt",),
            recursive=False,
        )


def test_ingest_mbox_date_parse_failure_logs_warning(tmp_path, caplog):
    """ingest_mbox: malformed Date header logs WARNING + falls back to raw value.

    Failure mode (TODO P1): silent fall-through hid date-parse regressions.
    Cycle 20 contract: WARNING with handler name, raw value, and reason.
    """
    from mullet.knowledge.engine import ingest as ingest_module

    # Build a minimal mbox with one message that has an unparseable date
    mbox_path = tmp_path / "test.mbox"
    mbox_path.write_text(
        "From sender@example.com Mon Jan 01 00:00:00 2024\r\n"
        "Subject: hello\r\n"
        "From: sender@example.com\r\n"
        "Date: not-a-real-date\r\n"
        "\r\n"
        "body content goes here that is long enough to not be skipped\r\n"
        "\r\n"
    )

    caplog.set_level(logging.WARNING, logger="mullet.knowledge.engine.ingest")

    fake_ingestor = MagicMock(spec=ingest_module.DocumentIngestor)
    fake_ingestor.chunk_max_chars = 1500
    fake_ingestor.chunk_overlap = 100
    # Stub out create_embeddings so we don't need a real model
    fake_ingestor.create_embeddings = lambda chunks: [[0.0]] * len(chunks)
    # Bind real method via the unbound descriptor
    real_method = ingest_module.DocumentIngestor.ingest_mbox.__get__(fake_ingestor)
    fake_ingestor._split_email_content = (
        ingest_module.DocumentIngestor._split_email_content.__get__(fake_ingestor)
    )

    real_method(mbox_path)

    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "date parse failed" in r.getMessage()
    ]
    assert warn_records, (
        "expected WARNING log for malformed mbox date; "
        f"got: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Bucket 2 — mcp_handlers.py silent-fix sites
# ---------------------------------------------------------------------------


def test_sanitize_paths_logs_warning_on_resolution_failure(caplog):
    """_sanitize_paths: previously silent except now emits WARNING.

    Failure mode: outer ``except Exception`` returned an error string but
    never logged the underlying failure (e.g. None target_dir, ENOENT,
    permission errors during ``Path.resolve``).
    """
    from mullet.server import mcp_handlers

    caplog.set_level(logging.WARNING, logger="mullet.server.mcp_handlers")

    # Force resolution failure by passing an unhashable type that breaks Path
    with patch("mullet.server.mcp_handlers.Path") as mock_path:
        mock_path.side_effect = TypeError("simulated path failure")
        files, target, err = mcp_handlers._sanitize_paths(["a.py"], "/tmp/x")

    assert err is not None and "Path sanitization error" in err
    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "_sanitize_paths" in r.getMessage()
    ]
    assert warn_records, (
        f"expected WARNING from _sanitize_paths; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )


def test_build_fallback_chain_kimi_failure_logs_warning(caplog, monkeypatch):
    """_build_fallback_chain: Kimi factory exception now logs WARNING.

    Failure mode: previously logged at DEBUG only, so a stack of broken
    Kimi imports / corrupt model registry returned an empty fallback chain
    silently from the operator's perspective.
    """
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    caplog.set_level(logging.WARNING, logger="mullet.server.mcp_handlers")

    with patch("mullet.core.providers.model_registry.get_registry") as mock_reg:
        mock_reg.side_effect = RuntimeError("registry boom")
        chain = mcp_handlers._build_fallback_chain(requested_tier="STANDARD")

    assert chain == []
    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "_build_fallback_chain" in r.getMessage()
        and "Kimi" in r.getMessage()
    ]
    assert warn_records, (
        f"expected WARNING from _build_fallback_chain Kimi factory; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )


def test_handle_mullet_budget_logs_warning_on_failure(caplog):
    """handle_mullet_budget: BudgetTracker failure now logs WARNING.

    Failure mode: previously DEBUG-only, the silent fallback dict masked
    a broken budget tracker (corrupt state file, missing module).
    """
    from mullet.server import mcp_handlers

    caplog.set_level(logging.WARNING, logger="mullet.server.mcp_handlers")

    with patch("mullet.core.router.budget.BudgetTracker") as mock_bt:
        mock_bt.side_effect = RuntimeError("tracker boom")
        result = json.loads(mcp_handlers.handle_mullet_budget({}))

    # Fallback shape preserved
    assert "session_tokens" in result
    assert result.get("note", "").startswith("Budget tracker unavailable")

    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "handle_mullet_budget" in r.getMessage()
    ]
    assert warn_records, (
        f"expected WARNING from handle_mullet_budget; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Bucket 3 — Regression guards: Category-A sites already logging
# ---------------------------------------------------------------------------


def test_handle_mullet_search_failure_logs_warning(caplog):
    """handle_mullet_search: regression guard — failure path keeps WARNING.

    Pre-Cycle-20 site at L1036 already logs at WARNING. This test pins
    the contract so a future refactor doesn't accidentally drop it.
    """
    from mullet.server import mcp_handlers

    caplog.set_level(logging.WARNING, logger="mullet.server.mcp_handlers")

    with patch("mullet.core.knowledge.manager.knowledge_session") as mock_sess:
        mock_sess.side_effect = RuntimeError("knowledge boom")
        result = json.loads(mcp_handlers.handle_mullet_search(
            {"query": "x", "client": "fake"}
        ))

    assert "error" in result
    assert any(
        r.levelno >= logging.WARNING and "MCP search failed" in r.getMessage()
        for r in caplog.records
    ), f"expected WARNING from handle_mullet_search; got: {caplog.records}"


def test_handle_mullet_remember_failure_logs_warning(caplog):
    """handle_mullet_remember: regression guard for existing WARNING contract."""
    from mullet.server import mcp_handlers

    caplog.set_level(logging.WARNING, logger="mullet.server.mcp_handlers")

    with patch("mullet.core.knowledge.manager.knowledge_session") as mock_sess:
        mock_sess.side_effect = RuntimeError("remember boom")
        result = json.loads(mcp_handlers.handle_mullet_remember(
            {"key": "k", "content": "c", "client": "fake"}
        ))

    assert "error" in result
    assert any(
        r.levelno >= logging.WARNING and "MCP remember failed" in r.getMessage()
        for r in caplog.records
    ), f"expected WARNING from handle_mullet_remember; got: {caplog.records}"


def test_handle_mullet_swarm_failure_logs_error_and_returns_dict(caplog):
    """handle_mullet_swarm: regression guard for ERROR-level failure log."""
    from mullet.server import mcp_handlers

    caplog.set_level(logging.WARNING, logger="mullet.server.mcp_handlers")

    with patch("mullet.core.orchestration.swarm.run_swarm") as mock_run:
        mock_run.side_effect = RuntimeError("swarm boom")
        result = json.loads(mcp_handlers.handle_mullet_swarm({"goal": "x"}))

    assert "error" in result and "swarm boom" in result["error"]
    assert any(
        r.levelno >= logging.ERROR and "Swarm failed" in r.getMessage()
        for r in caplog.records
    ), f"expected ERROR from handle_mullet_swarm; got: {caplog.records}"


# ---------------------------------------------------------------------------
# Bucket 4 — Failure surfaces in structured response (D1/D2/D7 contract)
# ---------------------------------------------------------------------------


def test_handle_mullet_run_path_traversal_returns_dict_with_error():
    """handle_mullet_run: path-traversal failure surfaces as dict with error.

    D7 contract: every return is a dict; sanitization warning side-channel
    additionally reaches the logger (covered above).
    """
    from mullet.server import mcp_handlers

    result = mcp_handlers.handle_mullet_run({
        "task": "do thing",
        "files": ["../../../etc/passwd"],
        "target_dir": "/tmp",
    })
    assert isinstance(result, dict)
    assert "error" in result
    assert "traversal" in result["error"].lower() or "sanitiz" in result["error"].lower()
