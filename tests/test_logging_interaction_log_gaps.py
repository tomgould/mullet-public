"""Gap tests for mullet.core.logging.interaction_log — targets uncovered lines."""

import sqlite3
import os
from unittest.mock import MagicMock, patch

import pytest

import mullet.core.logging.interaction_log as ilog
from mullet.core.logging.interaction_log import (
    log_interaction,
    _extract_last_user_message,
    _extract_preview,
)


MESSAGES = [{"role": "user", "content": "hello world"}]


@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    monkeypatch.setenv("MULLET_INTERACTIONS_DB", db)
    monkeypatch.delenv("MULLET_LOG_FULL_PROMPTS", raising=False)
    monkeypatch.delenv("MULLET_LOG_FULL_RESPONSES", raising=False)
    ilog._CONNECTION = None
    ilog._DB_INITIALIZED = False
    ilog.MULLET_LOG_FULL_PROMPTS = False
    ilog.MULLET_LOG_FULL_RESPONSES = False
    yield db
    ilog._CONNECTION = None
    ilog._DB_INITIALIZED = False


# ── _get_connection uncovered lines ──────────────────────────────────────────

def test_get_connection_returns_cached_connection(reset_db):
    """Line 44: healthy cached connection is returned directly."""
    conn1 = ilog._get_connection()
    assert conn1 is not None
    conn2 = ilog._get_connection()
    assert conn2 is conn1


def test_get_connection_recovers_from_stale_connection(reset_db):
    """Lines 45-51: stale connection (SELECT 1 fails) triggers close + reset."""
    # Prime a healthy connection
    conn = ilog._get_connection()
    assert conn is not None

    # Inject a stale connection that fails on SELECT 1 and also fails to close
    stale = MagicMock()
    stale.execute.side_effect = sqlite3.OperationalError("database is locked")
    stale.close.side_effect = OSError("bad fd")
    ilog._CONNECTION = stale

    # Next call should recover and create a new real connection
    new_conn = ilog._get_connection()
    assert new_conn is not None
    assert new_conn is not stale


def test_get_connection_returns_none_on_connect_failure(reset_db, monkeypatch):
    """Lines 65-67: connection creation fails entirely."""
    monkeypatch.setattr(ilog, "_get_db_path", lambda: MagicMock(parent=MagicMock(mkdir=MagicMock())))
    with patch("sqlite3.connect", side_effect=sqlite3.Error("disk full")):
        conn = ilog._get_connection()
    assert conn is None


# ── _init_db uncovered lines ─────────────────────────────────────────────────

def test_init_db_rollback_on_schema_error(reset_db):
    """Lines 93-98: schema init error triggers ROLLBACK."""
    # Obtain a real connection
    conn = ilog._get_connection()
    assert conn is not None

    # Force a schema error by passing a mock that raises on CREATE TABLE
    bad_conn = MagicMock()
    bad_conn.execute.side_effect = sqlite3.Error("read-only")
    # _init_db should attempt ROLLBACK on the same connection
    ilog._init_db(bad_conn)
    # ROLLBACK was attempted
    assert any("ROLLBACK" in str(call) for call in bad_conn.execute.call_args_list)


# ── log_interaction uncovered lines ──────────────────────────────────────────

def test_log_interaction_returns_when_conn_is_none(reset_db):
    """Line 163: conn None path logs nothing and returns silently."""
    # Pre-create the DB so we can verify nothing was written even when _get_connection is patched
    sqlite3.connect(reset_db).execute("CREATE TABLE IF NOT EXISTS interactions (id INTEGER PRIMARY KEY)").connection.close()
    with patch.object(ilog, "_get_connection", return_value=None):
        # Should not raise
        log_interaction("test", "model", MESSAGES, "resp", 1, 1, 1, True)
    # Verify nothing was written
    rows = sqlite3.connect(reset_db).execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    assert rows == 0


def test_log_interaction_rollback_failure_swallowed(reset_db):
    """Lines 204-205: rollback itself raising is swallowed."""
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.Error("constraint")
    with patch.object(ilog, "_get_connection", return_value=mock_conn):
        # Should not raise even though execute (INSERT and ROLLBACK) raises
        log_interaction("test", "model", MESSAGES, "resp", 1, 1, 1, True)


# ── _cleanup uncovered lines ─────────────────────────────────────────────────

def test_cleanup_close_failure_logged(reset_db, caplog):
    """Lines 212-219: _cleanup handles close() exception gracefully."""
    mock_conn = MagicMock()
    mock_conn.close.side_effect = OSError("bad fd")
    ilog._CONNECTION = mock_conn
    try:
        ilog._cleanup()
    finally:
        pass
    assert ilog._CONNECTION is None


# ── _extract_last_user_message uncovered lines ───────────────────────────────

def test_extract_last_user_message_list_content():
    """Lines 121-130: content as list of text/image blocks."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello from list"},
                {"type": "image", "url": "img.png"},
                {"type": "text", "text": "Second part"},
            ],
        }
    ]
    result = _extract_last_user_message(messages)
    assert "Hello from list" in result
    assert "Second part" in result
    assert "img.png" not in result


def test_extract_last_user_message_list_of_strings():
    """Line 128: content list containing plain strings."""
    messages = [{"role": "user", "content": ["plain string"]}]
    result = _extract_last_user_message(messages)
    assert result == "plain string"


def test_extract_last_user_message_empty_list():
    """Line 113: empty messages list returns empty string."""
    assert _extract_last_user_message([]) == ""


def test_extract_last_user_message_no_user_role():
    """Line 130: no user message in list returns empty string."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "reply"},
    ]
    assert _extract_last_user_message(messages) == ""


# ── _extract_preview uncovered lines ─────────────────────────────────────────

def test_extract_preview_none():
    """Line 104: None input returns empty string."""
    assert _extract_preview(None) == ""


def test_extract_preview_truncation():
    """Line 107: long text is truncated with ellipsis."""
    text = "x" * 600
    result = _extract_preview(text, max_length=500)
    assert result.endswith("...")
    assert len(result) == 500


# ── log_interaction success path ─────────────────────────────────────────────

def test_log_interaction_successful_write(reset_db):
    """Lines 176-199: successful INSERT into interactions table."""
    messages = [{"role": "user", "content": "hello"}]
    log_interaction(
        provider="kimi",
        model="kimi-k2",
        messages=messages,
        response_content="world",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=100,
        success=True,
        error=None,
        tier="STANDARD",
    )
    conn = sqlite3.connect(reset_db)
    rows = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    conn.close()
    assert rows == 1
