"""Tests for routing_history wiring in handle_mullet_run."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


from mullet.core.routing_history import make_signature
from mullet.server.mcp_handlers import handle_mullet_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUCCESS_RESULT = {
    "completed": True,
    "response": "done",
    "actual_provider": "kimi",
    "actual_model": "kimi-k2",
    "tool_calls": 0,
    "duration": 0.5,
    "tokens": 100,
    "prompt_tokens": 60,
    "completion_tokens": 40,
}

_FAILURE_RESULT = {
    "error": "provider resolution failed: no provider",
    "actual_provider": "unknown",
    "actual_model": "unknown",
}


def _read_rows(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_success_outcome_recorded(tmp_path: Path) -> None:
    """A successful call appends one row with success=True."""
    task = "summarise the codebase"
    tier = "STANDARD"

    with (
        patch("mullet.server.mcp_handlers._handle_mullet_run_inner", return_value=_SUCCESS_RESULT),
        patch("mullet.core.routing_history.history_path", return_value=tmp_path / ".mullet" / "routing_history.jsonl"),
    ):
        (tmp_path / ".mullet").mkdir(parents=True, exist_ok=True)
        result = handle_mullet_run({"task": task, "tier": tier})

    assert result == _SUCCESS_RESULT

    rows = _read_rows(tmp_path / ".mullet" / "routing_history.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["success"] is True
    assert row["provider"] == "kimi"
    assert row["model"] == "kimi-k2"
    assert row["tokens_in"] == 60
    assert row["tokens_out"] == 40
    assert row["task_signature"] == make_signature(task, "run", tier)
    assert "mcp" in row["tags"]


def test_failure_outcome_recorded(tmp_path: Path) -> None:
    """An error-dict return is recorded with success=False and error_class=None (no exception raised)."""
    task = "fix bug in router"
    tier = "PREMIUM"

    with (
        patch("mullet.server.mcp_handlers._handle_mullet_run_inner", return_value=_FAILURE_RESULT),
        patch("mullet.core.routing_history.history_path", return_value=tmp_path / ".mullet" / "routing_history.jsonl"),
    ):
        (tmp_path / ".mullet").mkdir(parents=True, exist_ok=True)
        result = handle_mullet_run({"task": task, "tier": tier})

    assert result["error"]

    rows = _read_rows(tmp_path / ".mullet" / "routing_history.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["success"] is False
    assert row["error_class"] is None  # error came from dict, not raised exception


def test_wall_time_ms_positive(tmp_path: Path) -> None:
    """wall_time_ms must be a positive integer."""
    import time

    def _slow_inner(_args):
        time.sleep(0.01)
        return _SUCCESS_RESULT

    with (
        patch("mullet.server.mcp_handlers._handle_mullet_run_inner", side_effect=_slow_inner),
        patch("mullet.core.routing_history.history_path", return_value=tmp_path / ".mullet" / "routing_history.jsonl"),
    ):
        (tmp_path / ".mullet").mkdir(parents=True, exist_ok=True)
        handle_mullet_run({"task": "a task"})

    rows = _read_rows(tmp_path / ".mullet" / "routing_history.jsonl")
    assert rows[0]["wall_time_ms"] > 0
    assert isinstance(rows[0]["wall_time_ms"], int)


def test_task_signature_populated(tmp_path: Path) -> None:
    """task_signature matches make_signature output."""
    task = "analyse performance bottlenecks"
    tier = "STANDARD"

    with (
        patch("mullet.server.mcp_handlers._handle_mullet_run_inner", return_value=_SUCCESS_RESULT),
        patch("mullet.core.routing_history.history_path", return_value=tmp_path / ".mullet" / "routing_history.jsonl"),
    ):
        (tmp_path / ".mullet").mkdir(parents=True, exist_ok=True)
        handle_mullet_run({"task": task, "tier": tier})

    rows = _read_rows(tmp_path / ".mullet" / "routing_history.jsonl")
    assert rows[0]["task_signature"] == make_signature(task, "run", tier)


def test_recording_error_does_not_change_return(tmp_path: Path) -> None:
    """If record() raises, the handler still returns the original result unchanged."""
    with (
        patch("mullet.server.mcp_handlers._handle_mullet_run_inner", return_value=_SUCCESS_RESULT),
        patch("mullet.core.routing_history.record", side_effect=RuntimeError("disk full")),
    ):
        result = handle_mullet_run({"task": "some task"})

    assert result == _SUCCESS_RESULT
