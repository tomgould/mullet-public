"""Tests for mullet/hooks/session_context.py"""
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

_HOOK = Path(__file__).resolve().parent.parent / "mullet" / "hooks" / "session_context.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("session_context", _HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(payload: dict, sessions: list | None = None) -> str:
    """Run the hook main() and return captured stdout. sys.exit raises SystemExit."""
    hook = _load_hook()
    buf = io.StringIO()
    mock_sessions = sessions or []

    try:
        with (
            patch.object(sys, "stdin", io.StringIO(json.dumps(payload))),
            patch("mullet.core.agent.session.list_recent_sessions", return_value=mock_sessions),
            patch("sys.stdout", buf),
        ):
            hook.main()
    except SystemExit:
        pass
    return buf.getvalue()


class TestSessionContextHook:
    def test_no_output_when_no_sessions(self):
        out = _run({"turn_number": 0, "prompt": "hello"}, sessions=[])
        assert out.strip() == ""

    def test_injects_context_on_first_turn(self):
        sessions = [
            {"title": "Rewrote auth module", "messages": 42, "created_at": 0},
            {"title": "Fixed WAL bug", "messages": 7, "created_at": 0},
        ]
        out = _run({"turn_number": 0, "prompt": "continue"}, sessions=sessions)
        assert "Rewrote auth module" in out
        assert "Fixed WAL bug" in out

    def test_silent_on_subsequent_turns(self):
        sessions = [{"title": "Recent work", "messages": 5, "created_at": 0}]
        out = _run({"turn_number": 3, "prompt": "continue"}, sessions=sessions)
        assert out.strip() == ""

    def test_silent_when_turn_is_2(self):
        sessions = [{"title": "Recent work", "messages": 5, "created_at": 0}]
        out = _run({"turn_number": 2, "prompt": "go"}, sessions=sessions)
        assert out.strip() == ""

    def test_injects_on_turn_1(self):
        sessions = [{"title": "Test session", "messages": 3, "created_at": 0}]
        out = _run({"turn_number": 1, "prompt": "go"}, sessions=sessions)
        assert "Test session" in out

    def test_invalid_json_no_crash(self):
        hook = _load_hook()
        buf = io.StringIO()
        try:
            with (
                patch.object(sys, "stdin", io.StringIO("not json")),
                patch("sys.stdout", buf),
            ):
                hook.main()
        except SystemExit:
            pass
        assert buf.getvalue().strip() == ""

    def test_time_ago_minutes(self):
        hook = _load_hook()
        ts = time.time() - 600  # 10 minutes ago
        result = hook._time_ago(ts)
        assert "m ago" in result

    def test_time_ago_hours(self):
        hook = _load_hook()
        ts = time.time() - 7200  # 2 hours ago
        result = hook._time_ago(ts)
        assert "h ago" in result

    def test_time_ago_days(self):
        hook = _load_hook()
        ts = time.time() - 172800  # 2 days ago
        result = hook._time_ago(ts)
        assert "d ago" in result

    def test_time_ago_zero_ts(self):
        hook = _load_hook()
        result = hook._time_ago(0)
        assert result == "unknown"

    def test_includes_message_count(self):
        sessions = [{"title": "Big session", "messages": 99, "created_at": 0}]
        out = _run({"turn_number": 0, "prompt": "hey"}, sessions=sessions)
        assert "99 messages" in out
