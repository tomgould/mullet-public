"""Tests for mullet/hooks/pre_agent_guard.py"""
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

_HOOK = Path(__file__).resolve().parent.parent / "mullet" / "hooks" / "pre_agent_guard.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("pre_agent_guard", _HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(payload: dict) -> str:
    hook = _load_hook()
    buf = io.StringIO()
    with (
        patch.object(sys, "stdin", io.StringIO(json.dumps(payload))),
        patch("sys.stdout", buf),
    ):
        hook.main()
    return buf.getvalue()


class TestPreAgentGuard:
    def test_warns_when_no_model_specified(self):
        out = _run({"tool_name": "Task", "tool_input": {"description": "explore"}})
        assert "pre-agent guard" in out
        assert "Opus" in out

    def test_silent_with_haiku_model(self):
        out = _run({"tool_name": "Task", "tool_input": {"description": "explore", "model": "haiku"}})
        assert out.strip() == ""

    def test_silent_with_sonnet_model(self):
        out = _run({"tool_name": "Task", "tool_input": {"description": "explore", "model": "sonnet"}})
        assert out.strip() == ""

    def test_silent_with_haiku_4_5(self):
        out = _run({"tool_name": "Task", "tool_input": {"description": "test", "model": "claude-haiku-4-5-20251001"}})
        assert out.strip() == ""

    def test_warns_with_opus_model(self):
        out = _run({"tool_name": "Task", "tool_input": {"description": "test", "model": "claude-opus-4-7"}})
        assert "pre-agent guard" in out

    def test_non_task_tool_silent(self):
        out = _run({"tool_name": "Read", "tool_input": {"file_path": "/tmp/foo.py"}})
        assert out.strip() == ""

    def test_invalid_json_no_crash(self):
        hook = _load_hook()
        buf = io.StringIO()
        with (
            patch.object(sys, "stdin", io.StringIO("not json")),
            patch("sys.stdout", buf),
        ):
            hook.main()  # must not raise
        # no crash is the assertion; any output is acceptable
        _ = buf.getvalue()

    def test_missing_tool_name_silent(self):
        out = _run({"tool_input": {"description": "test"}})
        assert out.strip() == ""

    def test_warning_is_non_blocking(self):
        out = _run({"tool_name": "Task", "tool_input": {}})
        assert "Proceeding anyway" in out
