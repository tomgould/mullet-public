"""handle_mullet_opencode is a thin wrapper over mullet.jobs.delegate.delegate.

delegate is replaced by a stub throughout, so no job runs and no ledger is opened.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mullet.core.env import DELEGATION_OFF_MESSAGE
from mullet.server import mcp_handlers

RESULT_KEYS = {"status", "reason", "job_id", "report_path", "rung", "model", "cost",
               "gates", "files_changed", "notes"}


def _stub_result(**overrides) -> dict:
    out = {
        "status": "applied", "reason": None, "job_id": "job-1", "report_path": "/x/report.md",
        "rung": "kimi", "model": "kimi-k2", "cost": 0.01, "gates": ["tests passed"],
        "files_changed": ["a.py"],
    }
    out.update(overrides)
    return out


@pytest.fixture()
def stub(monkeypatch):
    """Delegation on, no preset, a fake ledger, and delegate replaced by a stub."""
    monkeypatch.setenv("MULLET_DELEGATION", "on")
    ledger_cls = MagicMock()
    fake = MagicMock(return_value=_stub_result())
    with patch.object(mcp_handlers, "_active_preset", return_value=None), \
         patch("mullet.jobs.ledger.Ledger", ledger_cls), \
         patch("mullet.jobs.delegate.delegate", fake):
        fake.ledger_cls = ledger_cls
        yield fake


def _kwargs(stub) -> dict:
    stub.assert_called_once()
    return stub.call_args.kwargs


class TestArgumentsPassedThrough:
    def test_files_cwd_timeout_and_providers(self, stub):
        result = mcp_handlers.handle_mullet_opencode({
            "task": "rename foo", "files": ["a.py", "b.py"], "cwd": "/repo",
            "timeout": 5, "providers": ["kimi", "local"],
        })
        kw = _kwargs(stub)
        assert kw["task"] == "rename foo"
        assert kw["paths"] == ["a.py", "b.py"]
        assert kw["cwd"] == "/repo"
        assert kw["timeout_min"] == 5
        assert kw["providers"] == ["kimi", "local"]
        assert result["status"] == "applied"
        assert result["job_id"] == "job-1"

    def test_ledger_is_passed_to_delegate(self, stub):
        mcp_handlers.handle_mullet_opencode({"task": "t"})
        ledger = stub.ledger_cls.return_value.__enter__.return_value
        assert stub.call_args.args[0] is ledger

    def test_paths_key_and_repo_root(self, stub):
        mcp_handlers.handle_mullet_opencode({"task": "t", "paths": ["src/x.py"], "repo_root": "/r"})
        kw = _kwargs(stub)
        assert kw["paths"] == ["src/x.py"]
        assert kw["cwd"] == "/r"

    def test_defaults(self, stub):
        result = mcp_handlers.handle_mullet_opencode({"task": "t"})
        kw = _kwargs(stub)
        assert kw["paths"] is None
        assert kw["cwd"] is None
        assert kw["timeout_min"] == 20
        assert kw["providers"] is None
        assert result["notes"] == []

    @pytest.mark.parametrize("given, expected", [(0, 1), (-3, 1), (61, 60), (500, 60)])
    def test_timeout_is_clamped(self, stub, given, expected):
        result = mcp_handlers.handle_mullet_opencode({"task": "t", "timeout": given})
        assert _kwargs(stub)["timeout_min"] == expected
        assert any("clamped" in n for n in result["notes"])

    def test_non_numeric_timeout_uses_the_default(self, stub):
        result = mcp_handlers.handle_mullet_opencode({"task": "t", "timeout": "soon"})
        assert _kwargs(stub)["timeout_min"] == 20
        assert any("'timeout'" in n for n in result["notes"])

    def test_providers_as_comma_separated_string(self, stub):
        mcp_handlers.handle_mullet_opencode({"task": "t", "providers": "Kimi, openrouter"})
        assert _kwargs(stub)["providers"] == ["kimi", "openrouter"]


class TestIgnoredArguments:
    def test_old_arguments_are_noted_and_not_passed(self, stub):
        result = mcp_handlers.handle_mullet_opencode({
            "task": "t", "model": "openrouter/x", "tier": "LOW", "speed_mode": "fast",
            "provider": "kimi", "apply": True, "colour": "blue",
        })
        notes = result["notes"]
        for key in ("model", "tier", "speed_mode", "provider", "apply", "colour"):
            assert any(n.startswith(f"ignored '{key}'") for n in notes), key
        assert "ignored 'model': the delegation ladder picks the worker" in notes
        kw = _kwargs(stub)
        assert set(kw) == {"task", "paths", "cwd", "timeout_min", "providers"}


class TestFailures:
    @pytest.mark.parametrize("args", [{}, {"task": ""}, {"task": "   "}])
    def test_empty_task_fails_cleanly(self, stub, args):
        result = mcp_handlers.handle_mullet_opencode(args)
        assert result["status"] == "failed"
        assert result["reason"]
        assert set(result) == RESULT_KEYS
        stub.assert_not_called()

    @pytest.mark.parametrize("exc", [ValueError("path 'x' is outside the repo"),
                                     RuntimeError("fatal: not a git repository")])
    def test_exception_becomes_failed(self, stub, exc):
        stub.side_effect = exc
        result = mcp_handlers.handle_mullet_opencode({"task": "t", "model": "m"})
        assert result["status"] == "failed"
        assert result["reason"] == str(exc)
        assert result["job_id"] is None
        assert result["notes"] == ["ignored 'model': the delegation ladder picks the worker"]
        assert set(result) == RESULT_KEYS

    def test_ledger_error_becomes_failed(self, stub):
        stub.ledger_cls.side_effect = OSError("disk full")
        result = mcp_handlers.handle_mullet_opencode({"task": "t"})
        assert result["status"] == "failed"
        assert "disk full" in result["reason"]
        stub.assert_not_called()

    def test_delegation_off_short_circuits(self, stub, monkeypatch):
        monkeypatch.setenv("MULLET_DELEGATION", "off")
        result = mcp_handlers.handle_mullet_opencode({"task": "t"})
        assert result["error"] == DELEGATION_OFF_MESSAGE
        assert result["delegation"] == "off"
        stub.assert_not_called()

    def test_native_only_preset_refuses(self, stub):
        preset = MagicMock(native_only=True)
        preset.name = "claude-only"
        with patch.object(mcp_handlers, "_active_preset", return_value=preset):
            result = mcp_handlers.handle_mullet_opencode({"task": "t"})
        assert result["status"] == "blocked"
        assert "claude-only" in result["reason"]
        stub.assert_not_called()


class TestResultShape:
    def test_delegate_result_plus_notes(self, stub):
        result = mcp_handlers.handle_mullet_opencode({"task": "t"})
        assert set(result) == RESULT_KEYS
        assert result["gates"] == ["tests passed"]
        assert result["files_changed"] == ["a.py"]


class TestSchema:
    def _schema(self) -> dict:
        tool = next(t for t in mcp_handlers.MCP_TOOLS if t["name"] == "mullet_opencode")
        return tool

    def test_properties(self):
        tool = self._schema()
        props = tool["inputSchema"]["properties"]
        assert {"task", "files", "cwd", "timeout", "providers", "model"} <= set(props)
        assert tool["inputSchema"]["required"] == ["task"]
        assert props["files"]["type"] == "array"
        assert props["timeout"]["type"] == "integer"
        assert props["timeout"]["default"] == 20
        assert props["timeout"]["maximum"] == 60
        assert props["providers"]["type"] == "array"
        assert set(props["providers"]["items"]["enum"]) == {"kimi", "openrouter", "free", "local", "claude"}
        assert "ignored" in props["model"]["description"].lower()

    def test_description(self):
        desc = self._schema()["description"]
        assert "ladder" in desc
        assert "gates pass" in desc
        assert "never committed" in desc
        assert "—" not in desc

    def test_no_schema_claims_zero_anthropic_tokens(self):
        for tool in mcp_handlers.MCP_TOOLS:
            assert "zero Anthropic tokens" not in tool["description"], tool["name"]
