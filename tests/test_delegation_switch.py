"""MULLET_DELEGATION: on dispatches as before; off (the default) refuses every
tool and command that would send work to a model provider, without
contacting one, and leaves Trawl, status and the other read-only tools alone.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.core import env as mullet_env
from mullet.core.env import DELEGATION_OFF_MESSAGE, delegation_enabled
from mullet.server import mcp_handlers

DISPATCHING_TOOLS = ("mullet_run", "mullet_delegate", "mullet_swarm", "mullet_opencode")


def _off(monkeypatch) -> None:
    monkeypatch.setenv("MULLET_DELEGATION", "off")


def _as_dict(result) -> dict:
    return json.loads(result) if isinstance(result, str) else result


def _no_provider_calls():
    """Patch every route to a provider so that any dispatch attempt fails the test."""
    boom = MagicMock(side_effect=AssertionError("a provider was contacted"))
    return [
        patch.object(mcp_handlers, "_resolve_provider", boom),
        patch.object(mcp_handlers, "_provider_for_tier", boom),
        patch("mullet.core.orchestration.swarm.run_swarm", boom),
        patch("mullet.jobs.delegate.delegate", boom),
    ]


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------

class TestSetting:
    def test_default_is_off(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MULLET_DELEGATION", raising=False)
        monkeypatch.setattr(mullet_env, "dotenv_files", lambda: [])
        assert delegation_enabled() is False

    @pytest.mark.parametrize("value", ["off", "OFF", " off ", "0", "false", "no", "disabled"])
    def test_off_values(self, monkeypatch, value):
        monkeypatch.setenv("MULLET_DELEGATION", value)
        assert delegation_enabled() is False

    @pytest.mark.parametrize("value", ["on", "ON", "1", "true", "yes", "enabled"])
    def test_on_values(self, monkeypatch, value):
        monkeypatch.setenv("MULLET_DELEGATION", value)
        assert delegation_enabled() is True

    def test_unrecognised_value_keeps_the_default(self, monkeypatch):
        monkeypatch.setenv("MULLET_DELEGATION", "maybe")
        assert delegation_enabled() is False

    def test_read_from_dotenv_when_not_in_environment(self, monkeypatch, tmp_path):
        """Hooks run in Claude Code's environment, where .env isn't loaded."""
        dotenv = tmp_path / ".env"
        dotenv.write_text("# comment\nOTHER=1\nMULLET_DELEGATION='off'\n")
        monkeypatch.delenv("MULLET_DELEGATION", raising=False)
        monkeypatch.setattr(mullet_env, "dotenv_files", lambda: [dotenv])
        assert delegation_enabled() is False

    def test_on_read_from_dotenv(self, monkeypatch, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("MULLET_DELEGATION=on\n")
        monkeypatch.delenv("MULLET_DELEGATION", raising=False)
        monkeypatch.setattr(mullet_env, "dotenv_files", lambda: [dotenv])
        assert delegation_enabled() is True

    def test_environment_wins_over_dotenv(self, monkeypatch, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("MULLET_DELEGATION=off\n")
        monkeypatch.setenv("MULLET_DELEGATION", "on")
        monkeypatch.setattr(mullet_env, "dotenv_files", lambda: [dotenv])
        assert delegation_enabled() is True

    def test_message_names_the_setting(self):
        assert "MULLET_DELEGATION=off" in DELEGATION_OFF_MESSAGE
        assert ".env" in DELEGATION_OFF_MESSAGE


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

_ARGS = {
    "mullet_run": {"task": "write a haiku", "tier": "LOW"},
    "mullet_delegate": {"goal": "write a haiku"},
    "mullet_swarm": {"goal": "review these", "items": ["a", "b"]},
    "mullet_opencode": {"task": "edit a file", "model": "openrouter/x"},
}


class TestMcpToolsWhenOff:
    @pytest.mark.parametrize("tool", DISPATCHING_TOOLS)
    def test_refuses_without_contacting_a_provider(self, monkeypatch, tool):
        _off(monkeypatch)
        patches = _no_provider_calls()
        for p in patches:
            p.start()
        try:
            result = _as_dict(mcp_handlers.MCP_HANDLERS[tool](dict(_ARGS[tool])))
        finally:
            for p in patches:
                p.stop()
        assert result["error"] == DELEGATION_OFF_MESSAGE
        assert result["delegation"] == "off"

    def test_run_keeps_its_return_type(self, monkeypatch):
        _off(monkeypatch)
        assert isinstance(mcp_handlers.handle_mullet_run({"task": "x"}), dict)
        assert isinstance(mcp_handlers.handle_mullet_opencode({"task": "x"}), dict)
        assert isinstance(mcp_handlers.handle_mullet_delegate({"goal": "x"}), str)
        assert isinstance(mcp_handlers.handle_mullet_swarm({"goal": "x"}), str)

    def test_run_records_no_routing_history(self, monkeypatch):
        _off(monkeypatch)
        with patch("mullet.core.routing_history.record") as record:
            mcp_handlers.handle_mullet_run({"task": "x"})
        record.assert_not_called()

    def test_read_only_tools_still_work(self, monkeypatch):
        _off(monkeypatch)
        with patch("mullet.hud.budget_monitor.check_headroom", return_value={}):
            analyze = json.loads(mcp_handlers.handle_mullet_analyze({"task": "fix a typo"}))
        assert "tier" in analyze and "error" not in analyze
        # Knowledge search goes to the local pool, not a model provider.
        km = MagicMock()
        km.search.return_value = [{"content": "hit", "source": "a.py", "score": 0.9}]
        session = MagicMock()
        session.__enter__.return_value = km
        with patch("mullet.core.knowledge.manager.knowledge_session", return_value=session):
            search = json.loads(mcp_handlers.handle_mullet_search({"query": "q", "client": "p"}))
        assert search["count"] == 1

    def test_trawl_tools_are_not_gated(self, monkeypatch):
        _off(monkeypatch)
        from mullet.server.mcp_knowledge import MCP_KNOWLEDGE_HANDLERS

        with patch("mullet.server.mcp_knowledge._get_pools_dir") as pools_dir:
            pools_dir.return_value = Path("/nonexistent-pools-dir")
            out = MCP_KNOWLEDGE_HANDLERS["trawl_list_pools"]({})
        assert DELEGATION_OFF_MESSAGE not in str(out)


class TestMcpToolsWhenOn:
    def test_run_reaches_dispatch(self, monkeypatch):
        monkeypatch.setenv("MULLET_DELEGATION", "on")
        with patch.object(mcp_handlers, "_handle_mullet_run_inner",
                          return_value={"completed": True, "summary": "ok"}) as inner:
            result = mcp_handlers.handle_mullet_run({"task": "x"})
        inner.assert_called_once()
        assert result["completed"] is True

    def test_delegate_reaches_provider_resolution(self, monkeypatch):
        monkeypatch.setenv("MULLET_DELEGATION", "on")
        with patch.object(mcp_handlers, "_active_preset", return_value=None), \
             patch.object(mcp_handlers, "_resolve_provider", return_value=None) as resolve:
            result = json.loads(mcp_handlers.handle_mullet_delegate({"goal": "x"}))
        resolve.assert_called_once()
        assert result == {"error": "No LLM provider available"}

    def test_swarm_reaches_run_swarm(self, monkeypatch):
        monkeypatch.setenv("MULLET_DELEGATION", "on")

        async def fake_run_swarm(*_a, **_k):
            raise RuntimeError("stop here")

        with patch("mullet.core.orchestration.swarm.run_swarm", side_effect=fake_run_swarm) as rs:
            out = mcp_handlers.handle_mullet_swarm({"goal": "g", "items": ["a"]})
        rs.assert_called_once()
        assert DELEGATION_OFF_MESSAGE not in out

    def test_opencode_reaches_delegate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MULLET_DELEGATION", "on")
        with patch.object(mcp_handlers, "_active_preset", return_value=None), \
             patch("mullet.jobs.ledger.Ledger", MagicMock()), \
             patch("mullet.jobs.delegate.delegate",
                   return_value={"status": "applied", "job_id": "j1"}) as run:
            result = mcp_handlers.handle_mullet_opencode({"task": "edit", "cwd": str(tmp_path)})
        run.assert_called_once()
        assert result["status"] == "applied"
        assert result.get("error") != DELEGATION_OFF_MESSAGE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_args(**overrides) -> argparse.Namespace:
    base = dict(json_output=False, verbose=False, provider=None, resume=False)
    base.update(overrides)
    return argparse.Namespace(**base)


class TestCliWhenOff:
    def test_run_task_refuses_before_building_an_agent(self, monkeypatch, capsys):
        from mullet.cli import main as cli

        _off(monkeypatch)
        with patch.object(cli, "build_agent", side_effect=AssertionError("built an agent")):
            assert cli.run_task("do something", _cli_args()) == 2
        assert DELEGATION_OFF_MESSAGE in capsys.readouterr().err

    def test_run_task_json_output(self, monkeypatch, capsys):
        from mullet.cli import main as cli

        _off(monkeypatch)
        with patch.object(cli, "build_agent", side_effect=AssertionError("built an agent")):
            assert cli.run_task("do something", _cli_args(json_output=True)) == 2
        out = json.loads(capsys.readouterr().out)
        assert out["error"] == DELEGATION_OFF_MESSAGE

    @pytest.mark.parametrize("argv", [
        ["mullet", "write", "a", "haiku"],
        ["mullet", "watch", "rerun", "tests"],
        ["mullet", "live-test", "5"],
        ["mullet"],
    ])
    def test_entry_points_exit_with_the_message(self, monkeypatch, capsys, argv):
        from mullet.cli import main as cli

        _off(monkeypatch)
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # nothing piped in
        with patch("mullet.config.logging.setup_logging"), \
             patch.object(cli, "build_agent", side_effect=AssertionError("built an agent")), \
             patch.object(cli, "resolve_provider", side_effect=AssertionError("resolved a provider")), \
             patch("mullet.cli.live_test.run_live_test", side_effect=AssertionError("live test ran")), \
             patch("mullet.cli.watch.run_watch", side_effect=AssertionError("watch ran")):
            with pytest.raises(SystemExit) as exc:
                cli._main_inner()
        assert exc.value.code == 2
        assert DELEGATION_OFF_MESSAGE in capsys.readouterr().err

    def test_non_dispatching_commands_still_work(self, monkeypatch, capsys):
        from mullet.cli import main as cli

        _off(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["mullet", "--version"])  # returns before logging set-up
        cli._main_inner()
        assert capsys.readouterr().out.startswith("mullet ")


class TestCliWhenOn:
    def test_run_task_goes_on_to_build_the_agent(self, monkeypatch):
        from mullet.cli import main as cli

        monkeypatch.setenv("MULLET_DELEGATION", "on")
        monkeypatch.setattr(cli, "_maybe_preflight", lambda *a, **k: None)
        with patch.object(cli, "build_agent", side_effect=RuntimeError("reached")) as build:
            with pytest.raises(RuntimeError, match="reached"):
                cli.run_task("do something", _cli_args())
        build.assert_called_once()


# ---------------------------------------------------------------------------
# REST API (mullet serve)
# ---------------------------------------------------------------------------

async def test_chat_completions_refused_when_off(aiohttp_client, monkeypatch):
    from aiohttp import web
    from mullet.server.api import handle_chat_completions

    _off(monkeypatch)
    app = web.Application()
    app["active_tasks"] = {}
    app["sessions"] = {}
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    client = await aiohttp_client(app)
    with patch("mullet.server.api._build_server_agent",
               side_effect=AssertionError("built an agent")), \
         patch("mullet.server.metrics.get_metrics", return_value=MagicMock()):
        resp = await client.post("/v1/chat/completions",
                                 json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status == 503
    body = await resp.json()
    assert body["error"] == DELEGATION_OFF_MESSAGE


# ---------------------------------------------------------------------------
# Hooks: no delegation nudges (or blocks) while delegation is off
# ---------------------------------------------------------------------------

class TestHookGate:
    @pytest.fixture()
    def gate(self):
        from mullet.hooks import delegation_gate

        return delegation_gate

    def test_pressure_does_not_warrant_delegation_when_off(self, gate, monkeypatch):
        _off(monkeypatch)
        warranted, reason = gate.delegation_warranted({
            "status": "pause", "pressure": 1.0, "session_pressure": 1.0,
            "weekly_pressure": 1.0, "five_hour_pct": 99.0, "weekly_pct": 99.0,
        })
        assert warranted is False
        assert "MULLET_DELEGATION=off" in reason

    def test_pressure_still_warrants_delegation_when_on(self, gate, monkeypatch):
        monkeypatch.setenv("MULLET_DELEGATION", "on")
        warranted, _ = gate.delegation_warranted({
            "status": "pause", "pressure": 1.0, "session_pressure": 1.0,
            "weekly_pressure": 1.0, "five_hour_pct": 99.0, "weekly_pct": 99.0,
        })
        assert warranted is True

    def test_falls_back_to_the_environment_without_mullet(self, gate, monkeypatch):
        monkeypatch.setitem(sys.modules, "mullet.core.env", None)  # import fails
        _off(monkeypatch)
        assert gate.delegation_turned_off() is True
        monkeypatch.setenv("MULLET_DELEGATION", "on")
        assert gate.delegation_turned_off() is False
