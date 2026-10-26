"""The session strategy changes what Mullet does: `mullet job plan`, and the delegation switch."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from mullet.cli.job_cli import cmd_job
from mullet.core import env as mullet_env
from mullet.core import strategy
from mullet.core.env import DELEGATION_OFF_MESSAGE, delegation_enabled, delegation_off_message
from mullet.core.strategy import STRATEGY_OFF_MESSAGE, Strategy
from mullet.hooks import delegation_gate
from mullet.jobs.ledger import Ledger
from mullet.server import mcp_handlers

from tests.test_jobs_cli import _budget, _job_id, plan, store_profile
from tests.test_jobs_cli import ladder_user, repo  # noqa: F401  (fixtures)

DISPATCHING_TOOLS = ("mullet_run", "mullet_delegate", "mullet_swarm", "mullet_opencode")
_ARGS = {
    "mullet_run": {"task": "write a haiku", "tier": "LOW"},
    "mullet_delegate": {"goal": "write a haiku"},
    "mullet_swarm": {"goal": "review these", "items": ["a", "b"]},
    "mullet_opencode": {"task": "edit a file", "model": "openrouter/x"},
}


def _set(budget: float | None, delegation: str, age_s: float = 0.0) -> None:
    strategy.save(Strategy(budget, delegation, time.time() - age_s))


@pytest.fixture(autouse=True)
def switch_on(monkeypatch):
    """MULLET_DELEGATION is on, so only the strategy can turn delegation off."""
    monkeypatch.setenv("MULLET_DELEGATION", "on")
    monkeypatch.setattr(mullet_env, "dotenv_files", lambda: [])


# ---------------------------------------------------------------------------
# mullet job plan
# ---------------------------------------------------------------------------

class TestJobPlan:
    def test_budget_becomes_the_cap(self, ladder_user, capsys):  # noqa: F811
        _set(0.4, "ladder")
        assert cmd_job(["plan", "x", "--repo", str(ladder_user), "--single", "--no-review"]) == 0
        out = capsys.readouterr().out
        assert "Session strategy applied: spending cap $0.40" in out
        budget = _budget(_job_id(out))
        assert budget["max_spend"] == 0.4 and budget["ladder"] is True

    def test_explicit_max_spend_wins(self, ladder_user, capsys):  # noqa: F811
        _set(0.4, "ladder")
        assert _budget(plan(ladder_user, capsys, "--max-spend", "0.9"))["max_spend"] == 0.9

    def test_claude_only_plans_one_claude_worker(self, ladder_user, capsys):  # noqa: F811
        _set(None, "claude-only")
        assert cmd_job(["plan", "x", "--repo", str(ladder_user), "--single", "--no-review"]) == 0
        out = capsys.readouterr().out
        assert "Session strategy applied: claude-only" in out
        budget = _budget(_job_id(out))
        assert budget["ladder"] is False and budget["executor"] == "claude-code"
        assert budget["max_spend"] == 1.5  # no budget in the strategy: ladder.max_spend

    @pytest.mark.parametrize("flags", [["--ladder"], ["--ladder", "--executor", "claude"]])
    def test_explicit_ladder_flags_win_over_claude_only(self, ladder_user, capsys, flags):  # noqa: F811
        _set(None, "claude-only")
        assert _budget(plan(ladder_user, capsys, *flags))["ladder"] is True

    def test_off_refuses_and_names_the_command(self, ladder_user, capsys):  # noqa: F811
        _set(2.0, "off")
        assert cmd_job(["plan", "x", "--repo", str(ladder_user), "--single"]) == 1
        assert "mullet strategy set --delegation ladder" in capsys.readouterr().err
        with Ledger() as ledger:
            assert ledger.list_jobs() == []

    def test_an_expired_strategy_changes_nothing(self, ladder_user, capsys):  # noqa: F811
        _set(0.4, "off", age_s=13 * 3600)
        out_budget = _budget(plan(ladder_user, capsys))
        assert out_budget["max_spend"] == 1.5 and out_budget["ladder"] is True

    def test_an_unreadable_strategy_changes_nothing(self, repo, capsys, monkeypatch):  # noqa: F811
        store_profile(repo)
        monkeypatch.setattr(strategy, "load", MagicMock(side_effect=RuntimeError("boom")))
        assert _budget(plan(repo, capsys))["ladder"] is False
        assert "Session strategy applied" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The delegation switch, the hooks' gate and the MCP tools
# ---------------------------------------------------------------------------

class TestDelegationOff:
    def test_off_strategy_turns_delegation_off(self):
        assert delegation_enabled() is True
        _set(None, "off")
        assert delegation_enabled() is False
        assert delegation_off_message() == STRATEGY_OFF_MESSAGE
        assert "mullet strategy set --delegation ladder" in STRATEGY_OFF_MESSAGE

    @pytest.mark.parametrize("choice", ["ladder", "claude-only"])
    def test_other_choices_leave_it_on(self, choice):
        _set(None, choice)
        assert delegation_enabled() is True

    def test_expired_off_strategy_leaves_it_on(self):
        _set(None, "off", age_s=13 * 3600)
        assert delegation_enabled() is True

    def test_the_switch_message_wins_when_both_are_off(self, monkeypatch):
        _set(None, "off")
        monkeypatch.setenv("MULLET_DELEGATION", "off")
        assert delegation_off_message() == DELEGATION_OFF_MESSAGE

    def test_a_broken_strategy_fails_open(self, monkeypatch):
        monkeypatch.setattr(strategy, "delegation_off", MagicMock(side_effect=RuntimeError("boom")))
        assert delegation_enabled() is True

    def test_the_gate_stays_quiet_and_names_the_strategy(self):
        _set(None, "off")
        assert delegation_gate.delegation_turned_off() is True
        warranted, reason = delegation_gate.delegation_warranted(
            {"status": "pause", "pressure": 1, "session_pressure": 1, "weekly_pressure": 1,
             "five_hour_pct": 99, "weekly_pct": 99})
        assert warranted is False and "mullet strategy" in reason

    @pytest.mark.parametrize("tool", DISPATCHING_TOOLS)
    def test_mcp_tools_refuse(self, tool):
        _set(None, "off")
        boom = MagicMock(side_effect=AssertionError("a provider was contacted"))
        with patch.object(mcp_handlers, "_resolve_provider", boom), \
                patch.object(mcp_handlers, "_provider_for_tier", boom), \
                patch("mullet.core.orchestration.swarm.run_swarm", boom), \
                patch("mullet.jobs.delegate.delegate", boom):
            result = mcp_handlers.MCP_HANDLERS[tool](dict(_ARGS[tool]))
        result = json.loads(result) if isinstance(result, str) else result
        assert result["error"] == STRATEGY_OFF_MESSAGE
        assert result["delegation"] == "off"
