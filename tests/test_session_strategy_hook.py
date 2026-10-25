"""The session-strategy SessionStart hook (mullet/hooks/session_strategy.py)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mullet.core import strategy
from mullet.core.strategy import Strategy
from mullet.hooks import HOOKS, session_strategy
from mullet.hooks.session_strategy import Facts, build_context, recommend, suggest_budget
from mullet.jobs.ledger import Ledger
from mullet.jobs.types import JobStatus, NodeSpec

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "scripts" / "hooks" / "session_strategy.py"
LOW, HIGH = 0.5, 0.8


def _facts(**overrides) -> Facts:
    base = dict(now=time.time(), ttl_hours=12.0, claude_low=LOW, claude_high=HIGH, max_spend=2.0)
    base.update(overrides)
    return Facts(**base)


def _run(monkeypatch, capsys, gather=None) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    code = session_strategy.main(gather)
    out = capsys.readouterr()
    return code, out.out, out.err


def _context(stdout: str) -> str:
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1, "the hook must print exactly one JSON object"
    data = json.loads(lines[0])
    assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    return data["hookSpecificOutput"]["additionalContext"]


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    (tmp_path / "config").mkdir()

    def write(text: str) -> None:
        (tmp_path / "config" / "config.yaml").write_text(text)

    return write


class TestRecommend:
    def test_default_is_the_ladder(self):
        assert recommend(None, None, LOW, HIGH)[0] == "ladder"
        assert recommend(10.0, 0.6, LOW, HIGH)[0] == "ladder"

    def test_claude_week_above_high_recommends_the_ladder(self):
        choice, why = recommend(None, 0.9, LOW, HIGH)
        assert choice == "ladder" and "claude_high" in why

    def test_claude_week_below_low_still_recommends_the_ladder(self):
        choice, why = recommend(None, 0.2, LOW, HIGH)
        assert choice == "ladder" and "claude_low" in why

    @pytest.mark.parametrize("left", [0.0, -1.5])
    def test_no_envelope_room_recommends_claude_only(self, left):
        assert recommend(left, 0.6, LOW, HIGH)[0] == "claude-only"
        assert recommend(left, None, LOW, HIGH)[0] == "claude-only"

    def test_no_envelope_room_and_claude_above_high_recommends_off(self):
        assert recommend(0.0, 0.95, LOW, HIGH)[0] == "off"

    def test_thresholds_come_from_the_arguments(self):
        assert recommend(0.0, 0.7, LOW, 0.6)[0] == "off"
        assert recommend(0.0, 0.7, LOW, 0.75)[0] == "claude-only"

    def test_suggested_budget(self):
        assert suggest_budget(None, 2.0) == 2.0
        assert suggest_budget(0.5, 2.0) == 0.5
        assert suggest_budget(10.0, 2.0) == 2.0
        assert suggest_budget(-3.0, 2.0) == 0.0


class TestContext:
    def test_asks_one_question_with_facts_and_a_recommendation(self, monkeypatch, capsys):
        facts = _facts(envelope={"daily": (5.0, 3.25), "weekly": (20.0, 12.0)},
                       claude_weekly_pct=42.0, claude_pressure=0.55, claude_level="medium",
                       preset="balanced-mix", night_on=False)
        code, out, _ = _run(monkeypatch, capsys, lambda: facts)
        text = _context(out)
        assert code == 0
        assert "$3.25 of $5.00 left today" in text and "$12.00 of $20.00 left this week" in text
        assert "42% used, paced pressure 0.55" in text
        assert "preset: balanced-mix" in text and "night mode: off" in text
        assert "ONE short question" in text and "same message" in text
        assert "suggesting $2.00" in text
        for choice, meaning in strategy.DELEGATION_MEANINGS.items():
            assert f"{choice} = {meaning}" in text
        assert "Recommend ladder" in text
        assert "mullet strategy set --budget <usd> --delegation <choice>" in text
        assert "already states a budget and a strategy" in text

    def test_used_up_envelope_recommends_claude_only(self):
        text = build_context(_facts(envelope={"daily": (5.0, 0.0)}, claude_pressure=0.6, claude_level="medium"))
        assert "Recommend claude-only" in text and "suggesting $0.00" in text

    def test_unknowns_are_said_so(self):
        text = build_context(_facts())
        assert "no envelope set" in text and "Claude's week: unknown" in text
        assert "preset: unknown" in text and "night mode: unknown" in text

    def test_fresh_strategy_says_not_to_ask_again(self, monkeypatch, capsys):
        now = time.time()
        facts = _facts(now=now, strategy=Strategy(4.0, "claude-only", now - 2 * 3600))
        text = _context(_run(monkeypatch, capsys, lambda: facts)[1])
        assert "already set 2.0 hours ago" in text
        assert "budget $4.00, delegation claude-only" in text
        assert "Do not ask" in text and "ONE short question" not in text

    def test_silenced(self, monkeypatch, capsys, config):
        config("session:\n  ask_at_start: false\n")
        code, out, _ = _run(monkeypatch, capsys, lambda: pytest.fail("facts gathered while silenced"))
        assert code == 0 and out == ""

    def test_fails_open(self, monkeypatch, capsys):
        def broken():
            raise RuntimeError("boom")

        code, out, err = _run(monkeypatch, capsys, broken)
        assert code == 0 and out == "" and "boom" in err


class TestGather:
    """The real facts, from a throwaway config, ledger and state directory; sockets are blocked."""

    def test_envelope_from_the_ledger(self, config):
        config("budget:\n  envelope:\n    daily: 5\n    weekly: 20\n")
        now = time.time()
        with Ledger() as ledger:
            job = ledger.create_job("/r", "g", "main", "base000", {}, max_attempts=3)
            ledger.set_plan(job, [(NodeSpec(id="n1", goal="do n1"), None)])
            ledger.set_job_status(job, JobStatus.AWAITING_APPROVAL)
            ledger.approve(job)
            ledger.set_job_status(job, JobStatus.RUNNING)
            attempt = ledger.start_attempt(job, "n1", "opencode", "base000")
            ledger.record_progress(attempt, {"rung": "kimi", "total_cost_usd": 1.5, "cost_reported": True,
                                             "model": "moonshotai/kimi-k2.7-code"})
        facts = session_strategy.gather(now)
        assert facts.envelope["daily"] == (5.0, pytest.approx(3.5))
        assert facts.envelope["weekly"] == (20.0, pytest.approx(18.5))
        assert "$3.50 of $5.00 left today" in build_context(facts)

    def test_no_envelope_and_no_ledger(self, config, tmp_path):
        facts = session_strategy.gather()
        assert facts.envelope == {} and facts.envelope_known
        assert not (tmp_path / "jobs" / "ledger.db").exists()  # the hook doesn't create one

    def test_claude_figures_and_thresholds_from_the_ladder(self, config, monkeypatch):
        config("ladder:\n  claude_low: 0.3\n  claude_high: 0.6\n  max_spend: 4\n")
        limits = SimpleNamespace(weekly_pct=70.0, weekly_resets=None, five_hour_pct=10.0, five_hour_resets=None)
        monkeypatch.setattr("mullet.jobs.ladder.current_usage", lambda: limits)
        facts = session_strategy.gather()
        assert (facts.claude_low, facts.claude_high, facts.max_spend) == (0.3, 0.6, 4.0)
        assert facts.claude_weekly_pct == 70.0 and facts.claude_pressure == pytest.approx(0.7)
        assert facts.claude_level == "high"
        assert recommend(facts.envelope_left, facts.claude_pressure, 0.3, 0.6)[0] == "ladder"

    def test_fresh_strategy_is_read(self, config):
        strategy.save(Strategy(1.0, "ladder", time.time()))
        assert session_strategy.gather().strategy is not None


def test_registered_as_a_session_start_hook():
    spec = HOOKS["session-strategy"]
    assert spec.module == "session_strategy"
    assert spec.registrations == (("SessionStart", None),)
    settings_file = ROOT / ".claude" / "settings.json"
    if not settings_file.exists():
        pytest.skip("this repo's own Claude Code settings are not part of the public copy")
    settings = json.loads(settings_file.read_text())
    commands = [h["command"] for group in settings["hooks"]["SessionStart"] for h in group["hooks"]]
    assert any(c.endswith("scripts/hooks/session_strategy.py") for c in commands)


def test_the_shim_prints_one_json_object(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MULLET_CONFIG_DIR": str(tmp_path / "config"),
           "MULLET_JOBS_DIR": str(tmp_path / "jobs")}
    result = subprocess.run([sys.executable, str(SHIM)], input="{}", capture_output=True, text=True,
                            env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "ONE short question" in _context(result.stdout)
