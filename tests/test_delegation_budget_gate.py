"""The delegation hooks must stay quiet unless the Anthropic budget is short.

Delegating costs real third-party tokens; in-context work is already paid for
by the subscription. These pin that rule, and the ask-first requirement.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# The checkout's hook shims, run as Claude Code would; the code is in mullet/hooks/.
HOOKS = Path(__file__).resolve().parent.parent / "scripts" / "hooks"

from mullet.hooks import delegation_gate as gate  # noqa: E402

DELEGABLE_PROMPT = json.dumps({"prompt": "please review and analyse this plan"})


def _snap(**overrides) -> dict:
    base = {
        "status": "go",
        "pressure": 0.30,
        "session_pressure": 0.30,
        "weekly_pressure": 0.30,
        "five_hour_pct": 25.0,
        "weekly_pct": 50.0,
    }
    base.update(overrides)
    return base


class TestGate:
    def test_comfortable_budget_is_not_warranted(self) -> None:
        warranted, reason = gate.delegation_warranted(_snap())
        assert warranted is False
        assert "comfortable" in reason

    def test_unavailable_figures_fail_closed(self, monkeypatch) -> None:
        monkeypatch.setattr(gate, "pressure_snapshot", lambda: None)
        warranted, reason = gate.delegation_warranted()
        assert warranted is False, "a failed lookup must not nudge toward a paid provider"
        assert "unavailable" in reason

    def test_session_pressure_at_threshold_warrants(self) -> None:
        warranted, reason = gate.delegation_warranted(
            _snap(session_pressure=gate.SESSION_PRESSURE_THRESHOLD)
        )
        assert warranted is True
        assert "5-hour pressure" in reason

    def test_just_below_session_threshold_does_not_warrant(self) -> None:
        warranted, _ = gate.delegation_warranted(
            _snap(session_pressure=gate.SESSION_PRESSURE_THRESHOLD - 0.01)
        )
        assert warranted is False

    def test_weekly_pressure_warrants(self) -> None:
        warranted, reason = gate.delegation_warranted(
            _snap(weekly_pressure=gate.WEEKLY_PRESSURE_THRESHOLD)
        )
        assert warranted is True
        assert "weekly" in reason

    def test_raw_weekly_pct_danger_warrants_regardless_of_pacing(self) -> None:
        # Pacing can look calm while the weekly window is nearly spent.
        warranted, reason = gate.delegation_warranted(
            _snap(weekly_pct=gate.WEEKLY_PCT_DANGER, weekly_pressure=0.10)
        )
        assert warranted is True
        assert "danger line" in reason

    def test_pause_status_warrants(self) -> None:
        warranted, reason = gate.delegation_warranted(_snap(status="pause"))
        assert warranted is True
        assert "PAUSE" in reason

    def test_snapshot_never_raises(self, monkeypatch) -> None:
        def boom():
            raise RuntimeError("usage API down")

        monkeypatch.setitem(sys.modules, "mullet.hud.guard", type(sys)("mullet.hud.guard"))
        sys.modules["mullet.hud.guard"].check_budget = boom
        assert gate.pressure_snapshot() is None


def _run_hook(script: str, stdin: str, stub: str | None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    if stub:
        env["PYTHONPATH"] = stub
    return subprocess.run(
        [sys.executable, str(HOOKS / script)],
        input=stdin, capture_output=True, text=True, env=env,
    )


@pytest.fixture()
def stub_budget(tmp_path):
    def _make(**fields) -> str:
        vals = {
            "status": "go", "pressure": 0.3, "session_pressure": 0.3,
            "weekly_pressure": 0.3, "five_hour_pct": 25.0, "weekly_pct": 50.0,
        }
        vals.update(fields)
        (tmp_path / "sitecustomize.py").write_text(
            "import sys, types\n"
            "m = types.ModuleType('mullet.hud.guard')\n"
            "class _S:\n"
            f"    value = {vals['status']!r}\n"
            "class _R:\n"
            "    status = _S()\n"
            f"    pressure = {vals['pressure']}\n"
            f"    session_pressure = {vals['session_pressure']}\n"
            f"    weekly_pressure = {vals['weekly_pressure']}\n"
            f"    five_hour_pct = {vals['five_hour_pct']}\n"
            f"    weekly_pct = {vals['weekly_pct']}\n"
            "m.check_budget = lambda: _R()\n"
            "sys.modules['mullet.hud.guard'] = m\n"
        )
        return str(tmp_path)
    return _make


class TestReminderHook:
    def test_silent_when_budget_is_comfortable(self, stub_budget) -> None:
        r = _run_hook("delegation_reminder.py", DELEGABLE_PROMPT, stub_budget())
        assert r.stdout.strip() == "", "must not suggest a paid delegation with headroom left"
        assert r.returncode == 0

    def test_fires_under_weekly_danger(self, stub_budget) -> None:
        r = _run_hook("delegation_reminder.py", DELEGABLE_PROMPT,
                      stub_budget(weekly_pct=90.0))
        assert "budget pressure" in r.stdout
        assert "danger line" in r.stdout
        assert r.returncode == 0

    def test_fires_under_session_pressure(self, stub_budget) -> None:
        r = _run_hook("delegation_reminder.py", DELEGABLE_PROMPT,
                      stub_budget(session_pressure=0.85))
        assert "5-hour pressure" in r.stdout

    def test_always_demands_confirmation_when_it_fires(self, stub_budget) -> None:
        r = _run_hook("delegation_reminder.py", DELEGABLE_PROMPT,
                      stub_budget(weekly_pct=95.0))
        assert "ASK FIRST" in r.stdout
        assert "wait" in r.stdout

    def test_mullet_keyword_alone_does_not_fire_with_headroom(self, stub_budget) -> None:
        # "mullet" used to force a reminder regardless of anything else.
        r = _run_hook("delegation_reminder.py",
                      json.dumps({"prompt": "what does mullet do?"}), stub_budget())
        assert r.stdout.strip() == ""

    def test_never_blocks_the_prompt(self, stub_budget) -> None:
        for pct in (10.0, 99.0):
            r = _run_hook("delegation_reminder.py", DELEGABLE_PROMPT,
                          stub_budget(weekly_pct=pct))
            assert r.returncode == 0
