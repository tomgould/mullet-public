"""The credit and Claude status line, the doctor's spending checks and `mullet job ladder`'s credit lines."""
from __future__ import annotations

import argparse
from datetime import datetime

import pytest

from mullet.cli import doctor_checks, job_cli
from mullet.jobs import notify
from mullet.jobs.budget import BudgetConfig, Limits
from mullet.jobs.ladder import Ladder, parse_config
from mullet.jobs.notify import message, notify_job, status_line

WEBHOOK = "https://hooks.slack.com/services/T000/B000/fake"
TUESDAY_10AM = datetime(2026, 10, 27, 10, 0)
SNAP = {"kimi": {"used_usd": None, "remaining_usd": 100.5}, "openrouter": {"used_usd": 41.2, "remaining_usd": None}}


class TestStatusLine:
    def test_credit_and_claude(self):
        assert status_line(SNAP, 38.0) == "Credit: kimi $100.50 left, openrouter $41.20 used; Claude week 38% used"

    def test_nothing_to_say(self):
        assert status_line(None, None) == ""
        assert status_line({}, None) == ""
        assert status_line({"kimi": {"used_usd": None, "remaining_usd": None}}, None) == ""

    def test_parts_are_omitted_independently(self):
        assert status_line(None, 12.4) == "Claude week 12% used"
        assert status_line({"kimi": {"remaining_usd": 3}}, None) == "Credit: kimi $3.00 left"

    def test_remaining_wins_over_used(self):
        assert status_line({"kimi": {"used_usd": 9.0, "remaining_usd": 1.0}}, None) == "Credit: kimi $1.00 left"


@pytest.fixture
def calls(monkeypatch):
    record = {"run": [], "post": []}
    monkeypatch.setattr(notify.sys, "platform", "linux")
    monkeypatch.setattr(notify.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    record["kw"] = dict(run=lambda cmd, **kw: record["run"].append(cmd),
                        post=lambda url, payload: record["post"].append((url, payload)))
    return record


class TestNotifyStatus:
    @pytest.mark.parametrize("event", notify.EVENTS)
    def test_status_is_added_to_both_bodies(self, event):
        line = status_line(SNAP, 38)
        for local in (True, False):
            _, body = message(event, "j1", "Secret goal", "why", local=local, status=line)
            assert body.endswith("\n" + line)

    def test_slack_text_has_status_but_no_goal(self, calls):
        line = status_line(SNAP, 38)
        env = {"MULLET_NOTIFY": "desktop,slack", "MULLET_SLACK_WEBHOOK": WEBHOOK}
        sent = notify_job("finished", "j1", "Rewrite the secret thing", "detail", env=env,
                          now=TUESDAY_10AM, status=line, **calls["kw"])
        assert sent == ["desktop", "slack"]
        text = calls["post"][0][1]["text"]
        assert line in text
        assert "secret" not in text.lower() and "detail" not in text
        assert line in calls["run"][0][-1]

    def test_without_status_nothing_changes(self):
        assert message("finished", "j1", "g", "d") == message("finished", "j1", "g", "d", status="")


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    return tmp_path / "config.yaml"


class TestDoctor:
    def test_no_limits_warns(self, config_file):
        row = doctor_checks.spending_limit_check()
        assert row and row[2] == "warn" and "budget.envelope.daily/weekly" in row[1]

    def test_limits_set_is_silent(self, config_file):
        config_file.write_text("budget:\n  envelope:\n    daily: 10\n")
        assert doctor_checks.spending_limit_check() is None
        assert doctor_checks.spending_limit_check(lambda: BudgetConfig(envelope=Limits(weekly=1.0))) is None

    def test_threshold_setting(self, config_file):
        assert doctor_checks.warn_below_usd() == 5.0
        config_file.write_text("budget:\n  warn_below_usd: 12.5\n")
        assert doctor_checks.warn_below_usd() == 12.5
        config_file.write_text("budget:\n  warn_below_usd: nope\n")
        assert doctor_checks.warn_below_usd() == 5.0

    def test_low_credit_warns_and_healthy_is_ok(self, config_file):
        def probe():
            return {"kimi": {"remaining_usd": 2.0, "used_usd": None},
                    "openrouter": {"remaining_usd": 40.0, "used_usd": None}}
        rows = {r[0]: r for r in doctor_checks.credit_checks(probe, providers=["kimi", "openrouter"])}
        assert rows["kimi credit"][2] == "warn" and "budget.warn_below_usd" in rows["kimi credit"][1]
        assert rows["openrouter credit"][2] == "ok"

    def test_threshold_is_configurable(self, config_file):
        config_file.write_text("budget:\n  warn_below_usd: 50\n")
        rows = doctor_checks.credit_checks(lambda: {"kimi": {"remaining_usd": 40.0}}, providers=["kimi"])
        assert rows[0][2] == "warn"

    def test_usage_only_and_failed_probe_are_info(self, config_file):
        rows = doctor_checks.credit_checks(lambda: {"kimi": {"used_usd": 3.0, "remaining_usd": None}},
                                           providers=["kimi", "openrouter"])
        assert [r[2] for r in rows] == ["info", "info"]

    def test_no_keyed_provider_means_no_probe(self, config_file):
        def boom():
            raise AssertionError("probed")
        assert doctor_checks.credit_checks(boom, providers=[]) == []

    def test_offline_doctor_does_not_probe(self, config_file, monkeypatch):
        monkeypatch.setattr(doctor_checks, "credit_checks", lambda *a, **k: pytest.fail("probed"))
        rows = doctor_checks.spending_checks(full=False)
        assert [r[0] for r in rows] == ["Spending limits"]

    def test_full_doctor_adds_credit(self, config_file, monkeypatch):
        monkeypatch.setattr(doctor_checks, "credit_checks", lambda probe=None, **k: [("kimi credit", "x", "ok")])
        assert [r[0] for r in doctor_checks.spending_checks(full=True)] == ["Spending limits", "kimi credit"]


def _ladder(has_key: bool) -> Ladder:
    return Ladder(config=parse_config({"kimi": "moonshotai/kimi-k2.7-code"}), usage=lambda: None,
                  has_key=lambda rung: has_key)


class TestLadderCredit:
    def test_credit_line_and_low_warning(self, config_file):
        lines = job_cli.credit_lines(_ladder(True), lambda: {"kimi": {"remaining_usd": 2.0},
                                                             "openrouter": {"remaining_usd": 90.0}})
        assert lines[0] == "Credit: kimi $2.00 left, openrouter $90.00 left"
        assert lines[1].startswith("Warning: kimi has $2.00 left, under $5")

    def test_healthy_balance_has_no_warning(self, config_file):
        lines = job_cli.credit_lines(_ladder(True), lambda: {"kimi": {"remaining_usd": 90.0}})
        assert lines == ["Credit: kimi $90.00 left"]

    def test_no_key_no_probe(self, config_file):
        assert job_cli.credit_lines(_ladder(False), lambda: pytest.fail("probed")) == []

    def test_cmd_ladder_prints_credit_after_the_table(self, config_file, monkeypatch, capsys):
        monkeypatch.setattr(job_cli, "make_ladder", lambda: _ladder(True))
        monkeypatch.setattr(job_cli, "probe_credit", lambda: {"kimi": {"remaining_usd": 1.0}})
        job_cli.cmd_ladder(argparse.Namespace(repo=".", offline=False), None)
        out = capsys.readouterr().out
        assert out.index("Rungs:") < out.index("Credit: kimi $1.00 left") < out.index("Warning:")
