"""Tests for ``mullet night`` (mullet/cli/night_cli.py) and the ladder report's night line."""

from __future__ import annotations

from datetime import datetime

from mullet.cli import job_cli
from mullet.cli.night_cli import cmd_night
from mullet.jobs import night
from mullet.jobs.ladder import Ladder, parse_config
from mullet.jobs.night import NightState

# 23:00 local time, so the default end is 07:30 the next morning.
EVENING = datetime(2026, 10, 25, 23, 0).timestamp()


def run(argv, now, path):
    out: list[str] = []
    err: list[str] = []
    code = cmd_night(argv, now=now, path=path, out=out.append, err=err.append)
    return code, out, err


def test_on_status_off_status_round_trip(tmp_path):
    path = tmp_path / "night.json"

    code, out, _ = run(["on"], EVENING, path)
    assert code == 0
    assert out == ["Night mode: on until 07:30"]
    assert night.read_state(path).on

    # A minute later: too soon for the idle probe, so it stays on.
    code, out, _ = run(["status"], EVENING + 60, path)
    assert code == 0
    assert out == ["Night mode: on until 07:30"]

    code, out, _ = run(["off"], EVENING + 120, path)
    assert code == 0
    assert out == ["Night mode: off"]
    assert night.read_state(path).reason == "switched off"

    code, out, _ = run(["status"], EVENING + 180, path)
    assert code == 0
    assert out == ["Night mode: off", "Last switched off: switched off"]


def test_until_is_honoured(tmp_path):
    path = tmp_path / "night.json"
    code, out, _ = run(["on", "--until", "06:00"], EVENING, path)
    assert code == 0
    assert out == ["Night mode: on until 06:00"]
    until = night.read_state(path).until
    assert until == datetime(2026, 10, 26, 6, 0).timestamp()


def test_bad_until_exits_2(tmp_path):
    path = tmp_path / "night.json"
    for bad in ("25:99", "abc"):
        code, out, err = run(["on", "--until", bad], EVENING, path)
        assert code == 2
        assert out == []
        assert err and bad in err[0]
    assert not path.exists()


def test_until_without_injected_clock(tmp_path):
    path = tmp_path / "night.json"
    out: list[str] = []
    assert cmd_night(["on", "--until", "06:00"], path=path, out=out.append) == 0
    assert out == ["Night mode: on until 06:00"]


def test_status_after_end_time_shows_off(tmp_path):
    path = tmp_path / "night.json"
    run(["on"], EVENING, path)
    morning = datetime(2026, 10, 26, 8, 0).timestamp()
    code, out, _ = run(["status"], morning, path)
    assert code == 0
    assert out[0] == "Night mode: off"
    assert "ended at 07:30" in out[1]


def _ladder(night_fn):
    return Ladder(config=parse_config({"kimi": "moonshotai/kimi-k2.7-code"}), usage=lambda: None,
                  has_key=lambda rung: False, night=night_fn)


def test_ladder_report_night_line():
    until = datetime(2026, 10, 26, 7, 30).timestamp()
    report = job_cli.ladder_report(_ladder(lambda: NightState(on=True, until=until)), None, offline=True)
    assert "Night mode: on until 07:30" in report.splitlines()

    report = job_cli.ladder_report(_ladder(lambda: NightState(on=False, reason="you're back")), None,
                                   offline=True)
    assert "Night mode: off" in report.splitlines()


def test_ladder_report_night_line_when_night_fails():
    def broken():
        raise OSError("no state dir")

    report = job_cli.ladder_report(_ladder(broken), None, offline=True)
    assert "Night mode: off" in report.splitlines()
