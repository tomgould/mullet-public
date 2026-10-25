"""Tests for mullet.jobs.night.

Every outside call is injected: the clock, the idle probe and the state path.
No real xprintidle, loginctl or Mullet state directory is touched.
"""

from datetime import datetime

import pytest

from mullet.jobs import night


@pytest.fixture
def state(tmp_path):
    return tmp_path / "night.json"


# State round trip


def test_read_state_is_off_when_the_file_is_missing(state):
    assert night.read_state(state) == night.NightState()


def test_on_off_read_round_trip(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state, who="tester")
    s = night.read_state(state)
    assert s.on is True
    assert s.until == night.default_until(now)
    assert s.reason is None

    night.switch_off("manual", now=now + 1, path=state)
    s = night.read_state(state)
    assert s.on is False
    assert s.reason == "manual"


def test_switch_on_uses_configured_until(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    until = datetime(2024, 1, 15, 12, 0, 0).timestamp()
    night.switch_on(until=until, now=now, path=state)
    assert night.read_state(state).until == until


# Default and parsed until times


def test_default_until_before_0730_is_today():
    now = datetime(2024, 1, 15, 5, 0, 0).timestamp()
    assert night.default_until(now) == datetime(2024, 1, 15, 7, 30, 0).timestamp()


def test_default_until_after_0730_is_tomorrow():
    now = datetime(2024, 1, 15, 9, 0, 0).timestamp()
    assert night.default_until(now) == datetime(2024, 1, 16, 7, 30, 0).timestamp()


def test_default_until_at_0730_is_tomorrow():
    now = datetime(2024, 1, 15, 7, 30, 0).timestamp()
    assert night.default_until(now) == datetime(2024, 1, 16, 7, 30, 0).timestamp()


def test_parse_until_returns_the_next_occurrence():
    now = datetime(2024, 1, 15, 10, 0, 0).timestamp()
    assert night.parse_until("15:00", now) == datetime(2024, 1, 15, 15, 0, 0).timestamp()
    assert night.parse_until("09:00", now) == datetime(2024, 1, 16, 9, 0, 0).timestamp()


@pytest.mark.parametrize("value", ["25:00", "not-a-time", "07:30:00", ""])
def test_parse_until_rejects_bad_values(value):
    now = datetime(2024, 1, 15, 10, 0, 0).timestamp()
    with pytest.raises(ValueError):
        night.parse_until(value, now)


# is_on expiry


def test_is_on_is_true_before_until(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    until = datetime(2024, 1, 15, 7, 30, 0).timestamp()
    night.switch_on(until=until, now=now, path=state)
    assert night.is_on(now=now + 1, path=state) is True


def test_is_on_turns_off_past_until_and_records_the_time(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    until = datetime(2024, 1, 15, 7, 30, 0).timestamp()
    night.switch_on(until=until, now=now, path=state)
    assert night.is_on(now=datetime(2024, 1, 15, 7, 31, 0).timestamp(), path=state) is False
    s = night.read_state(state)
    assert s.on is False
    assert s.reason == "ended at 07:30"


# user_is_back with an injected idle time


def test_user_is_back_after_ten_minutes_with_xprintidle_under_two_minutes(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    assert night.user_is_back(now=now + 600, path=state, idle_ms=119_999) is True


def test_user_is_back_is_not_true_within_the_first_ten_minutes(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    assert night.user_is_back(now=now + 599, path=state, idle_ms=1_000) is False


def test_user_is_back_is_not_true_when_idle_over_two_minutes(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    assert night.user_is_back(now=now + 600, path=state, idle_ms=120_001) is False


def test_user_is_back_is_false_when_night_mode_is_off(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    assert night.user_is_back(now=now + 600, path=state, idle_ms=1_000) is False


def test_user_is_back_is_false_without_any_idle_tool(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)

    def no_tool(name):
        return None

    def broken_run(*args, **kwargs):
        raise RuntimeError("no subprocess in tests")

    assert night.user_is_back(
        now=now + 600,
        path=state,
        which=no_tool,
        run=broken_run,
    ) is False


def test_user_is_back_survives_a_raising_probe(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)

    def broken_tool(name):
        raise RuntimeError("probe failed")

    assert night.user_is_back(now=now + 600, path=state, which=broken_tool) is False


# loginctl idle probe


def _make_loginctl_run(active: bool, session_type: str = "wayland", locked: bool = False):
    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 0
            stdout = ""
        if cmd[1] == "list-sessions":
            Result.stdout = "c1 1000 tester seat0 tty1\n"
            return Result()
        if cmd[1] == "show-session":
            idle = "no" if active else "yes"
            lock = "yes" if locked else "no"
            Result.stdout = f"IdleHint={idle}\nType={session_type}\nLockedHint={lock}\n"
            return Result()
        raise RuntimeError(cmd)
    return fake_run


def _which_loginctl_only(name):
    return None if name == "xprintidle" else "/usr/bin/loginctl"


@pytest.fixture
def loginctl_session(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_ID", "c1")


def test_loginctl_active_graphical_unlocked_session_means_back(state, loginctl_session):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    assert night.user_is_back(
        now=now + 600,
        path=state,
        which=_which_loginctl_only,
        run=_make_loginctl_run(active=True, session_type="wayland", locked=False),
    ) is True


def test_loginctl_x11_session_means_back(state, loginctl_session):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    assert night.user_is_back(
        now=now + 600,
        path=state,
        which=_which_loginctl_only,
        run=_make_loginctl_run(active=True, session_type="x11", locked=False),
    ) is True


def test_loginctl_locked_session_is_not_back(state, loginctl_session):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    assert night.user_is_back(
        now=now + 600,
        path=state,
        which=_which_loginctl_only,
        run=_make_loginctl_run(active=True, session_type="wayland", locked=True),
    ) is False


def test_loginctl_tty_session_is_not_back(state, loginctl_session):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    assert night.user_is_back(
        now=now + 600,
        path=state,
        which=_which_loginctl_only,
        run=_make_loginctl_run(active=True, session_type="tty", locked=False),
    ) is False


def test_loginctl_idle_session_is_not_back(state, loginctl_session):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    assert night.user_is_back(
        now=now + 600,
        path=state,
        which=_which_loginctl_only,
        run=_make_loginctl_run(active=False, session_type="wayland", locked=False),
    ) is False


# check


def test_check_switches_off_when_user_is_back(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    s = night.check(now=now + 600, path=state, back=True)
    assert s.on is False
    assert s.reason == "you're back"


def test_check_keeps_night_on_when_user_is_not_back(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    night.switch_on(now=now, path=state)
    s = night.check(now=now + 600, path=state, back=False)
    assert s.on is True
    assert s.reason is None


def test_check_returns_off_state_after_expiry(state):
    now = datetime(2024, 1, 15, 0, 0, 0).timestamp()
    until = datetime(2024, 1, 15, 7, 30, 0).timestamp()
    night.switch_on(until=until, now=now, path=state)
    s = night.check(now=datetime(2024, 1, 15, 8, 0, 0).timestamp(), path=state)
    assert s.on is False
    assert s.reason == "ended at 07:30"


# Settings


def test_night_settings_defaults():
    assert night.SETTINGS.cheap_minutes_factor == 2.0
    assert night.SETTINGS.extra_below_claude == 2
    assert night.SETTINGS.local_headroom_wait_s == 1800.0
