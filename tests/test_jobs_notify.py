"""Notifications when a job blocks, pauses or finishes."""
from datetime import datetime

import pytest

from mullet.jobs import notify
from mullet.jobs.notify import message, notify_job, within_hours

WEBHOOK = "https://hooks.slack.com/services/T000/B000/fake"
TUESDAY_10AM = datetime(2026, 10, 27, 10, 0)
TUESDAY_8PM = datetime(2026, 10, 27, 20, 0)
SATURDAY_10AM = datetime(2026, 10, 24, 10, 0)


@pytest.fixture
def calls(monkeypatch):
    """Record desktop and Slack sends; pretend a Linux desktop session exists."""
    record = {"run": [], "post": []}
    monkeypatch.setattr(notify.sys, "platform", "linux")
    monkeypatch.setattr(notify.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    record["kw"] = dict(
        run=lambda cmd, **kw: record["run"].append(cmd),
        post=lambda url, payload: record["post"].append((url, payload)),
    )
    return record


class TestWithinHours:
    @pytest.mark.parametrize("now, expected", [
        (TUESDAY_10AM, True),
        (datetime(2026, 10, 27, 9, 0), True),
        (datetime(2026, 10, 27, 17, 29), True),
        (datetime(2026, 10, 27, 17, 30), False),
        (datetime(2026, 10, 27, 8, 59), False),
        (TUESDAY_8PM, False),
        (SATURDAY_10AM, False),
    ])
    def test_office_hours(self, now, expected):
        assert within_hours("Mon-Fri 09:00-17:30", now) is expected

    def test_always(self):
        assert within_hours("always", SATURDAY_10AM)
        assert within_hours("", TUESDAY_8PM)

    def test_a_day_range_can_wrap_the_weekend(self):
        assert within_hours("Sat-Mon 00:00-23:59", SATURDAY_10AM)
        assert not within_hours("Sat-Mon 00:00-23:59", TUESDAY_10AM)

    @pytest.mark.parametrize("spec", ["Funday 09:00-17:00", "Mon-Fri 9am-5pm", "Mon-Fri"])
    def test_unreadable_specs_raise(self, spec):
        with pytest.raises(ValueError):
            within_hours(spec, TUESDAY_10AM)


class TestMessage:
    def test_blocked_says_what_to_run(self):
        title, body = message("blocked", "j1", "Refactor the parser", "node 2 failed G5")
        assert title == "Mullet job j1 blocked"
        assert "Refactor the parser" in body
        assert "node 2 failed G5" in body
        assert "mullet job report j1" in body

    def test_finished_says_how_to_merge(self):
        assert "mullet job merge j1" in message("finished", "j1", "goal", "passed")[1]

    def test_a_long_goal_is_shortened(self):
        body = message("paused", "j1", "x" * 200, "14:00")[1]
        assert body.splitlines()[0].endswith("...")
        assert len(body.splitlines()[0]) == 80

    def test_only_the_goals_first_line_is_shown_locally(self):
        body = message("finished", "j1", "Tidy the parser\nwith a secret plan below", "2 of 2 steps passed")[1]
        assert body.splitlines()[0] == "Tidy the parser" and "secret plan" not in body

    def test_messages_that_leave_the_machine_are_fixed_wording_only(self):
        for event, detail in (("blocked", "G5 failed: test_secret_thing"), ("finished", "1 of 2"),
                              ("paused", "14:00"), ("spend", "detail-text")):
            title, body = message(event, "j1", "Rewrite billing for ACME", detail, local=False)
            assert "ACME" not in body and detail not in body and "j1" in title
        assert "--max-spend" in message("spend", "j1", "g", local=False)[1]

    def test_unknown_event(self):
        with pytest.raises(ValueError):
            message("exploded", "j1", "goal")


class TestNotifyJob:
    def test_desktop_by_default(self, calls):
        sent = notify_job("blocked", "j1", "goal", env={}, **calls["kw"])
        assert sent == ["desktop"]
        cmd = calls["run"][0]
        assert cmd[:2] == ["notify-send", "--app-name=Mullet"]
        assert cmd[2] == "Mullet job j1 blocked"

    def test_off_sends_nothing(self, calls):
        assert notify_job("blocked", "j1", "goal", env={"MULLET_NOTIFY": "off"}, **calls["kw"]) == []
        assert calls["run"] == [] and calls["post"] == []

    def test_slack_inside_working_hours(self, calls):
        env = {"MULLET_NOTIFY": "slack", "MULLET_SLACK_WEBHOOK": WEBHOOK}
        sent = notify_job("finished", "j1", "goal", "passed", env=env, now=TUESDAY_10AM, **calls["kw"])
        assert sent == ["slack"]
        url, payload = calls["post"][0]
        assert url == WEBHOOK
        assert payload["text"].startswith("Mullet job j1 finished")
        assert "goal" not in payload["text"] and "passed" not in payload["text"]
        assert calls["run"] == []

    @pytest.mark.parametrize("now", [TUESDAY_8PM, SATURDAY_10AM])
    def test_no_slack_outside_working_hours(self, calls, now):
        env = {"MULLET_NOTIFY": "desktop,slack", "MULLET_SLACK_WEBHOOK": WEBHOOK}
        assert notify_job("blocked", "j1", "goal", env=env, now=now, **calls["kw"]) == ["desktop"]
        assert calls["post"] == []

    def test_custom_hours(self, calls):
        env = {"MULLET_NOTIFY": "slack", "MULLET_SLACK_WEBHOOK": WEBHOOK, "MULLET_NOTIFY_HOURS": "always"}
        assert notify_job("blocked", "j1", "goal", env=env, now=SATURDAY_10AM, **calls["kw"]) == ["slack"]

    def test_bad_hours_mean_no_slack(self, calls):
        env = {"MULLET_NOTIFY": "slack", "MULLET_SLACK_WEBHOOK": WEBHOOK, "MULLET_NOTIFY_HOURS": "whenever"}
        assert notify_job("blocked", "j1", "goal", env=env, now=TUESDAY_10AM, **calls["kw"]) == []

    def test_only_slack_webhooks_are_posted_to(self, calls):
        env = {"MULLET_NOTIFY": "slack", "MULLET_SLACK_WEBHOOK": "https://example.com/hook"}
        assert notify_job("blocked", "j1", "goal", env=env, now=TUESDAY_10AM, **calls["kw"]) == []
        assert calls["post"] == []

    def test_no_desktop_session_means_no_desktop(self, calls, monkeypatch):
        for var in ("DBUS_SESSION_BUS_ADDRESS", "DISPLAY", "WAYLAND_DISPLAY"):
            monkeypatch.delenv(var, raising=False)
        assert notify_job("blocked", "j1", "goal", env={}, **calls["kw"]) == []

    def test_macos_uses_osascript(self, calls, monkeypatch):
        monkeypatch.setattr(notify.sys, "platform", "darwin")
        notify_job("finished", "j1", 'say "hi" café', env={}, **calls["kw"])
        cmd = calls["run"][0]
        assert cmd[:2] == ["osascript", "-e"]
        assert 'with title "Mullet job j1 finished"' in cmd[2]
        assert '\\"hi\\" café' in cmd[2]

    def test_failures_never_raise(self, calls):
        def boom(*args, **kwargs):
            raise OSError("no dbus")

        env = {"MULLET_NOTIFY": "desktop,slack", "MULLET_SLACK_WEBHOOK": WEBHOOK}
        sent = notify_job("blocked", "j1", "goal", env=env, now=TUESDAY_10AM, run=boom, post=boom)
        assert sent == []

    def test_settings_come_from_the_dotenv_when_not_in_the_environment(self, calls, monkeypatch):
        monkeypatch.delenv("MULLET_NOTIFY", raising=False)
        values = {"MULLET_NOTIFY": "off"}
        monkeypatch.setattr("mullet.core.env.dotenv_value", values.get)
        assert notify_job("blocked", "j1", "goal", **calls["kw"]) == []
        assert calls["run"] == []


class TestNotify:
    def test_sends_given_title_and_body_on_desktop(self, calls):
        sent = notify.notify("Local LLM restarted", "not answering", env={}, **calls["kw"])
        assert sent == ["desktop"]
        cmd = calls["run"][0]
        assert cmd[:2] == ["notify-send", "--app-name=Mullet"]
        assert cmd[2] == "Local LLM restarted"
        assert cmd[3] == "not answering"

    def test_off_sends_nothing(self, calls):
        assert notify.notify("title", "body", env={"MULLET_NOTIFY": "off"}, **calls["kw"]) == []
        assert calls["run"] == [] and calls["post"] == []

    def test_failures_never_raise(self, calls):
        def boom(*args, **kwargs):
            raise OSError("no dbus")

        env = {"MULLET_NOTIFY": "desktop,slack", "MULLET_SLACK_WEBHOOK": WEBHOOK}
        sent = notify.notify("title", "body", env=env, now=TUESDAY_10AM, run=boom, post=boom)
        assert sent == []
