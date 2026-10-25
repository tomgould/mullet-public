"""Tell the job's owner when a job needs them or is done.

A job that blocks, pauses for a usage limit or finishes sends one short
message, so an unattended job never sits waiting unnoticed. Two channels:

- ``desktop``: ``notify-send`` on Linux, ``osascript`` on macOS. On by default.
- ``slack``: a POST to a Slack incoming webhook (``MULLET_SLACK_WEBHOOK``),
  sent only inside working hours (``MULLET_NOTIFY_HOURS``, default
  ``Mon-Fri 09:00-17:30``, in ``MULLET_NOTIFY_TZ``, default Europe/London).

``MULLET_NOTIFY`` picks the channels: ``desktop``, ``slack``, both
comma-separated, or ``off``. Settings come from the environment, then
Mullet's ``.env`` files. Messages carry the job id, its goal and what to run
next, never code or test output; Slack messages carry only fixed wording
(no goal, block reason or error text), since they leave the machine. Sending never raises: a job's outcome
doesn't depend on whether anyone was told.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from typing import Callable, Mapping, Optional

logger = logging.getLogger(__name__)

EVENTS = ("blocked", "paused", "spend", "finished")
DEFAULT_HOURS = "Mon-Fri 09:00-17:30"
DEFAULT_TZ = "Europe/London"
_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WEBHOOK_PREFIX = "https://hooks.slack.com/"


def within_hours(spec: str, now: datetime) -> bool:
    """Whether ``now`` falls inside ``spec``, such as ``Mon-Fri 09:00-17:30``.

    ``always`` (or an empty spec) means any time. A day range may wrap
    (``Sat-Mon``). Raises ValueError for a spec it can't read.
    """
    spec = spec.strip().lower()
    if spec in ("", "always"):
        return True
    days, _, times = spec.partition(" ")
    first, _, last = days.partition("-")
    try:
        lo, hi = _DAYS.index(first[:3]), _DAYS.index((last or first)[:3])
    except ValueError:
        raise ValueError(f"unknown day in {spec!r}") from None
    start, _, end = times.strip().partition("-")
    for t in (start, end):
        datetime.strptime(t, "%H:%M")
    weekday = now.weekday()
    on_day = lo <= weekday <= hi if lo <= hi else (weekday >= lo or weekday <= hi)
    clock = now.strftime("%H:%M")
    return on_day and start <= clock < end


def _now_in(tz_name: str) -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_name))
    except Exception as exc:  # unknown zone or no tz database
        logger.warning("notify: time zone %r unavailable (%s); using local time", tz_name, exc)
        return datetime.now()


def message(event: str, job_id: str, goal: str, detail: str = "", *, local: bool = True) -> tuple[str, str]:
    """The (title, body) for ``event``: one of ``EVENTS``.

    A local (desktop) message may carry the goal's first line and the detail.
    Anything that leaves the machine (``local=False``: Slack) is fixed
    wording only: the job id, its status and what to run next, never the
    goal, gate output, block reasons or error text.
    """
    title = f"Mullet job {job_id} {event}"
    shown = detail if local else ""
    if event == "blocked":
        body = f"Blocked: {shown or 'a step needs attention'}. Run: mullet job report {job_id}"
    elif event == "paused":
        body = f"Paused for a usage limit{' until ' + shown if shown else ''}; it carries on by itself."
    elif event == "spend":
        body = ("Paused: the spending cap is reached and Claude's weekly window is tight. Raise the cap with "
                f"mullet job resume {job_id} --max-spend <dollars>, or resume later.")
    elif event == "finished":
        body = (f"Finished: {shown or 'done'}. Review: mullet job report {job_id}; "
                f"merge: mullet job merge {job_id}")
    else:
        raise ValueError(f"unknown event {event!r}")
    if not local:
        return title, body
    lines = goal.strip().splitlines()
    short_goal = " ".join(lines[0].split()) if lines else ""
    if len(short_goal) > 80:
        short_goal = short_goal[:77] + "..."
    return title, f"{short_goal}\n{body}" if short_goal else body


def _desktop(title: str, body: str, run: Callable[..., object]) -> bool:
    if sys.platform == "darwin":
        # AppleScript strings take \" \\ and \n but not \u escapes.
        quote = lambda s: json.dumps(s, ensure_ascii=False)  # noqa: E731
        script = f"display notification {quote(body)} with title {quote(title)}"
        cmd = ["osascript", "-e", script]
    else:
        if not shutil.which("notify-send"):
            return False
        if not (os.environ.get("DBUS_SESSION_BUS_ADDRESS") or os.environ.get("DISPLAY")
                or os.environ.get("WAYLAND_DISPLAY")):
            return False
        cmd = ["notify-send", "--app-name=Mullet", title, body]
    try:
        run(cmd, timeout=10, check=False, capture_output=True)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("notify: desktop notification failed: %s", exc)
        return False
    return True


def _post_json(url: str, payload: dict) -> None:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


def _slack(url: str, title: str, body: str, post: Callable[[str, dict], None]) -> bool:
    if not url.startswith(_WEBHOOK_PREFIX):
        logger.warning("notify: MULLET_SLACK_WEBHOOK is not a Slack incoming webhook; not sending")
        return False
    try:
        post(url, {"text": f"{title}\n{body}"})
    except Exception as exc:
        logger.warning("notify: Slack webhook failed: %s", type(exc).__name__)
        return False
    return True


def _setting(name: str, env: Optional[Mapping[str, str]]) -> Optional[str]:
    if env is not None:
        return env.get(name)
    value = os.environ.get(name)
    if value:
        return value
    from mullet.core.env import dotenv_value
    return dotenv_value(name)


def notify(
    title: str,
    body: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
    run: Callable[..., object] = subprocess.run,
    post: Callable[[str, dict], None] = _post_json,
    slack: Optional[tuple[str, str]] = None,
) -> list[str]:
    """Send ``title`` and ``body`` on every configured channel; return the ones used.

    ``slack`` is the (title, body) Slack gets instead, when it must differ
    (a job's Slack message is fixed wording, without the goal). ``env``
    replaces the environment and ``.env`` lookup, and ``now`` the clock in
    the configured time zone; both are for tests.
    """
    try:
        channels = {
            c.strip().lower()
            for c in (_setting("MULLET_NOTIFY", env) or "desktop").split(",")
            if c.strip()
        }
        if "off" in channels:
            return []
        sent: list[str] = []
        if "desktop" in channels and _desktop(title, body, run):
            sent.append("desktop")
        webhook = _setting("MULLET_SLACK_WEBHOOK", env)
        if "slack" in channels and webhook:
            hours = _setting("MULLET_NOTIFY_HOURS", env) or DEFAULT_HOURS
            when = now or _now_in(_setting("MULLET_NOTIFY_TZ", env) or DEFAULT_TZ)
            try:
                in_hours = within_hours(hours, when)
            except ValueError as exc:
                logger.warning("notify: MULLET_NOTIFY_HOURS %s; not sending to Slack", exc)
                in_hours = False
            if in_hours and _slack(webhook, *(slack or (title, body)), post):
                sent.append("slack")
        return sent
    except Exception as exc:
        logger.warning("notify: %s", exc)
        return []


def notify_job(
    event: str,
    job_id: str,
    goal: str,
    detail: str = "",
    *,
    env: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
    run: Callable[..., object] = subprocess.run,
    post: Callable[[str, dict], None] = _post_json,
) -> list[str]:
    """Send ``event`` for a job on every configured channel; return the ones used.

    ``env`` replaces the environment and ``.env`` lookup, and ``now`` the
    clock in the configured time zone; both are for tests.
    """
    try:
        title, body = message(event, job_id, goal, detail)
        return notify(title, body, env=env, now=now, run=run, post=post,
                      slack=message(event, job_id, goal, detail, local=False))
    except Exception as exc:
        logger.warning("notify: %s", exc)
        return []
