"""CLI dispatcher for ``mullet night``: switch night mode on or off, or show it.

Wired into ``mullet/cli/main.py`` through the same ``sys.argv[1]`` early
branch as ``mullet job``. Night mode itself lives in ``mullet.jobs.night``.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from mullet.jobs import night


def _hhmm(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M")


def _state_line(state: night.NightState) -> str:
    if state.on and state.until is not None:
        return f"Night mode: on until {_hhmm(state.until)}"
    return "Night mode: off"


def build_parser() -> argparse.ArgumentParser:
    """The ``mullet night`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mullet night",
        description="Lean on the local GPU while you are away; switches itself off when you are back.")
    sub = parser.add_subparsers(dest="action", required=True)

    on = sub.add_parser("on", help="switch night mode on (until the next 07:30 by default)")
    on.add_argument("--until", metavar="HH:MM", help="local time to switch off at (the next one to come)")
    sub.add_parser("off", help="switch night mode off")
    sub.add_parser("status", help="whether night mode is on, until when, and why it last went off")
    return parser


def cmd_night(argv: list[str], *, now: float | None = None, path: Path | None = None,
              out: Callable[[str], None] = print,
              err: Callable[[str], None] = functools.partial(print, file=sys.stderr)) -> int:
    """Entry point used by ``mullet night ...``. `path` is the state file (default in the state dir)."""
    args = build_parser().parse_args(argv)
    if now is None:
        now = time.time()

    if args.action == "on":
        until = None
        if args.until is not None:
            try:
                until = night.parse_until(args.until, now)
            except ValueError as exc:
                err(f"mullet night: {exc}")
                return 2
        night.switch_on(until=until, now=now, path=path)
        out(_state_line(night.read_state(path)))
        return 0

    if args.action == "off":
        night.switch_off("switched off", now=now, path=path)
        out("Night mode: off")
        return 0

    state = night.check(now=now, path=path)
    out(_state_line(state))
    if not state.on and state.reason:
        out(f"Last switched off: {state.reason}")
    return 0
