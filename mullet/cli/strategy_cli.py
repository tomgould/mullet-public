"""CLI dispatcher for ``mullet strategy``: show, set or clear the session strategy.

Wired into ``mullet/cli/main.py`` through the same ``sys.argv[1]`` early
branch as ``mullet reap``. The strategy itself lives in ``mullet.core.strategy``.
"""

from __future__ import annotations

import argparse
import functools
import math
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from mullet.core import strategy as session_strategy

_SET_USAGE = ("usage: mullet strategy set [--budget USD] [--delegation "
              f"{'|'.join(session_strategy.DELEGATION_CHOICES)}] [--note TEXT]")


def _when(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


def _budget(text: str) -> float:
    """An argparse type for the session budget: a finite number of dollars, 0 or more."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an amount: {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"the budget must be 0 or more dollars (got {text!r})")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The ``mullet strategy`` argument parser."""
    choices = session_strategy.DELEGATION_CHOICES
    meanings = "; ".join(f"{c}: {session_strategy.DELEGATION_MEANINGS[c]}" for c in choices)
    parser = argparse.ArgumentParser(
        prog="mullet strategy",
        description="This session's budget and delegation choice, which `mullet job plan` and the "
                    "delegation tools follow while it is fresh (session.strategy_ttl_hours).")
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("show", help="the strategy in force, when it was set and when it expires")
    set_p = sub.add_parser("set", help="set this session's strategy", description=meanings)
    set_p.add_argument("--budget", type=_budget, metavar="USD",
                       help="dollars this session's jobs may spend on paid rungs")
    set_p.add_argument("--delegation", choices=choices, help=meanings)
    set_p.add_argument("--note", help="a short note kept with the strategy")
    sub.add_parser("clear", help="remove the strategy")
    return parser


def _show(now: float, path: Path | None, out: Callable[[str], None]) -> int:
    ttl = session_strategy.load_settings().strategy_ttl_hours
    stored = session_strategy.read(path)
    if stored is None:
        out("No session strategy is set. Set one with: mullet strategy set --budget USD "
            "--delegation ladder|claude-only|off")
        return 0
    if not session_strategy.is_fresh(stored, now, ttl):
        out(f"No session strategy is in force: the last one expired at {_when(stored.expires_at(ttl))}.")
        return 0
    out(f"Session strategy: {stored.describe()}")
    out(f"Set at {_when(stored.set_at)}; expires at {_when(stored.expires_at(ttl))} "
        f"(session.strategy_ttl_hours: {ttl:g}).")
    return 0


def cmd_strategy(argv: list[str], *, now: float | None = None, path: Path | None = None,
                 out: Callable[[str], None] = print,
                 err: Callable[[str], None] = functools.partial(print, file=sys.stderr)) -> int:
    """Entry point used by ``mullet strategy ...``. `path` is the strategy file (default in the state dir)."""
    args = build_parser().parse_args(argv)
    if now is None:
        now = time.time()
    if args.action in (None, "show"):
        return _show(now, path, out)
    if args.action == "clear":
        out("Session strategy cleared." if session_strategy.clear(path) else "No session strategy was set.")
        return 0
    if args.budget is None and args.delegation is None and args.note is None:
        err(_SET_USAGE)
        return 2
    current = session_strategy.load(now, path=path)
    chosen = session_strategy.Strategy(
        budget_usd=args.budget if args.budget is not None else (current.budget_usd if current else None),
        delegation=args.delegation or (current.delegation if current else "ladder"),
        set_at=now,
        note=args.note if args.note is not None else (current.note if current else ""),
    )
    session_strategy.save(chosen, path)
    ttl = session_strategy.load_settings().strategy_ttl_hours
    out(f"Session strategy set: {chosen.describe()}")
    out(f"In force until {_when(chosen.expires_at(ttl))}.")
    return 0
