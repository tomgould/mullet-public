"""CLI dispatcher for ``mullet ladder``: cross-job reports from the ledger.

Reads the jobs ledger only and never calls a model, a provider or the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

from mullet.jobs.ladder_report import build, render_json, render_text
from mullet.jobs.ledger import Ledger


def _now() -> float:
    """Current time, patchable in tests."""
    return time.time()


def _parse_date(value: str) -> float:
    """Parse a YYYY-MM-DD date and return its local midnight timestamp."""
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid YYYY-MM-DD date") from exc
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _positive_int(value: str) -> int:
    """Parse a positive integer."""
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if n <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive integer")
    return n


def cmd_report(args: argparse.Namespace) -> int:
    """Build and print the cross-job ladder report."""
    since: float | None = None
    if args.since is not None:
        since = args.since
    elif args.days is not None:
        since = _now() - args.days * 86400

    with Ledger() as ledger:
        report = build(ledger, since=since)
        if args.json:
            print(json.dumps(render_json(report), indent=2))
        else:
            print(render_text(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The ``mullet ladder`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mullet ladder",
        description="Cross-job reports about the delegation ladder.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    report = sub.add_parser("report", help="report on how the delegation ladder has done across jobs")
    group = report.add_mutually_exclusive_group()
    group.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        type=_parse_date,
        help="include jobs created on or after this date (local midnight)",
    )
    group.add_argument(
        "--days",
        type=_positive_int,
        metavar="N",
        help="include jobs created in the last N days",
    )
    report.add_argument("--json", action="store_true", help="output as JSON")
    report.set_defaults(fn=cmd_report)
    return parser


def cmd_ladder(argv: list[str]) -> int:
    """Entry point used by ``mullet ladder ...``."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return args.fn(args)
