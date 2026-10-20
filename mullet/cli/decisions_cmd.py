"""CLI subcommand: ``mullet decisions <action>``.

Actions
-------
scan                                          Scan all sources and persist to .mullet/decisions.jsonl.
list [--kind K] [--source S] [--limit N]     Print decisions, newest first.
search <query> [--kind K] [--limit N]        Substring search across excerpts.
"""
from __future__ import annotations

import sys

from mullet.core.decisions import (
    Decision,
    load,
    persist,
    scan_all,
    search as dec_search,
)

def cyan(t: str) -> str: return t
def dim(t: str) -> str: return t
def bold(t: str) -> str: return t
def yellow(t: str) -> str: return t
def green(t: str) -> str: return t
def magenta(t: str) -> str: return t
RESET = ""
_HAS_COLOR = False

try:
    from mullet.hud.colors import cyan, dim, bold, yellow, green, magenta, RESET
    _HAS_COLOR = True
except Exception:  # pragma: no cover — colors always available in test env
    pass


_KIND_COLOR = {
    "decision": cyan,
    "rejection": yellow,
    "rationale": green,
}

_SOURCE_COLOR = {
    "docs": dim,
    "roadmap": magenta,
    "progress": green,
    "commit": cyan,
}


def _kind_str(kind: str) -> str:
    fn = _KIND_COLOR.get(kind, lambda t: t)
    return fn(kind)


def _source_str(source_type: str, source_path: str) -> str:
    fn = _SOURCE_COLOR.get(source_type, lambda t: t)
    return fn(f"{source_type}:{source_path}")


def _print_decision(d: Decision) -> None:
    ts = d.timestamp or ""
    print(f"  {_kind_str(d.kind)} [{_source_str(d.source_type, d.source_path)}] {dim(ts)}")
    print(f"    {d.excerpt[:200]}")
    print()


def _cmd_scan(args: list[str]) -> int:
    """Run scan_all, persist, report counts by source_type and kind."""
    decisions = scan_all()
    if not decisions:
        print("No decisions found.")
        return 0

    written = persist(decisions)

    # Tally
    by_type: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for d in decisions:
        by_type[d.source_type] = by_type.get(d.source_type, 0) + 1
        by_kind[d.kind] = by_kind.get(d.kind, 0) + 1

    print(bold(f"Scanned {len(decisions)} decisions ({written} total on disk)"))
    print()
    print("  By source:")
    for src, count in sorted(by_type.items()):
        print(f"    {src:15s}  {count}")
    print()
    print("  By kind:")
    for kind, count in sorted(by_kind.items()):
        print(f"    {kind:12s}  {count}")

    return 0


def _cmd_list(args: list[str]) -> int:
    """Print decisions from disk, with optional filters."""
    import argparse
    p = argparse.ArgumentParser(prog="mullet decisions list", add_help=False)
    p.add_argument("--kind", choices=["decision", "rejection", "rationale"], default=None)
    p.add_argument("--source", default=None, help="Filter by source_type (docs, commit, ...)")
    p.add_argument("--limit", type=int, default=20)
    ns, _ = p.parse_known_args(args)

    all_d = load()
    if not all_d:
        print("No decisions on disk — run: mullet decisions scan")
        return 0

    # Sort newest first, None last
    timestamped = [d for d in all_d if d.timestamp]
    no_ts = [d for d in all_d if not d.timestamp]
    timestamped.sort(key=lambda d: d.timestamp or "", reverse=True)
    ordered = timestamped + no_ts

    results = [
        d for d in ordered
        if (ns.kind is None or d.kind == ns.kind)
        and (ns.source is None or d.source_type == ns.source)
    ][: ns.limit]

    if not results:
        print("No matching decisions.")
        return 0

    for d in results:
        _print_decision(d)

    return 0


def _cmd_search(args: list[str]) -> int:
    """Search decisions by substring."""
    import argparse
    p = argparse.ArgumentParser(prog="mullet decisions search", add_help=False)
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--kind", choices=["decision", "rejection", "rationale"], default=None)
    p.add_argument("--limit", type=int, default=20)
    ns, _ = p.parse_known_args(args)

    if not ns.query:
        print("Usage: mullet decisions search <query> [--kind K] [--limit N]", file=sys.stderr)
        return 1

    results = dec_search(query=ns.query, limit=ns.limit, kind=ns.kind)
    if not results:
        print(f"No decisions found matching {ns.query!r}")
        return 0

    for d in results:
        _print_decision(d)

    return 0


def run(args: list[str]) -> int:
    """Entry point for ``mullet decisions``."""
    if not args or args[0] in ("-h", "--help", "help"):
        print(
            "Usage: mullet decisions <action>\n"
            "  scan               Scan all sources (git + docs/decisions) and persist\n"
            "  list [--kind K] [--source S] [--limit N]   List recorded decisions\n"
            "  search <query> [--kind K] [--limit N]      Full-text search across decisions\n"
            "\n"
            "Decisions are architectural choices auto-extracted from commit messages\n"
            "and docs/decisions/ ADR-style records."
        )
        return 0

    action = args[0]
    rest = args[1:]

    if action == "scan":
        return _cmd_scan(rest)
    elif action == "list":
        return _cmd_list(rest)
    elif action == "search":
        return _cmd_search(rest)
    else:
        print(f"Unknown action: {action!r}.  Use scan, list, or search.", file=sys.stderr)
        return 1
