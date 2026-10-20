"""CLI subcommand: ``mullet observations <action>``.

Actions
-------
list   [--session SID] [--limit N]   Print observations table, newest first.
tail   [--session SID] [--follow]    Print last N lines; --follow polls for new entries.
search <query> [--session SID] [--limit N]  Substring search across previews.
"""
import time
from pathlib import Path

from mullet.core.observations import (
    Observation,
    iter_recent,
    iter_session,
    search as obs_search,
)

try:
    from mullet.hud.colors import dim, YELLOW, GREEN, RESET
    _HAS_COLOR = True
except Exception:
    _HAS_COLOR = False


def _tool_color(tool: str) -> str:
    if not _HAS_COLOR:
        return tool
    colors = {
        "Read": GREEN,
        "Write": YELLOW,
        "Edit": YELLOW,
        "Bash": "\033[35m",  # magenta
    }
    c = colors.get(tool, "\033[36m")  # cyan default
    return f"{c}{tool}{RESET}"


def _fmt_row(obs: Observation) -> str:
    ts = obs.timestamp[:16].replace("T", " ")
    tool = _tool_color(obs.tool_name)
    preview = obs.tool_input_preview[:80]
    if _HAS_COLOR:
        return f"{dim(ts)}  {tool:<20}  {preview}"
    return f"{ts}  {obs.tool_name:<14}  {preview}"


def _project_root() -> Path:
    """Best-effort: find project root by walking up from cwd."""
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        if (candidate / ".mullet").is_dir() or (candidate / "mullet").is_dir():
            return candidate
    return here


def _run_list(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="mullet observations list")
    p.add_argument("--session", default=None)
    p.add_argument("--limit", type=int, default=20)
    ns = p.parse_args(args)

    base = _project_root()
    if ns.session:
        rows = list(iter_session(ns.session, base=base))
        rows = list(reversed(rows))[:ns.limit]
    else:
        rows = list(iter_recent(limit=ns.limit, base=base))

    if not rows:
        print("No observations found.")
        return 0

    for obs in rows:
        print(_fmt_row(obs))
    return 0


def _run_tail(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="mullet observations tail")
    p.add_argument("--session", default=None)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--follow", action="store_true")
    ns = p.parse_args(args)

    base = _project_root()

    def _get_rows(seen: int = 0) -> list[Observation]:
        if ns.session:
            rows = list(iter_session(ns.session, base=base))
            return list(reversed(rows))[seen : seen + ns.limit] if not seen else list(reversed(rows))[seen:]
        else:
            rows = list(iter_recent(limit=ns.limit + seen, base=base))
            return rows[seen:]

    initial = list(iter_recent(limit=ns.limit, base=base)) if not ns.session else list(reversed(list(iter_session(ns.session, base=base))))[-ns.limit:]
    for obs in reversed(list(initial)):
        print(_fmt_row(obs))

    if not ns.follow:
        return 0

    # Follow mode: track file sizes and poll
    seen = len(initial)
    try:
        while True:
            time.sleep(1)
            if ns.session:
                all_rows = list(iter_session(ns.session, base=base))
                new_rows = list(reversed(all_rows))[seen:][::-1]
            else:
                # Re-read recent; detect additions by count change
                all_rows = list(iter_recent(limit=seen + 100, base=base))
                # newest first; new items appear at front
                new_count = len(all_rows) - seen
                new_rows = all_rows[:new_count][::-1] if new_count > 0 else []

            for obs in new_rows:
                print(_fmt_row(obs))
            seen += len(new_rows)
    except KeyboardInterrupt:
        pass
    return 0


def _run_search(args: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="mullet observations search")
    p.add_argument("query")
    p.add_argument("--session", default=None)
    p.add_argument("--limit", type=int, default=20)
    ns = p.parse_args(args)

    base = _project_root()
    results = obs_search(ns.query, limit=ns.limit, base=base)

    if ns.session:
        results = [r for r in results if r.session_id == ns.session]

    if not results:
        print(f"No observations matching '{ns.query}'.")
        return 0

    for obs in results:
        print(_fmt_row(obs))
    return 0


def _run_summarise(args: list[str]) -> int:
    import argparse
    import json as _json
    from datetime import datetime, timezone

    p = argparse.ArgumentParser(prog="mullet observations summarise")
    p.add_argument("--session", default=None, help="Summarise a specific session ID")
    p.add_argument("--window", type=int, default=None, metavar="MINUTES",
                   help="Summarise the last N minutes across all sessions")
    p.add_argument("--provider", default="auto", choices=["local", "haiku", "auto"],
                   help="LLM provider to use (default: auto)")
    p.add_argument("--save", action="store_true",
                   help="Persist summary JSON to .mullet/observations/summaries/")
    ns = p.parse_args(args)

    from mullet.core.observation_summary import summarise_session, summarise_window
    base = _project_root()

    if ns.session:
        summary = summarise_session(ns.session, provider=ns.provider, base=base)
        label = f"session:{ns.session}"
    else:
        minutes = ns.window or 60
        summary = summarise_window(minutes=minutes, provider=ns.provider, base=base)
        label = f"window:{minutes}m"

    if summary is None:
        print("No observations found or no provider available.")
        return 1

    # Human-readable output
    sep = "-" * 60
    print(sep)
    print(f"Observation Summary  [{label}]")
    print(f"Period: {summary.started_at} → {summary.ended_at}")
    print(f"Tools:  {', '.join(f'{k}×{v}' for k, v in sorted(summary.tool_counts.items()))}")
    if summary.files_touched:
        print(f"Files:  {', '.join(summary.files_touched[:10])}")
    print(sep)
    print(summary.summary)
    if summary.bullet_points:
        print("\nWhat happened:")
        for b in summary.bullet_points:
            print(f"  • {b}")
    if summary.open_questions:
        print("\nOpen questions:")
        for q in summary.open_questions:
            print(f"  ? {q}")
    print(sep)

    if ns.save:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = (ns.session or f"window-{ns.window or 60}m").replace(":", "-")
        save_dir = base / ".mullet" / "observations" / "summaries"
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{slug}-{ts}.json"
        import dataclasses
        out_path.write_text(_json.dumps(dataclasses.asdict(summary), indent=2))
        print(f"Saved: {out_path}")

        # Sync into memory store (fail-open)
        try:
            from mullet.core.memory_sync import sync_observation_summary
            sync_observation_summary(summary, base=base)
        except Exception:
            pass  # memory sync is optional

    return 0


def _print_help() -> None:
    print("Usage: mullet observations <list|tail|search|summarise> [options]")
    print()
    print("Actions:")
    print("  list                       List observations (filterable)")
    print("  tail [-n N]                Show the most recent N observations")
    print("  search <query>             Full-text search across observations")
    print("  summarise [--window N]     LLM digest of recent observations")
    print()
    print("Observations are tool-call records emitted by the agent; useful for")
    print("auditing what an agent ran during a session.")


def run(args: list[str]) -> int:
    """Entry point for ``mullet observations``."""
    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return 0 if args else 1

    action, rest = args[0], args[1:]
    if action == "list":
        return _run_list(rest)
    if action == "tail":
        return _run_tail(rest)
    if action == "search":
        return _run_search(rest)
    if action == "summarise":
        return _run_summarise(rest)

    print(f"Unknown observations action: {action!r}. Use list, tail, search, or summarise.")
    return 1
