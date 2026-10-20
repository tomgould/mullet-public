"""CLI subcommand: ``mullet memory-sync``.

Usage
-----
    mullet memory-sync --observations   Sync saved observation summaries into memory
    mullet memory-sync --decisions      Sync .mullet/decisions.jsonl into memory
    mullet memory-sync --all            Run both (default when no flag given)
    mullet memory-sync --dry-run        Print what would be synced without writing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        if (candidate / ".mullet").is_dir() or (candidate / "mullet").is_dir():
            return candidate
    return here


def _dry_run_observations(base: Path) -> int:
    """Count observation summaries that would be synced."""
    summaries_dir = base / ".mullet" / "observations" / "summaries"
    if not summaries_dir.is_dir():
        return 0
    files = list(summaries_dir.glob("*.json"))
    if files:
        print(f"  Would sync {len(files)} observation summary file(s):")
        for f in files[:5]:
            print(f"    {f.name}")
        if len(files) > 5:
            print(f"    … and {len(files) - 5} more")
    return len(files)


def _dry_run_decisions(base: Path) -> int:
    """Count decisions that would be synced."""
    decisions_file = base / ".mullet" / "decisions.jsonl"
    if not decisions_file.exists():
        return 0
    count = 0
    try:
        for line in decisions_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                count += 1
    except Exception:
        pass
    if count:
        print(f"  Would sync {count} decision(s) from {decisions_file.name}")
        # Show a sample
        try:
            lines = [ln.strip() for ln in
                     decisions_file.read_text(encoding="utf-8").splitlines()
                     if ln.strip()][:3]
            for line in lines:
                d = json.loads(line)
                print(f"    [{d.get('kind','?')}] {d.get('excerpt','')[:80]}")
        except Exception:
            pass
    return count


def run(args: list[str]) -> int:
    """Entry point for ``mullet memory-sync``."""
    p = argparse.ArgumentParser(
        prog="mullet memory-sync",
        description="Sync observation summaries and decisions into Mullet memory.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--observations", action="store_true",
        help="Sync saved observation summaries only",
    )
    mode.add_argument(
        "--decisions", action="store_true",
        help="Sync .mullet/decisions.jsonl only",
    )
    mode.add_argument(
        "--all", action="store_true", dest="all_",
        help="Sync everything (default)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be synced without writing anything",
    )
    p.add_argument("--base", default=None,
                   help="Project root (default: auto-detect)")
    ns = p.parse_args(args)

    base = Path(ns.base) if ns.base else _project_root()

    do_observations = ns.observations or ns.all_ or not (ns.decisions)
    do_decisions = ns.decisions or ns.all_ or not (ns.observations)

    if ns.dry_run:
        print("[dry-run] No writes will be performed.\n")
        obs_count = dec_count = 0
        if do_observations:
            print("Observations:")
            obs_count = _dry_run_observations(base)
            if obs_count == 0:
                print("  No observation summaries found.")
        if do_decisions:
            print("Decisions:")
            dec_count = _dry_run_decisions(base)
            if dec_count == 0:
                print("  No decisions found.")
        print(f"\nTotal (dry-run): {obs_count} observations, {dec_count} decisions")
        return 0

    from mullet.core.memory_sync import (
        sync_observation_summary,
        sync_decisions,
        backfill_all,
    )

    if ns.all_ or (not ns.observations and not ns.decisions):
        # Default: sync everything
        counts = backfill_all(base=base)
        obs_synced = counts["observations"]
        dec_synced = counts["decisions"]
    elif ns.observations:
        # Observations only
        from mullet.core.observation_summary import SessionSummary
        summaries_dir = base / ".mullet" / "observations" / "summaries"
        obs_synced = 0
        if summaries_dir.is_dir():
            for json_file in sorted(summaries_dir.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    summary = SessionSummary(
                        session_id=data.get("session_id", ""),
                        started_at=data.get("started_at", ""),
                        ended_at=data.get("ended_at", ""),
                        tool_counts=data.get("tool_counts", {}),
                        files_touched=data.get("files_touched", []),
                        summary=data.get("summary", ""),
                        bullet_points=data.get("bullet_points", []),
                        open_questions=data.get("open_questions", []),
                    )
                    sync_observation_summary(summary, base=base)
                    obs_synced += 1
                except Exception as exc:
                    print(f"  Warning: skipped {json_file.name}: {exc}",
                          file=sys.stderr)
        dec_synced = 0
    else:
        # Decisions only
        obs_synced = 0
        dec_synced = sync_decisions(base=base)

    print(f"memory-sync complete: {obs_synced} observation(s), {dec_synced} decision(s) synced")

    # Sample output — show last entry from each category if any data was processed
    if obs_synced > 0:
        summaries_dir = base / ".mullet" / "observations" / "summaries"
        files = sorted(summaries_dir.glob("*.json"))
        if files:
            try:
                data = json.loads(files[-1].read_text(encoding="utf-8"))
                sid = data.get("session_id", "?")
                snippet = data.get("summary", "")[:100]
                print(f"  Sample observation: [{sid}] {snippet}")
            except Exception:
                pass

    if dec_synced > 0:
        decisions_file = base / ".mullet" / "decisions.jsonl"
        if decisions_file.exists():
            try:
                lines = [ln.strip() for ln in
                         decisions_file.read_text(encoding="utf-8").splitlines()
                         if ln.strip()]
                if lines:
                    d = json.loads(lines[-1])
                    print(f"  Sample decision: [{d.get('kind','?')}] "
                          f"{d.get('excerpt','')[:80]}")
            except Exception:
                pass

    return 0
