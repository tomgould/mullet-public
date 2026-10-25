"""CLI for ``mullet migrate-state``: move an older ``~/.mullet`` into the XDG directories.

Wired into ``mullet/cli/main.py`` through the same ``sys.argv[1]`` early
branch as ``mullet preset``. This is the only code that moves user data; the
rule for where everything lives is in :mod:`mullet.core.paths`.

Each entry of ``~/.mullet`` goes to ``<its kind's directory>/<same name>``
(kinds come from ``paths.LEGACY_LAYOUT``; unknown entries count as data).
The command refuses to move anything when a destination already exists and
isn't an empty directory, and leaves ``~/.mullet/MOVED.txt`` saying where
everything went. While that note is there, ``~/.mullet`` no longer counts as
an install.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mullet.core import paths

_DESCRIPTION = (
    "Move an older install's ~/.mullet into the XDG directories and leave "
    "~/.mullet/MOVED.txt saying where things went. Stop running Mullet "
    "processes (such as the MCP server in open Claude Code sessions) first."
)


@dataclass(frozen=True)
class Move:
    """One entry of ``~/.mullet`` and where it goes."""

    source: Path
    target: Path
    kind: str


def _kind_dirs() -> dict[str, Path]:
    return {
        "config": paths.config_dir(),
        "state": paths.xdg_state_dir(),
        "data": paths.xdg_data_dir(),
        "cache": paths.xdg_cache_dir(),
    }


def plan_moves() -> list[Move]:
    """Every entry of ``~/.mullet`` except ``MOVED.txt``, with its destination."""
    legacy = paths.legacy_dir()
    if not legacy.is_dir():
        return []
    dirs = _kind_dirs()
    moves = []
    for entry in sorted(legacy.iterdir(), key=lambda p: p.name):
        if entry.name == paths.MOVED_NOTE:
            continue
        kind = paths.LEGACY_LAYOUT.get(entry.name, "data")
        moves.append(Move(entry, dirs[kind] / entry.name, kind))
    return moves


def _occupied(path: Path) -> bool:
    """True when ``path`` exists and isn't an empty directory."""
    if path.is_symlink():
        return True
    if path.is_dir():
        return any(path.iterdir())
    return path.exists()


def _inside(path: Path, root: Path) -> bool:
    resolved = Path(path).expanduser().resolve()
    root = root.resolve()
    return resolved == root or root in resolved.parents


def find_problems(moves: list[Move]) -> list[str]:
    """Reasons not to move anything; empty when the migration is safe."""
    problems = []
    legacy = paths.legacy_dir()
    for kind, directory in _kind_dirs().items():
        if _inside(directory, legacy):
            problems.append(f"the {kind} directory {directory} is inside {legacy}; point it elsewhere first")
    pools_override = os.environ.get("TRAWL_POOLS_DIR")
    if pools_override and _inside(Path(pools_override), legacy):
        problems.append(f"TRAWL_POOLS_DIR points inside {legacy}; unset it or point it elsewhere first")
    for move in moves:
        if _occupied(move.target):
            problems.append(f"{move.target} already exists and isn't empty (it would receive {move.source})")
    return problems


def _note_text(moves: list[Move]) -> str:
    dirs = _kind_dirs()
    when = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        f"Mullet moved its files out of this directory on {when} (mullet migrate-state).",
        "",
        "It now keeps them in the XDG base directories:",
        *(f"  {kind:<7} {directory}" for kind, directory in dirs.items()),
        "",
        "What went where:",
        *(f"  {move.source.name} -> {move.target}" for move in moves),
        "",
        "While this file is here, Mullet doesn't use this directory. You can delete it.",
        "",
    ]
    return "\n".join(lines)


def _write_note(moves: list[Move]) -> Path:
    note = paths.legacy_dir() / paths.MOVED_NOTE
    text = _note_text(moves)
    if note.exists():
        text = note.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + text
    note.write_text(text, encoding="utf-8")
    return note


def _move(move: Move) -> None:
    move.target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if move.target.is_dir() and not move.target.is_symlink():
        move.target.rmdir()  # empty, as find_problems checked
    shutil.move(str(move.source), str(move.target))


def _print_plan(moves: list[Move], verb: str) -> None:
    print(f"{verb} {len(moves)} entr{'y' if len(moves) == 1 else 'ies'} from {paths.legacy_dir()}:")
    for move in moves:
        print(f"  {move.source.name} -> {move.target}")


def cmd_migrate_state(argv: list[str]) -> int:
    """Run ``mullet migrate-state [--dry-run]``; return the exit code."""
    parser = argparse.ArgumentParser(prog="mullet migrate-state", description=_DESCRIPTION)
    parser.add_argument("--dry-run", action="store_true", help="show what would move and change nothing")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code not in (0, None) else 0

    legacy = paths.legacy_dir()
    if not legacy.is_dir():
        print(f"Nothing to migrate: {legacy} does not exist.")
        return 0
    moves = plan_moves()
    if not moves:
        print(f"Nothing to migrate: {legacy} holds nothing to move.")
        return 0

    problems = find_problems(moves)
    if problems:
        print("Can't migrate, so nothing was moved:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    if args.dry_run:
        _print_plan(moves, "Would move")
        print("Nothing was moved (dry run).")
        return 0

    done: list[Move] = []
    for move in moves:
        try:
            _move(move)
        except OSError as exc:
            if done:
                _print_plan(done, "Moved")
            print(f"Stopped at {move.source}: {exc}")
            print("Fix that and run `mullet migrate-state` again; it carries on with what is left.")
            return 1
        done.append(move)

    _print_plan(done, "Moved")
    print(f"Left a note at {_write_note(done)}.")
    return 0
