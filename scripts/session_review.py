#!/usr/bin/env python3
"""Session-end design principles review.

Usage:
    python3 scripts/session_review.py [--since <git-ref>] [--apply]

Options:
    --since <git-ref>   Review commits since this ref (default: HEAD~10)
    --apply             Print numbered improvement list for manual action
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Paths (relative to repo root — script is invoked from repo root)
# ---------------------------------------------------------------------------

PRINCIPLES_PATH = Path("docs/DESIGN_PRINCIPLES.md")
IMPROVEMENTS_PATH = Path("docs/PROCESS_IMPROVEMENTS.md")

DEFAULT_SINCE = "HEAD~10"


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="session_review.py",
        description="Review recent session against Mullet design principles.",
    )
    parser.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help=f"Git ref to review from (default: {DEFAULT_SINCE}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Print numbered ACTION list for manual application (no auto-apply).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_principles() -> str | None:
    """Return content of DESIGN_PRINCIPLES.md, or None if missing."""
    if not PRINCIPLES_PATH.exists():
        return None
    return PRINCIPLES_PATH.read_text(encoding="utf-8")


def read_prior_improvements(n: int = 40) -> str:
    """Return the last *n* lines of PROCESS_IMPROVEMENTS.md, or empty string."""
    if not IMPROVEMENTS_PATH.exists():
        return ""
    lines = IMPROVEMENTS_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: list[str]) -> str:
    """Run a git command and return stdout; empty string on failure."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def build_git_context(since: str) -> tuple[str, str, str]:
    """Return (log_output, stat_output, combined_context).

    Returns three empty strings when there are no commits in range.
    """
    log_range = f"{since}..HEAD"
    log = _git(["log", "--oneline", log_range])
    stat = _git(["diff", log_range, "--stat"])
    if not log and not stat:
        return "", "", ""
    combined = "\n".join([
        "## Recent commits",
        log or "(no commits)",
        "",
        "## Diff stat",
        stat or "(no diff stat)",
    ])
    return log, stat, combined


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_review_prompt(
    principles_content: str,
    git_context: str,
    since: str,
    prior_improvements: str,
) -> str:
    """Assemble the LLM review prompt."""
    return (
        "You are reviewing a development session against Mullet's design principles.\n"
        "Respond using EXACTLY these section headers (include the ## prefix):\n"
        "\n"
        "## VIOLATIONS\n"
        "List any principle violations with: principle name, file/evidence, severity (low/medium/high).\n"
        'Write "None" if clean.\n'
        "\n"
        "## PRINCIPLE_UPDATES\n"
        "List principles that should be updated based on what emerged. "
        'Format: "Principle name: proposed change".\n'
        'Write "None" if no updates needed.\n'
        "\n"
        "## NEW_PRINCIPLES\n"
        "List new principles that emerged from this session's patterns.\n"
        'Write "None" if none.\n'
        "\n"
        "## IMPROVEMENTS\n"
        "List concrete fixes. "
        'Format: "FILE: path/to/file — what to change (1-2 sentences)".\n'
        'Write "None" if no improvements needed.\n'
        "\n"
        "---\n"
        f"DESIGN PRINCIPLES:\n{principles_content}\n"
        "\n"
        f"RECENT CHANGES (since {since}):\n{git_context}\n"
        "\n"
        f"PRIOR PROCESS IMPROVEMENTS (last 40 lines):\n{prior_improvements}\n"
    )


# ---------------------------------------------------------------------------
# Mullet dispatch
# ---------------------------------------------------------------------------

def call_mullet(prompt: str) -> str:
    """Write prompt to a temp file and call mullet; return stdout."""
    prompt_file: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(prompt)
            prompt_file = f.name

        result = subprocess.run(
            ["mullet", "run", "--providers", "local,kimi",
             f"Review this file: {prompt_file}"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        return result.stdout.strip()

    finally:
        if prompt_file and os.path.exists(prompt_file):
            os.unlink(prompt_file)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_SECTION_ORDER = ("VIOLATIONS", "PRINCIPLE_UPDATES", "NEW_PRINCIPLES", "IMPROVEMENTS")


def parse_sections(response: str) -> dict[str, str]:
    """Split the LLM response into the four expected sections."""
    sections: dict[str, str] = {name: "" for name in _SECTION_ORDER}
    current: str | None = None
    buf: list[str] = []

    for raw_line in response.splitlines():
        stripped = raw_line.strip()
        matched = False
        for name in _SECTION_ORDER:
            if stripped.startswith(f"## {name}"):
                if current is not None:
                    sections[current] = "\n".join(buf).strip()
                current = name
                buf = []
                matched = True
                break
        if not matched and current is not None:
            buf.append(raw_line)

    if current is not None:
        sections[current] = "\n".join(buf).strip()

    return sections


def _count_section(content: str) -> str:
    """Return item count or 'None' string for a section's content."""
    stripped = content.strip()
    if not stripped or stripped.lower() == "none":
        return "None"
    # Count non-empty, non-separator lines as rough item count
    items = [ln for ln in stripped.splitlines() if ln.strip() and ln.strip() != "---"]
    return str(len(items))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_findings(sections: dict[str, str]) -> None:
    """Print all four sections to stdout with clear headers."""
    for name in _SECTION_ORDER:
        content = sections.get(name, "").strip()
        bar = "=" * 60
        print(f"\n{bar}")
        print(f"## {name.replace('_', ' ')}")
        print(bar)
        print(content if content else "None")


def print_improvements_action_list(sections: dict[str, str]) -> None:
    """Print numbered ACTION N: improvements list — no auto-apply."""
    content = sections.get("IMPROVEMENTS", "").strip()
    if not content or content.lower() == "none":
        print("\n(No improvements queued.)")
        return

    print("\n--- IMPROVEMENTS (manual action required) ---")
    items: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Lines starting with FILE: or a digit bullet are individual items
        if stripped.upper().startswith("FILE:") or (
            len(stripped) > 1 and stripped[0].isdigit() and stripped[1] in ".)"
        ):
            items.append(stripped)
        elif stripped.startswith(("-", "*")):
            items.append(stripped.lstrip("-* ").strip())
        elif items:
            items[-1] += " " + stripped
        else:
            items.append(stripped)

    if not items:
        items = [ln.strip() for ln in content.splitlines() if ln.strip()]

    for n, item in enumerate(items, start=1):
        print(f"  ACTION {n}: {item}")

    print(f"\n{len(items)} improvement(s) queued — apply manually.")


# ---------------------------------------------------------------------------
# Append summary to PROCESS_IMPROVEMENTS.md
# ---------------------------------------------------------------------------

def append_summary(sections: dict[str, str]) -> None:
    """Append a structured session-review block to PROCESS_IMPROVEMENTS.md."""
    today = datetime.date.today().isoformat()

    violations_content = sections.get("VIOLATIONS", "").strip() or "None"
    improvements_content = sections.get("IMPROVEMENTS", "").strip() or "None"

    v_count = _count_section(violations_content)
    u_count = _count_section(sections.get("PRINCIPLE_UPDATES", ""))
    n_count = _count_section(sections.get("NEW_PRINCIPLES", ""))
    i_count = _count_section(improvements_content)

    block = (
        f"\n## {today} — Session review\n"
        f"\n"
        f"**Violations:** {v_count}\n"
        f"**Principle updates:** {u_count}\n"
        f"**New principles:** {n_count}\n"
        f"**Improvements queued:** {i_count}\n"
        f"\n"
        f"{violations_content}\n"
        f"\n"
        f"{improvements_content}\n"
    )

    with IMPROVEMENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(block)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    since: str = args.since

    # Step 2 — Read design principles
    principles_content = read_principles()
    if principles_content is None:
        print(f"[WARN] {PRINCIPLES_PATH} not found — skipping review.")
        return 0

    # Step 3 — Git context
    _log, _stat, git_context = build_git_context(since)
    if not git_context:
        print(f"No changes since {since}, skipping review.")
        return 0

    # Step 4 — Prior improvements
    prior_improvements = read_prior_improvements(40)

    # Step 5 — Build prompt
    prompt = build_review_prompt(principles_content, git_context, since, prior_improvements)

    # Step 6 — Call mullet
    response = call_mullet(prompt)

    # Step 7 — Parse
    sections = parse_sections(response)

    # Step 8 — Print findings
    print_findings(sections)

    # Step 9 — Apply listing
    if args.apply:
        print_improvements_action_list(sections)

    # Step 10 — Append to PROCESS_IMPROVEMENTS.md
    append_summary(sections)

    return 0


if __name__ == "__main__":
    sys.exit(main())
