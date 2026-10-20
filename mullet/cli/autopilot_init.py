"""`mullet autopilot init` — scaffold `.mullet/autopilot/` state files.

Creates `.mullet/autopilot/` with five role-specific state files
(`goal.md`, `plans.md`, `standards.md`, `implement.md`, `progress.md`) from
starter templates, so overnight autopilot runs have a clear last-human-gate
before autonomous execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Templates ship under this package at `templates/<scenario>/`.
_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "autopilot_templates"

TEMPLATES = ("greenfield-api", "refactor", "feature-sprint")
STATE_FILES = ("goal.md", "plans.md", "standards.md", "implement.md", "progress.md")


def _target_dir(project_root: Path) -> Path:
    return project_root / ".mullet" / "autopilot"


def run_autopilot_init(argv: list[str], cwd: Path | None = None) -> int:
    """CLI entry point for `mullet autopilot init [--template NAME] [--force]`.

    Returns a POSIX exit code. Does not call `sys.exit()` so tests can drive it.
    """
    cwd = cwd or Path.cwd()
    template = "greenfield-api"
    force = False
    list_only = False

    it = iter(argv)
    for arg in it:
        if arg in ("--template", "-t"):
            template = next(it, "")
        elif arg == "--force":
            force = True
        elif arg in ("--list", "-l"):
            list_only = True
        elif arg in ("--help", "-h"):
            _print_help()
            return 0
        else:
            print(f"autopilot init: unknown argument '{arg}'", file=sys.stderr)
            _print_help()
            return 2

    if list_only:
        print("Available autopilot templates:")
        for name in TEMPLATES:
            print(f"  - {name}")
        return 0

    if template not in TEMPLATES:
        print(
            f"autopilot init: unknown template '{template}'. "
            f"Available: {', '.join(TEMPLATES)}",
            file=sys.stderr,
        )
        return 2

    target = _target_dir(cwd)
    target.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    skipped: list[Path] = []

    for name in STATE_FILES:
        dest = target / name
        if dest.exists() and not force:
            skipped.append(dest)
            continue
        content = _load_template(template, name)
        if name == "standards.md":
            content = _with_commit_rules(content)
        dest.write_text(content, encoding="utf-8")
        created.append(dest)

    if created:
        print(f"Created {len(created)} file(s) in {target}:")
        for p in created:
            print(f"  + {p.relative_to(cwd)}")
    if skipped:
        print(f"Skipped {len(skipped)} existing file(s) (pass --force to overwrite):")
        for p in skipped:
            print(f"  = {p.relative_to(cwd)}")

    if created and not skipped:
        print()
        print("Next steps:")
        print(f"  1. Edit {target.relative_to(cwd)}/goal.md — define the project goal")
        print(f"  2. Edit {target.relative_to(cwd)}/plans.md — flesh out milestones/tasks")
        print( "  3. Run:  mullet autopilot --require-goal \"<your task>\"")
    return 0


_NO_AI_ATTRIBUTION_RULE = "- No `Co-Authored-By` or other AI attribution trailers in commits."


def _with_commit_rules(standards: str) -> str:
    """Add the no-attribution rule to standards.md when MULLET_NO_AI_ATTRIBUTION is on."""
    from mullet.core.env import no_ai_attribution

    if not no_ai_attribution() or "Co-Authored-By" in standards:
        return standards
    if "## Git" in standards:
        head, _, tail = standards.partition("## Git")
        # Append to the end of the Git section (up to the next heading).
        section, sep, rest = tail.partition("\n## ")
        section = section.rstrip("\n") + "\n" + _NO_AI_ATTRIBUTION_RULE + "\n"
        return head + "## Git" + section + (("\n## " + rest) if sep else "")
    return standards.rstrip("\n") + "\n\n## Git\n" + _NO_AI_ATTRIBUTION_RULE + "\n"


def _load_template(template: str, filename: str) -> str:
    """Load a template file. Falls back to a minimal skeleton if missing."""
    candidate = _TEMPLATES_ROOT / template / filename
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    # Shared fallback — the skeleton for any state file not specialised by the template.
    shared = _TEMPLATES_ROOT / "_shared" / filename
    if shared.exists():
        return shared.read_text(encoding="utf-8")
    return f"# {filename}\n\n(template missing — fill in manually)\n"


def _print_help() -> None:
    print("Usage: mullet autopilot init [--template NAME] [--force] [--list]")
    print()
    print("Scaffolds .mullet/autopilot/{goal,plans,standards,implement,progress}.md")
    print()
    print("Options:")
    print("  --template, -t NAME   Template to use (default: greenfield-api)")
    print("  --list, -l            List available templates")
    print("  --force               Overwrite existing state files")
    print("  --help, -h            Show this help")


def require_goal(cwd: Path | None = None) -> tuple[bool, str]:
    """Enforce the 'last human interaction' gate.

    Returns (ok, message). ok=False when `.mullet/autopilot/goal.md` is missing
    or effectively empty (header-only). Used by `--require-goal` on autopilot/ralph.
    """
    cwd = cwd or Path.cwd()
    goal = _target_dir(cwd) / "goal.md"
    if not goal.exists():
        return False, (
            f"--require-goal: {goal.relative_to(cwd)} not found. "
            f"Run `mullet autopilot init` first, then edit goal.md."
        )
    text = goal.read_text(encoding="utf-8")
    # Compare against the shared template — if the file matches any of the shipped
    # templates substantially, it's unfilled. This is stricter than "count non-header
    # lines" because our templates include filled-looking bullets (e.g. `- Language:
    # [...]`) that would otherwise pass a naive content check.
    template_signatures = _template_signatures()
    stripped = "\n".join(
        line.rstrip() for line in text.splitlines() if line.strip()
    )
    for sig in template_signatures:
        # If >80% of the template's non-blank lines match verbatim, it's unfilled.
        overlap = sum(1 for line in sig if line in stripped)
        if sig and overlap / len(sig) > 0.8:
            return False, (
                f"--require-goal: {goal.relative_to(cwd)} looks like an unfilled template. "
                "Fill in Problem / Outcome / Acceptance Criteria before running autopilot."
            )
    # Also require at least one line that is not a header, blank, or placeholder
    meaningful = [
        line for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith("- [ ]")
        and not (line.strip().startswith("[") and line.strip().endswith("]"))
    ]
    if len(meaningful) < 3:
        return False, (
            f"--require-goal: {goal.relative_to(cwd)} has <3 content lines. "
            "Fill in Problem / Outcome / Acceptance Criteria before running autopilot."
        )
    return True, f"goal.md OK ({len(meaningful)} content lines)"


def _template_signatures() -> list[list[str]]:
    """Return stripped non-blank lines for every shipped goal.md template."""
    sigs: list[list[str]] = []
    for name in (*TEMPLATES, "_shared"):
        path = _TEMPLATES_ROOT / name / "goal.md"
        if not path.exists():
            continue
        sigs.append([
            line.rstrip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ])
    return sigs
