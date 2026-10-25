"""`mullet init` — scaffold a target repo with Mullet's delegation rules, hooks,
loop agreement, pool config, and seeded project memory.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates" / "init"

# Patterns to append to .gitignore
_GITIGNORE_PATTERNS = [
    "# Mullet runtime state",
    ".mullet/progress.txt",
    ".mullet/routing_history.jsonl",
    ".mullet/decisions.jsonl",
    ".mullet/state/",
    ".mullet/observations/",
    ".mullet/checkpoints/",
    ".mullet/overnight_loop.log",
    ".mullet/memory/",
]

_MEMORY_FILES = [
    "feedback_prefer_concurrent.md",
    "feedback_measure_before_delegate.md",
    "feedback_opencode_for_edits.md",
    "feedback_test_writing_delegation.md",
]


# The project's .claude/settings.json runs the delegation reminder through the
# installed `mullet`, found on PATH, so the committed file holds no
# machine-specific path. Older versions copied the hook script into
# .claude/hooks/ and ran it with python3, which broke once the script needed
# Mullet's own modules.
_HOOK_COMMAND = "mullet hook delegation-reminder"
_OLD_HOOK_COMMAND = "python3 ${CLAUDE_PROJECT_DIR}/.claude/hooks/delegation_reminder.py"


def _render_template(path: Path, **kwargs: str) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


def _pool_config_path(pool: str) -> Path:
    from mullet.core.env import get_pools_dir

    return get_pools_dir() / pool / "config.yaml"


def _memory_dir(target_dir: Path) -> Path:
    encoded = "-" + str(target_dir).replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded / "memory"


# Seeded only when MULLET_NO_AI_ATTRIBUTION is on: a personal commit-trailer
# preference, not something every project should inherit.
_NO_AI_ATTRIBUTION_MEMORY = "feedback_coauthored_by.md"
_NO_AI_ATTRIBUTION_INDEX_LINE = (
    "- [No Co-Authored-By in commits](feedback_coauthored_by.md)"
    " — Never append Co-Authored-By to git commits"
)


def _memory_files() -> list[str]:
    from mullet.core.env import no_ai_attribution

    files = list(_MEMORY_FILES)
    if no_ai_attribution():
        files.insert(0, _NO_AI_ATTRIBUTION_MEMORY)
    return files


def _memory_index_text(template: Path) -> str:
    """MEMORY.md for a new project, listing the opt-in note when it's seeded."""
    from mullet.core.env import no_ai_attribution

    text = template.read_text(encoding="utf-8")
    if no_ai_attribution() and _NO_AI_ATTRIBUTION_MEMORY not in text:
        head, sep, rest = text.partition("\n\n")
        text = f"{head}{sep}{_NO_AI_ATTRIBUTION_INDEX_LINE}\n{rest}" if sep else (
            text.rstrip("\n") + "\n" + _NO_AI_ATTRIBUTION_INDEX_LINE + "\n"
        )
    return text


def _flag_from_remaining(remaining: list[str] | None, flag: str) -> bool:
    return bool(remaining and flag in remaining)


def _value_from_remaining(remaining: list[str] | None, flag: str, default: str | None = None) -> str | None:
    if not remaining:
        return default
    try:
        idx = remaining.index(flag)
        if idx + 1 < len(remaining) and not remaining[idx + 1].startswith("-"):
            return remaining[idx + 1]
    except ValueError:
        pass
    return default


def _target_dir_from_remaining(remaining: list[str] | None) -> str | None:
    if not remaining:
        return None
    for item in remaining[1:]:
        if not item.startswith("-"):
            return item
    return None


def run_init(args: argparse.Namespace) -> None:
    """Initialize a project directory with Mullet scaffolding."""
    remaining = getattr(args, "remaining", None) or []
    target_dir = Path(
        getattr(args, "target_dir", None)
        or _value_from_remaining(remaining, "--target-dir")
        or _target_dir_from_remaining(remaining)
        or "."
    ).resolve()
    pool = (
        getattr(args, "pool", None)
        or getattr(args, "client", None)
        or _value_from_remaining(remaining, "--pool")
        or target_dir.name
    )
    providers_raw = (
        getattr(args, "providers", None)
        or _value_from_remaining(remaining, "--providers")
    )
    providers = providers_raw.split(",") if providers_raw else ["openrouter", "haiku", "kimi", "sonnet"]
    model = getattr(args, "model", None) or _value_from_remaining(remaining, "--model") or "deepseek/deepseek-v4-flash"
    speed = getattr(args, "speed", None) or _value_from_remaining(remaining, "--speed") or "balanced"
    force = getattr(args, "force", False) or _flag_from_remaining(remaining, "--force")
    dry_run = getattr(args, "dry_run", False) or _flag_from_remaining(remaining, "--dry-run")
    no_claude = getattr(args, "no_claude", False) or _flag_from_remaining(remaining, "--no-claude")
    no_memory = getattr(args, "no_memory", False) or _flag_from_remaining(remaining, "--no-memory")
    no_pool = getattr(args, "no_pool", False) or _flag_from_remaining(remaining, "--no-pool")

    def _log(msg: str) -> None:
        prefix = "[dry-run] " if dry_run else ""
        print(f"{prefix}{msg}")

    # ------------------------------------------------------------------
    # a) Pool setup
    # ------------------------------------------------------------------
    if not no_pool:
        pool_cfg = _pool_config_path(pool)
        if pool_cfg.exists():
            _log(f"⚠ Pool {pool} already exists, skipping")
        else:
            from mullet.cli.create_pool import create_pool_silent

            if dry_run:
                _log(f"Would create pool: {pool}")
            else:
                create_pool_silent(slug=pool, machine_name=target_dir.name)
            _log(f"✓ Pool: {pool}")
    else:
        _log("Pool creation skipped (--no-pool)")

    # ------------------------------------------------------------------
    # b) mullet.yaml
    # ------------------------------------------------------------------
    mullet_yaml = target_dir / "mullet.yaml"
    if mullet_yaml.exists() and not force:
        _log("⚠ mullet.yaml exists, skipping")
    else:
        providers_yaml = "[" + ", ".join(providers) + "]"
        content = _render_template(
            _TEMPLATES_ROOT / "mullet.yaml.tmpl",
            pool=pool,
            providers_yaml=providers_yaml,
            model=model,
            speed=speed,
        )
        if dry_run:
            _log(f"Would write {mullet_yaml}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            mullet_yaml.write_text(content, encoding="utf-8")
            _log(f"✓ Created {mullet_yaml}")

    # ------------------------------------------------------------------
    # c) .mullet/ directory + LOOP_AGREEMENT.md
    # ------------------------------------------------------------------
    dot_mullet = target_dir / ".mullet"
    loop_agreement = dot_mullet / "LOOP_AGREEMENT.md"
    if loop_agreement.exists() and not force:
        _log("⚠ .mullet/LOOP_AGREEMENT.md exists, skipping")
    else:
        source_loop = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "LOOP_AGREEMENT.md"
        )
        if not source_loop.exists():
            # Fallback for installed package where docs/ may not be present
            source_loop = Path(__file__).resolve().parent.parent / "docs" / "LOOP_AGREEMENT.md"
        if source_loop.exists():
            if dry_run:
                _log(f"Would copy LOOP_AGREEMENT.md → {loop_agreement}")
            else:
                dot_mullet.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_loop, loop_agreement)
                _log(f"✓ Copied LOOP_AGREEMENT.md → {loop_agreement}")
        else:
            _log("⚠ docs/LOOP_AGREEMENT.md not found, skipping")

    # ------------------------------------------------------------------
    # d) .claude/ scaffolding
    # ------------------------------------------------------------------
    if not no_claude:
        claude_dir = target_dir / ".claude"
        claude_md = claude_dir / "CLAUDE.md"
        settings_json = claude_dir / "settings.json"

        if claude_md.exists() and not force:
            _log("⚠ .claude/CLAUDE.md exists, skipping")
        else:
            providers_str = ", ".join(providers)
            content = _render_template(
                _TEMPLATES_ROOT / "claude_md.tmpl",
                pool=pool,
                providers_str=providers_str,
                model=model,
            )
            if dry_run:
                _log(f"Would write {claude_md}")
            else:
                claude_dir.mkdir(parents=True, exist_ok=True)
                claude_md.write_text(content, encoding="utf-8")
                _log(f"✓ Created {claude_md}")

        if settings_json.exists() and not force:
            _log("⚠ .claude/settings.json exists, skipping")
        else:
            content = _render_template(_TEMPLATES_ROOT / "settings.json.tmpl",
                                       hook_command=_HOOK_COMMAND)
            if dry_run:
                _log(f"Would write {settings_json} (runs `{_HOOK_COMMAND}` on each prompt)")
            else:
                claude_dir.mkdir(parents=True, exist_ok=True)
                settings_json.write_text(content, encoding="utf-8")
                _log(f"✓ Created {settings_json} (runs `{_HOOK_COMMAND}` on each prompt)")
    else:
        _log("Claude scaffolding skipped (--no-claude)")

    # ------------------------------------------------------------------
    # e) .gitignore
    # ------------------------------------------------------------------
    gitignore = target_dir / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if "# Mullet runtime state" in existing:
        _log("⚠ .gitignore already contains Mullet patterns, skipping")
    else:
        block = "\n".join(_GITIGNORE_PATTERNS) + "\n"
        if dry_run:
            _log(f"Would append Mullet patterns to {gitignore}")
        else:
            with open(gitignore, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(block)
            _log(f"✓ Appended Mullet patterns to {gitignore}")

    # ------------------------------------------------------------------
    # f) Project memory seeding
    # ------------------------------------------------------------------
    if not no_memory:
        mem_dir = _memory_dir(target_dir)
        if dry_run:
            _log(f"Would seed memory in {mem_dir}")
        else:
            mem_dir.mkdir(parents=True, exist_ok=True)

            for fname in _memory_files():
                src = _TEMPLATES_ROOT / "memory" / fname
                dst = mem_dir / fname
                if dst.exists():
                    _log(f"⚠ Memory file {fname} exists, skipping")
                    continue
                if src.exists():
                    shutil.copy2(src, dst)
                    _log(f"✓ Seeded memory: {fname}")
                else:
                    _log(f"⚠ Memory template {fname} not found, skipping")

            memory_index = mem_dir / "MEMORY.md"
            if not memory_index.exists():
                src = _TEMPLATES_ROOT / "memory" / "MEMORY.md"
                if src.exists():
                    memory_index.write_text(_memory_index_text(src), encoding="utf-8")
                    _log("✓ Seeded memory: MEMORY.md")
            else:
                _log("⚠ MEMORY.md exists, skipping")
    else:
        _log("Memory seeding skipped (--no-memory)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    claude_status = "installed" if not no_claude else "skipped"
    memory_status = "seeded" if not no_memory else "skipped"
    print()
    print(f"mullet init complete for {target_dir}")
    print(f"  pool: {pool}")
    print(f"  providers: {', '.join(providers)}")
    print(f"  model: {model}")
    print(f"  Claude hooks: {claude_status}")
    print(f"  Memory: {memory_status}")
    print(f"Run: mullet reindex --pool {pool}   ← next step")


# ---------------------------------------------------------------------------
# Updateable files and their old-provider patterns
# ---------------------------------------------------------------------------

_OLD_PROVIDERS_PATTERNS = [
    "[kimi, gemini, anthropic]",
    "[kimi, gemini]",
    "[kimi, anthropic]",
    "kimi, gemini, anthropic",
    "[openrouter, kimi, sonnet, haiku]",
    "openrouter, kimi, sonnet, haiku",
]
_NEW_PROVIDERS_LINE = "providers: [openrouter, haiku, kimi, sonnet]"
_NEW_MODEL_LINE = "  default_model: deepseek/deepseek-v4-flash"


def run_update(args: argparse.Namespace) -> None:
    """Update an existing project's Mullet scaffolding files to latest versions."""
    remaining = getattr(args, "remaining", None) or []
    target_dir = Path(
        getattr(args, "target_dir", None)
        or _value_from_remaining(remaining, "--target-dir")
        or _target_dir_from_remaining(remaining)
        or "."
    ).resolve()
    dry_run = getattr(args, "dry_run", False) or _flag_from_remaining(remaining, "--dry-run")
    no_claude = getattr(args, "no_claude", False) or _flag_from_remaining(remaining, "--no-claude")

    def _log(msg: str) -> None:
        prefix = "[dry-run] " if dry_run else ""
        print(f"{prefix}{msg}")

    updated = 0

    # ------------------------------------------------------------------
    # 1) .mullet/LOOP_AGREEMENT.md — always overwrite with latest
    # ------------------------------------------------------------------
    source_loop = (
        Path(__file__).resolve().parent.parent.parent / "docs" / "LOOP_AGREEMENT.md"
    )
    if not source_loop.exists():
        source_loop = Path(__file__).resolve().parent.parent / "docs" / "LOOP_AGREEMENT.md"

    loop_dest = target_dir / ".mullet" / "LOOP_AGREEMENT.md"
    if source_loop.exists():
        if dry_run:
            _log(f"Would update {loop_dest}")
        else:
            loop_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_loop, loop_dest)
            _log("✓ Updated .mullet/LOOP_AGREEMENT.md")
        updated += 1
    else:
        _log("⚠ docs/LOOP_AGREEMENT.md not found — skipping")

    # ------------------------------------------------------------------
    # 2) .claude/settings.json — run the hook through `mullet hook`
    # ------------------------------------------------------------------
    if not no_claude:
        settings_json = target_dir / ".claude" / "settings.json"
        text = settings_json.read_text(encoding="utf-8") if settings_json.exists() else ""
        if _OLD_HOOK_COMMAND in text:
            if dry_run:
                _log(f"Would point the delegation hook in {settings_json} at `{_HOOK_COMMAND}`")
            else:
                settings_json.write_text(text.replace(_OLD_HOOK_COMMAND, _HOOK_COMMAND),
                                         encoding="utf-8")
                _log(f"✓ Pointed the delegation hook in .claude/settings.json at `{_HOOK_COMMAND}`")
            updated += 1
            old_copy = target_dir / ".claude" / "hooks" / "delegation_reminder.py"
            if old_copy.exists():
                _log(f"  {old_copy} is no longer used; delete it when you like")
        else:
            _log("⚠ .claude/settings.json has no old delegation hook to update, skipping")

    # ------------------------------------------------------------------
    # 3) mullet.yaml — update providers if using old pattern
    # ------------------------------------------------------------------
    mullet_yaml = target_dir / "mullet.yaml"
    if mullet_yaml.exists():
        content = mullet_yaml.read_text(encoding="utf-8")
        original = content
        for old_pattern in _OLD_PROVIDERS_PATTERNS:
            if old_pattern in content:
                # Replace the providers line
                lines = content.splitlines()
                new_lines = []
                for line in lines:
                    if line.startswith("providers:") and any(p in line for p in _OLD_PROVIDERS_PATTERNS):
                        new_lines.append(_NEW_PROVIDERS_LINE)
                    elif "default_model: kimi-k2.6" in line or "default_model: kimi-k2.5" in line:
                        new_lines.append(_NEW_MODEL_LINE)
                    else:
                        new_lines.append(line)
                content = "\n".join(new_lines) + "\n"
                break
        if content != original:
            if dry_run:
                _log("Would update mullet.yaml providers → [haiku, sonnet, openrouter, kimi]")
            else:
                mullet_yaml.write_text(content, encoding="utf-8")
                _log("✓ Updated mullet.yaml providers → [haiku, sonnet, openrouter, kimi]")
            updated += 1
        else:
            _log("⚠ mullet.yaml providers already up to date, skipping")
    else:
        _log("⚠ mullet.yaml not found — run `mullet init` first")

    # ------------------------------------------------------------------
    # 4) .claude/CLAUDE.md — update delegation ladder and defaults only
    # ------------------------------------------------------------------
    if not no_claude:
        claude_md = target_dir / ".claude" / "CLAUDE.md"
        if claude_md.exists():
            content = claude_md.read_text(encoding="utf-8")
            original = content

            # Replace delegation ladder table (everything between the table header and "**Hard rules:**")
            _NEW_LADDER = (
                "| **1. Local LLM** | Free. Economy mode, simple short edits | `mullet_opencode` (auto-selects) or `mullet_run --providers local` |\n"
                "| **2. OpenRouter** | Pay-per-token. Default cloud — cheap, reliable workhorse | `mullet_opencode` (default) or `mullet_run --providers openrouter` |\n"
                "| **3. Anthropic Haiku** | Prepaid (Max plan). Substitute OpenRouter when pressure <60% — saves pay-per-token spend | Direct in-context or `mullet_run --providers haiku` |\n"
                "| **4. Anthropic Sonnet** | Prepaid (Max plan). Complex tasks; substitute when pressure <60% | Direct in-context or `mullet_run --providers sonnet` |\n"
                "| **5. Kimi K2** | Pay-per-token. Large context; OpenRouter fallback | `mullet_opencode` or `mullet_run --providers kimi` |\n"
                "| **6. Gemini** | Last resort — pay-per-token, expensive for large contexts | `mullet_run --providers gemini` |\n"
                "| **7. Opus** | Orchestration and commit decisions only | Direct in-context |"
            )

            _NEW_HARD_RULES = (
                "**Hard rules:**\n"
                "- File edits >5 lines → `mullet_opencode` always — never in-context Opus writing code\n"
                "- Check `check_budget()` before each task — if Anthropic pressure <60%, substitute Haiku/Sonnet for the OpenRouter task (saves pay-per-token spend)\n"
                "- Default to OpenRouter when Anthropic pressure >60%\n"
                "- Mullet's routing matrix (`routing_matrix.py`) enforces this automatically for API delegation calls"
            )

            # Find and replace the old ladder table rows
            # Replace the table (lines starting with | **1. through | **5. or | **6.)
            table_pattern = re.compile(
                r"(\| Tier \| When to use \| How \|\n\|---\|---\|---\|\n)"
                r"((?:\|.*\|\n?)+)",
                re.MULTILINE,
            )
            content = table_pattern.sub(
                lambda m: m.group(1) + _NEW_LADDER + "\n",
                content,
            )

            # Replace the hard rules block
            hard_rules_pattern = re.compile(
                r"\*\*Hard rules:\*\*\n(?:- .*\n?)+",
                re.MULTILINE,
            )
            content = hard_rules_pattern.sub(_NEW_HARD_RULES + "\n", content)

            # Update Delegation defaults section: replace the two bullet lines
            content = re.sub(
                r"- Default provider: .*",
                "- Default provider: openrouter, kimi, sonnet, haiku",
                content,
            )
            content = re.sub(
                r"- Default model: .*",
                "- Default model: deepseek/deepseek-v4-flash",
                content,
            )

            if content != original:
                if dry_run:
                    _log("Would update .claude/CLAUDE.md delegation ladder and defaults")
                else:
                    claude_md.write_text(content, encoding="utf-8")
                    _log("✓ Updated .claude/CLAUDE.md delegation ladder and defaults")
                updated += 1
            else:
                _log("⚠ .claude/CLAUDE.md already up to date, skipping")
        else:
            _log("⚠ .claude/CLAUDE.md not found — run `mullet init` first")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(f"mullet update complete for {target_dir}")
    print(f"  {updated} file(s) updated")
    if dry_run:
        print("  (dry-run — no files written)")
