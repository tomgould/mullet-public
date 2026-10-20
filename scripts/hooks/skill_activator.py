"""Claude Code UserPromptSubmit hook: suggest relevant skills based on trigger keywords.

Reads user prompt from stdin JSON, checks it against registered Mullet skill
triggers, and injects a nudge if a skill matches. Exits 0 always — never blocks.

Install via .claude/settings.json → hooks.UserPromptSubmit (alongside delegation_reminder.py).
"""

import json
import os
import sys
import time
from pathlib import Path


def _emit_runtime(hook_start: float) -> None:
    """Emit a one-line runtime metric to stderr when >10ms."""
    elapsed_ms = int((time.time() - hook_start) * 1000)
    if elapsed_ms > 10:
        print(f"[hook:{Path(__file__).stem}] {elapsed_ms}ms", file=sys.stderr)


def main():
    _hook_start = time.time()
    try:
        input_data = json.loads(sys.stdin.read())
        user_prompt = input_data.get("prompt", input_data.get("user_prompt", ""))
    except Exception:
        _emit_runtime(_hook_start)
        sys.exit(0)

    if not user_prompt or len(user_prompt) < 5:
        _emit_runtime(_hook_start)
        sys.exit(0)

    # Add mullet to path if needed
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if project_dir and project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    try:
        from mullet.core.skills.loader import SkillLoader
        from mullet.core.skills.registry import SkillRegistry

        loader = SkillLoader()
        registry = SkillRegistry(loader=loader)
        registry.load_all()

        match = registry.match(user_prompt)
        if match and match.match_type == "keyword" and match.confidence >= 0.9:
            skill = match.skill
            name = skill.metadata.name
            slash = f"/{name}"
            print(
                f"\n[Mullet skill hint] The '{name}' skill may be relevant here. "
                f"Type '{slash}' to invoke it, or continue naturally.\n"
            )
    except Exception:
        pass  # Never block the prompt

    _emit_runtime(_hook_start)
    sys.exit(0)


if __name__ == "__main__":
    main()
