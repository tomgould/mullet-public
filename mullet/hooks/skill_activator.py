"""Claude Code UserPromptSubmit hook: suggest relevant skills based on trigger keywords.

Reads user prompt from stdin JSON, checks it against registered Mullet skill
triggers, and injects a nudge if a skill matches. Never blocks the prompt: if
Mullet's skill registry can't be imported it says so on stderr and exits 1.

Run as `mullet hook skill-activator` (UserPromptSubmit).
"""

import json
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

    try:
        from mullet.core.skills.loader import SkillLoader
        from mullet.core.skills.registry import SkillRegistry
    except ImportError as exc:
        print(f"[hook:skill_activator] can't import Mullet's skill registry: {exc}", file=sys.stderr)
        _emit_runtime(_hook_start)
        sys.exit(1)

    try:
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
    except Exception as exc:
        # Never block the prompt, but don't hide the failure either.
        print(f"[hook:skill_activator] error: {exc}", file=sys.stderr)

    _emit_runtime(_hook_start)
    sys.exit(0)


if __name__ == "__main__":
    main()
