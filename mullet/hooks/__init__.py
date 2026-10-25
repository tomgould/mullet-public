"""Claude Code hooks that ship with Mullet, run as `mullet hook <name>`.

Each hook is a module in this package whose `main()` reads Claude Code's JSON
payload on stdin. `mullet setup-claude --hooks <names>` registers the ones you
choose in Claude Code's settings.json by that command; none are registered by
default. A hook that can't load prints the reason to stderr and exits 1, which
Claude Code shows as a non-blocking error, instead of failing silently.

This module is imported on every hook run, so it imports nothing heavy.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class HookSpec:
    """One hook: its module, what it does, and the events it registers for."""

    name: str
    module: str
    summary: str
    # (event, matcher) pairs; a matcher of None means the event takes none.
    registrations: tuple[tuple[str, str | None], ...]


_SPECS = (
    HookSpec(
        "agent-hud-tracker", "agent_hud_tracker",
        "Shows Claude Code subagents on Mullet's HUD while they run "
        "(writes .mullet/state/agents.json in the project).",
        (("PreToolUse", "Agent"), ("PostToolUse", "Agent")),
    ),
    HookSpec(
        "compress-whitespace", "compress_whitespace",
        "Collapses runs of spaces in a prompt, leaving code blocks and tables alone.",
        (("UserPromptSubmit", ""),),
    ),
    HookSpec(
        "delegation-reminder", "delegation_reminder",
        "Suggests delegating work to Mullet, but only when the Anthropic usage "
        "allowance is nearly used up and delegation is on.",
        (("UserPromptSubmit", ""),),
    ),
    HookSpec(
        "manual-tool-guard", "manual_tool_guard",
        "Blocks long runs of manual Read/Grep/Glob/Bash calls without a delegation, "
        "only while the Anthropic allowance is nearly used up.",
        (("PreToolUse", "Bash|Grep|Read|Glob|Task|mcp__mullet__.*"),),
    ),
    HookSpec(
        "observation-capture", "observation_capture",
        "Logs each Read/Write/Edit/Bash call to .mullet/observations/ in the project, "
        "for `mullet observations`.",
        (("PostToolUse", "Read|Write|Edit|Bash"),),
    ),
    HookSpec(
        "opus-override", "user_prompt_detect_override",
        'Turns Mullet\'s Opus override on or off when a prompt says "use opus" or '
        '"stop using opus".',
        (("UserPromptSubmit", ""),),
    ),
    HookSpec(
        "pre-agent-guard", "pre_agent_guard",
        "Warns when a subagent is started without a cheaper model override.",
        (("PreToolUse", "Task|Agent"),),
    ),
    HookSpec(
        "preset-prompt", "preset_prompt",
        "At session start, tells Claude which Mullet provider preset is active.",
        (("SessionStart", None),),
    ),
    HookSpec(
        "pretooluse-rewriter", "pretooluse_rewriter",
        "Fills in a routing tier for mullet_run, mullet_delegate and mullet_swarm "
        "calls from the current budget and any Opus override.",
        (("PreToolUse", "mcp__mullet__.*"),),
    ),
    HookSpec(
        "session-context", "session_context",
        "On the first prompt, lists the three most recent Mullet sessions.",
        (("UserPromptSubmit", ""),),
    ),
    HookSpec(
        "skill-activator", "skill_activator",
        "Suggests a Mullet skill when a prompt matches one of its trigger words.",
        (("UserPromptSubmit", ""),),
    ),
)

HOOKS: dict[str, HookSpec] = {spec.name: spec for spec in _SPECS}

# Script names from before the hooks moved into the package.
_ALIASES = {
    "mullet-observation-capture": "observation-capture",
    "mullet-pretooluse-rewriter": "pretooluse-rewriter",
    "user-prompt-detect-override": "opus-override",
}


def find(name: str) -> HookSpec | None:
    """Look a hook up by name; underscores, a `.py` suffix and old script names are accepted."""
    key = name.strip().removesuffix(".py").replace("_", "-").lower()
    return HOOKS.get(_ALIASES.get(key, key))


def describe() -> str:
    """One line per hook, for `mullet hook list` and `setup-claude --hooks list`."""
    width = max(len(name) for name in HOOKS)
    return "\n".join(f"  {spec.name:<{width}}  {spec.summary}" for spec in HOOKS.values())


def run(name: str) -> int:
    """Run one hook's `main()` and return its exit code.

    An unknown name or a module that can't be imported is reported on stderr
    with exit code 1: Claude Code shows that as a non-blocking error, so a
    broken hook is visible without blocking the prompt or the tool call.
    """
    spec = find(name)
    if spec is None:
        print(f"mullet hook: there is no hook called {name!r}; run `mullet hook list`",
              file=sys.stderr)
        return 1
    try:
        module = importlib.import_module(f"{__name__}.{spec.module}")
    except Exception as exc:
        print(f"mullet hook {spec.name}: can't load {__name__}.{spec.module}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    result = module.main()
    return result if isinstance(result, int) else 0
