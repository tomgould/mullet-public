"""Claude Code UserPromptSubmit hook: suggest delegation, but only when the
Anthropic budget is actually under pressure.

Delegating spends third-party tokens, which is real money. Work Claude Code
does in-context is already covered by the subscription. So this hook stays
silent unless `_delegation_gate.delegation_warranted()` says the 5-hour or
weekly allowance is genuinely short, and when it does fire it tells Claude to
get the operator's go-ahead before delegating anything.

The prompt-shape patterns below decide only *whether this looks like delegable
work*. They never fire on their own — the budget gate is checked first.

Claude Code sends the prompt as JSON on stdin with key "prompt" (not
"user_prompt"). stdout is appended as extra context the model sees before
responding. We exit 0 unconditionally so the hook never blocks the prompt.

Install via .claude/settings.json → hooks.UserPromptSubmit.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _delegation_gate import CONFIRM_LINE, delegation_warranted  # noqa: E402

ESCALATION_THRESHOLD = 5

ESCALATION_NORMAL = """\
[delegation suggestion — budget pressure, ESCALATED] {reason}.
{count} manual tool calls logged without a delegation.
Two more manual calls and the hard block activates (exit 2), for as long as the
budget stays under pressure.

If in-context work is no longer viable, offloading the next chunk is an option:

  mcp__mullet__mullet_opencode(
      task="<describe what you need done>"
  )

{confirm}

Only fall back to mullet_run if opencode is unavailable, and only for
research/analysis tasks (NOT file edits — mullet_run has no filesystem access).
The Agent tool (a Claude Code subagent) stays on the Anthropic allowance, so it
does not relieve this pressure, but it costs no third-party tokens either."""

ESCALATION_LAST_CHANCE = """\
[delegation suggestion — budget pressure, FINAL WARNING] {reason}.
{count} manual tool calls logged without a delegation.
One more manual call and the hard block activates (exit 2), for as long as the
budget stays under pressure.

  mcp__mullet__mullet_opencode(
      task="<describe what you need done>"
  )

{confirm}

Only fall back to mullet_run if opencode is unavailable, and only for
research/analysis tasks (NOT file edits — mullet_run has no filesystem access)."""

# Back-compat alias for tests/imports that reference ESCALATED_TEMPLATE.
ESCALATED_TEMPLATE = ESCALATION_NORMAL

SOFT_REMINDER = """\
[delegation suggestion — budget pressure] {reason}.
This prompt looks like delegable work, if in-context work is no longer viable.

  mcp__mullet__mullet_opencode(
      task="<describe what you need done>"
  )

{confirm}

Only use mullet_run for pure research/analysis with no file edits."""


def _session_file(data: dict) -> Path:
    session_id = data.get("session_id") or str(os.getppid())
    return Path(f"/tmp/claude_session_{session_id}.json")


def _read_manual_tool_count(data: dict) -> int:
    try:
        sf = _session_file(data)
        if sf.exists():
            state = json.loads(sf.read_text())
            return int(state.get("manual_tools", 0))
    except Exception:
        pass
    return 0


_runtime_emitted = False


def _emit_runtime(hook_start: float) -> None:
    """Emit a one-line runtime metric to stderr when >10ms.

    Emits at most once. Every exit path calls this and then falls through the
    `finally` in main(), which would otherwise print the same line twice.
    """
    global _runtime_emitted
    if _runtime_emitted:
        return
    _runtime_emitted = True
    elapsed_ms = int((time.time() - hook_start) * 1000)
    if elapsed_ms > 10:
        print(f"[hook:{Path(__file__).stem}] {elapsed_ms}ms", file=sys.stderr)


def main() -> None:
    _hook_start = time.time()
    try:
        input_data = json.loads(sys.stdin.read())
        user_prompt = input_data.get("prompt", input_data.get("user_prompt", ""))

        # Budget gate first, before anything else. Delegation costs real money
        # at a third-party provider; in-context work is already paid for. With
        # the Anthropic allowance comfortable there is nothing to suggest, so
        # say nothing at all rather than nudging toward a spend.
        warranted, reason = delegation_warranted()
        if not warranted:
            _emit_runtime(_hook_start)
            sys.exit(0)

        # Check escalation next (regardless of prompt content).
        # The hard guard in manual_tool_guard.py blocks when count > 6, i.e.
        # the 7th manual call is the first one rejected. So:
        #   count == 5 → two more manual calls allowed
        #   count == 6 → one more manual call allowed (final warning)
        #   count >= 7 → block has already fired; stay silent to avoid noise.
        count = _read_manual_tool_count(input_data)
        if count == 5:
            print(ESCALATION_NORMAL.format(count=count, reason=reason, confirm=CONFIRM_LINE))
            _emit_runtime(_hook_start)
            sys.exit(0)
        if count == 6:
            print(ESCALATION_LAST_CHANCE.format(count=count, reason=reason, confirm=CONFIRM_LINE))
            _emit_runtime(_hook_start)
            sys.exit(0)
        if count >= 7:
            _emit_runtime(_hook_start)
            sys.exit(0)

        # Soft reminder based on prompt content / length
        prompt_lower = user_prompt.lower()

        # Hard override: user explicitly wants Mullet/agents — always remind.
        mullet_override_patterns = [
            r"\bmullet\b", r"\buse agents?\b", r"\bdelegate\b",
            r"\bfire up\b", r"\buse kimi\b", r"\buse gemini\b",
        ]
        for pattern in mullet_override_patterns:
            if re.search(pattern, prompt_lower):
                print(SOFT_REMINDER.format(reason=reason, confirm=CONFIRM_LINE))
                _emit_runtime(_hook_start)
                sys.exit(0)

        patterns = [
            r"\bplan\b", r"\bresearch\b", r"\banalyse\b", r"\banalyze\b",
            r"\bdesign\b", r"\bcompare\b", r"\bevaluate\b", r"\brecommend\b",
            r"\bwhat do you think\b", r"\broadmap\b", r"\bshould we\b",
            r"\bdecide\b", r"\bassessment\b", r"\breview\b",
        ]

        match_found = len(user_prompt) > 400
        if not match_found:
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    match_found = True
                    break

        if match_found:
            print(SOFT_REMINDER.format(reason=reason, confirm=CONFIRM_LINE))

    except Exception as exc:
        print(f"Error in delegation_reminder hook: {exc}", file=sys.stderr)
    finally:
        _emit_runtime(_hook_start)
        sys.exit(0)


if __name__ == "__main__":
    main()
