"""Claude Code UserPromptSubmit hook: inject recent session context on first prompt.

Reads the 3 most recent Mullet session titles from its session store
and injects a brief context note so CC knows what was worked on recently.
Only fires on the first user prompt in a new CC session (detected by
CLAUDE_CONVERSATION_ID env changing or not existing).

Never blocks the prompt: if Mullet's session store can't be imported it says
so on stderr and exits 1. Run as `mullet hook session-context`.
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
    except Exception:
        _emit_runtime(_hook_start)
        sys.exit(0)

    # Only inject on first prompt (heuristic: conversation < 10s old or no turn counter)
    turn = input_data.get("turn_number", input_data.get("turn", 0))
    if turn and int(turn) > 1:
        _emit_runtime(_hook_start)
        sys.exit(0)

    try:
        from mullet.core.agent.session import list_recent_sessions
    except ImportError as exc:
        print(f"[hook:session_context] can't import Mullet's session store: {exc}", file=sys.stderr)
        _emit_runtime(_hook_start)
        sys.exit(1)

    try:
        sessions = list_recent_sessions(limit=3)
        if not sessions:
            _emit_runtime(_hook_start)
            sys.exit(0)

        lines = ["[Mullet session context] Recent work:"]
        for s in sessions:
            title = s.get("title", "Untitled")
            msgs = s.get("messages", 0)
            ago = _time_ago(s.get("created_at", 0))
            lines.append(f"  • {title} ({msgs} messages, {ago})")

        print("\n" + "\n".join(lines) + "\n")
    except Exception as exc:
        print(f"[hook:session_context] error: {exc}", file=sys.stderr)

    _emit_runtime(_hook_start)
    sys.exit(0)


def _time_ago(ts: float) -> str:
    if not ts:
        return "unknown"
    delta = time.time() - ts
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


if __name__ == "__main__":
    main()
