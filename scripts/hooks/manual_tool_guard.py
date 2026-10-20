#!/usr/bin/env python3
"""PreToolUse hook: hard-block Claude from running too many manual tool calls
(Bash, Grep, Read, Glob) without an intervening delegation via mullet_run.

Threshold: 6 counted calls. On the 7th, exit code 2 to block the tool call.

Session state is persisted in /tmp/claude_session_<id>.json so the counter
survives across individual hook invocations within the same Claude session.

Set MULLET_GUARD_BYPASS=1 to skip blocking (still logs).

Install via .claude/settings.json → hooks.PreToolUse with matcher
Bash|Grep|Read|Glob.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

COUNTED_TOOLS = {"Bash", "Grep", "Read", "Glob"}

DELEGATION_TOOLS = {
    "mcp__mullet__mullet_run",
    "mcp__mullet__mullet_delegate",
    "mcp__mullet__mullet_swarm",
    "Task",
}

# Bash commands that are allow-listed and should never be counted
ALLOW_LIST_RE = re.compile(
    r"^(git |pwd|python3 -m py_compile|python3 -m pytest|mullet |ls |which )"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _delegation_gate import delegation_warranted  # noqa: E402

THRESHOLD = 6

BLOCK_TEMPLATE_MCP = """\
[manual-tool guard] Blocked: {n} manual tool calls ({tool_names}) without a delegation.
This is a hard limit set by CLAUDE.md (Token Budget rule).

To proceed, delegate the next chunk of work via:

  mcp__mullet__mullet_run(
      task="<describe what you need done>",
      files=["<paths>"],         # paste actual files to edit
      apply=true,
      providers="kimi,gemini",   # stay off Anthropic pool
      tier="STANDARD"
  )

If the delegate fails, read its error — do NOT pivot to doing it yourself.
Override (only if truly necessary): set env MULLET_GUARD_BYPASS=1 and retry."""

BLOCK_TEMPLATE_FALLBACK = """\
[manual-tool guard] Blocked: {n} manual tool calls ({tool_names}) without a delegation.
Mullet MCP is not available in this session — use the Agent tool
(Claude Code subagent) instead:

  Agent(
      description="<short task>",
      subagent_type="general-purpose",  # or "Explore" for read-only
      prompt="<task with file paths + what to return>"
  )

To bypass the guard instead: set MULLET_GUARD_BYPASS=1 in the
parent environment (inline `VAR=1 cmd` does not work — hook
subprocesses don't inherit inline assignments)."""

# Back-compat alias: some tests/imports reference BLOCK_TEMPLATE directly.
BLOCK_TEMPLATE = BLOCK_TEMPLATE_MCP


def _is_mcp_available() -> bool:
    """Probe whether the Mullet MCP server is registered with this
    Claude Code session.

    The `mcp` SDK being importable is NOT a reliable signal — the
    package is almost always installed even when no MCP server is
    running. We check only explicit positive signals:

      1. MULLET_MCP_ENABLED env var is truthy (user- or harness-set).
      2. An MCP server PID marker file exists at
         ~/.mullet/state/mcp_server.pid (future: server writes this
         on startup).
      3. A delegation via mcp__mullet__* has been observed in this
         session's recent history (last 2 hours).

    If none fire, assume MCP is unavailable and recommend the Agent
    tool as the fallback delegation path. This is the safer default:
    mistakenly pointing the user at an unavailable MCP tool creates
    the exact deadlock P0-5 was meant to fix.
    """
    if os.environ.get("MULLET_MCP_ENABLED", "").strip().lower() in (
        "1", "true", "yes",
    ):
        return True

    try:
        marker = Path.home() / ".mullet" / "state" / "mcp_server.pid"
        if marker.exists():
            return True
    except Exception:
        pass

    return False


def _mcp_seen_recently(state: dict, window_seconds: int = 7200) -> bool:
    """Second-chance probe: has an mcp__mullet__* tool fired in this
    session recently? Complements the env/marker checks — if the
    harness has actually dispatched an MCP tool, MCP is live."""
    cutoff = int(time.time()) - window_seconds
    for entry in reversed(state.get("history", [])):
        if entry.get("ts", 0) < cutoff:
            break
        tool = entry.get("tool", "")
        if tool.startswith("mcp__mullet__"):
            return True
    return False


def _session_file(data: dict) -> Path:
    """Resolve the session state file path."""
    session_id = data.get("session_id") or str(os.getppid())
    return Path(f"/tmp/claude_session_{session_id}.json")


def _load_state(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {"manual_tools": 0, "last_delegation_ts": 0, "history": []}


def _save_state(path: Path, state: dict) -> None:
    try:
        path.write_text(json.dumps(state))
    except Exception:
        pass


def _is_allowlisted_bash(tool_input: dict) -> bool:
    command = tool_input.get("command", "") if tool_input else ""
    return bool(ALLOW_LIST_RE.match(str(command)))


def _emit_runtime(hook_start: float) -> None:
    """Emit a one-line runtime metric to stderr when >10ms."""
    elapsed_ms = int((time.time() - hook_start) * 1000)
    if elapsed_ms > 10:
        print(f"[hook:{Path(__file__).stem}] {elapsed_ms}ms", file=sys.stderr)


def main() -> int:
    _hook_start = time.time()
    rc = 0
    try:
        data = json.loads(sys.stdin.read() or "{}")
        tool_name: str = data.get("tool_name", "")
        tool_input: dict = data.get("tool_input") or {}

        bypass = os.environ.get("MULLET_GUARD_BYPASS", "").strip() == "1"

        sf = _session_file(data)
        state = _load_state(sf)

        # Delegation resets the counter
        if tool_name in DELEGATION_TOOLS:
            state["manual_tools"] = 0
            state["last_delegation_ts"] = int(time.time())
            state["history"].append({"tool": tool_name, "action": "reset", "ts": int(time.time())})
            _save_state(sf, state)
            _emit_runtime(_hook_start)
            return 0

        # Only count tools in the counted set
        if tool_name not in COUNTED_TOOLS:
            _emit_runtime(_hook_start)
            return 0

        # Allow-list certain Bash commands
        if tool_name == "Bash" and _is_allowlisted_bash(tool_input):
            _emit_runtime(_hook_start)
            return 0

        # Increment counter
        state["manual_tools"] += 1
        state["history"].append({"tool": tool_name, "ts": int(time.time())})
        # Keep history manageable
        state["history"] = state["history"][-50:]
        _save_state(sf, state)

        count = state["manual_tools"]

        if count > THRESHOLD:
            # Blocking exists to force delegation, and delegation spends real
            # money at a third-party provider. With the Anthropic allowance
            # comfortable there is nothing to save, so working in-context is the
            # cheaper option and the block would only get in the way.
            warranted, _gate_reason = delegation_warranted()
            if not warranted:
                _emit_runtime(_hook_start)
                return 0

            # Collect distinct tool names from recent history for the message
            recent_tools = list(dict.fromkeys(
                e["tool"] for e in state["history"] if e.get("tool") in COUNTED_TOOLS
            ))
            tool_names_str = ", ".join(recent_tools) if recent_tools else tool_name

            if _is_mcp_available() or _mcp_seen_recently(state):
                msg = BLOCK_TEMPLATE_MCP.format(n=count, tool_names=tool_names_str)
            else:
                msg = BLOCK_TEMPLATE_FALLBACK.format(n=count, tool_names=tool_names_str)

            if bypass:
                print(f"[manual-tool guard] BYPASS active — would have blocked at {count} calls.", file=sys.stderr)
                _emit_runtime(_hook_start)
                return 0

            print(msg, file=sys.stderr)
            rc = 2

    except Exception as exc:
        print(f"[manual-tool guard] error (fail-open): {exc}", file=sys.stderr)

    _emit_runtime(_hook_start)
    return rc


if __name__ == "__main__":
    sys.exit(main())
