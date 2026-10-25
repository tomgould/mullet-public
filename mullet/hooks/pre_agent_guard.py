#!/usr/bin/env python3
"""PreToolUse hook: intercept Task (Agent) tool calls and warn when no
model override is specified. Non-blocking — prints a warning to stdout
(visible to Claude) and exits 0.

Run as ``mullet hook pre-agent-guard`` (PreToolUse, matcher ``Task|Agent``).
"""

import json
import sys
import time
from pathlib import Path

CHEAP_MODELS = {
    "haiku",
    "sonnet",
    "haiku-4-5",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
}

WARNING = """[pre-agent guard] Task/Agent call detected without a cheap-model override.
Spawned agents inherit Opus by default — this burns your Anthropic quota
for work a cheap model can do.
Consider:
  - Pass model: "haiku" for read-and-report subagents (Explore, research, summarise)
  - Pass model: "sonnet" for multi-file reasoning
  - Or delegate via mcp__mullet__mullet_run with providers="kimi" instead of spawning an Agent
Proceeding anyway — this is a warning only."""


def _emit_runtime(hook_start: float) -> None:
    """Emit a one-line runtime metric to stderr when >10ms."""
    elapsed_ms = int((time.time() - hook_start) * 1000)
    if elapsed_ms > 10:
        print(f"[hook:{Path(__file__).stem}] {elapsed_ms}ms", file=sys.stderr)


def main() -> int:
    _hook_start = time.time()
    try:
        data = json.loads(sys.stdin.read() or "{}")
        # Claude Code calls the subagent tool "Agent"; older versions called it "Task".
        if data.get("tool_name") not in ("Task", "Agent"):
            _emit_runtime(_hook_start)
            return 0
        tool_input = data.get("tool_input") or {}
        model = tool_input.get("model")
        if model in CHEAP_MODELS:
            _emit_runtime(_hook_start)
            return 0
        print(WARNING)
    except Exception as exc:
        print(f"[hook:pre_agent_guard] error: {exc}", file=sys.stderr)
    _emit_runtime(_hook_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
