#!/usr/bin/env python3
"""PostToolUse hook — capture every tool call to the observation log.

Reads Claude Code's PostToolUse JSON from stdin:
  {"session_id": "...", "tool_name": "...", "tool_input": {...}, "tool_response": ...}

Writes one JSONL line via mullet.core.observations.record(), under
.mullet/observations/ in $MULLET_OBSERVATIONS_BASE, else the Claude Code
project ($CLAUDE_PROJECT_DIR), else the current directory.
Never blocks a tool call: if Mullet's observation log can't be imported it says
so on stderr and exits 1. Set MULLET_DISABLE_OBSERVATIONS=1 to skip entirely.
Run as `mullet hook observation-capture` (PostToolUse).
"""
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    if os.environ.get("MULLET_DISABLE_OBSERVATIONS"):
        sys.exit(0)

    t0 = time.monotonic()

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
    except Exception:
        sys.exit(0)

    session_id = payload.get("session_id", "unknown")
    tool_name = payload.get("tool_name", "unknown")
    tool_input = payload.get("tool_input", {})
    tool_response = payload.get("tool_response", "")

    # tool_response may be a dict or a string
    if isinstance(tool_response, (dict, list)):
        output_str = json.dumps(tool_response, ensure_ascii=False)
    else:
        output_str = str(tool_response) if tool_response is not None else ""

    if not isinstance(tool_input, dict):
        tool_input = {"value": str(tool_input)}

    try:
        from mullet.core.observations import record
    except ImportError as exc:
        print(f"[hook:observation_capture] can't import Mullet's observation log: {exc}",
              file=sys.stderr)
        sys.exit(1)

    base = os.environ.get("MULLET_OBSERVATIONS_BASE") or os.environ.get("CLAUDE_PROJECT_DIR")
    try:
        record(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=output_str,
            session_id=session_id,
            base=Path(base) if base else None,
        )
    except Exception as exc:
        # Fail open: a logging failure must never get in the way of the tool call.
        print(f"[hook:observation_capture] error: {exc}", file=sys.stderr)

    elapsed_ms = (time.monotonic() - t0) * 1000
    if elapsed_ms > 10:
        print(f"[hook:observation_capture] {elapsed_ms:.0f}ms", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
