#!/usr/bin/env python3
"""PostToolUse hook — capture every tool call to the observation log.

Reads Claude Code's PostToolUse JSON from stdin:
  {"session_id": "...", "tool_name": "...", "tool_input": {...}, "tool_response": ...}

Writes one JSONL line via mullet.core.observations.record().
Always exits 0 — observation capture must never block real tool calls.
Set MULLET_DISABLE_OBSERVATIONS=1 to skip entirely.
"""
import json
import os
import sys
import time


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
        # Project root: honour MULLET_OBSERVATIONS_BASE env override, else two levels up
        from pathlib import Path
        env_base = os.environ.get("MULLET_OBSERVATIONS_BASE")
        if env_base:
            project_root = Path(env_base)
        else:
            project_root = Path(__file__).resolve().parent.parent.parent

        # Add project to path so import works from hook context
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from mullet.core.observations import record
        record(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=output_str,
            session_id=session_id,
            base=project_root,
        )
    except Exception:
        pass  # fail open

    elapsed_ms = (time.monotonic() - t0) * 1000
    if elapsed_ms > 10:
        print(f"[hook:observation_capture] {elapsed_ms:.0f}ms", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
