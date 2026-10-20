#!/usr/bin/env python3
"""Claude Code SessionStart hook: report active Mullet preset and nudge confirmation.

Contract: Claude Code passes a JSON object on stdin (ignored).
The hook must print EXACTLY one JSON object of this shape:
    {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "<text>"}}
Fully fail-open: on ANY error, print nothing (empty stdout) and exit 0.
"""

import json
import os
import sys
from pathlib import Path


def main():
    try:
        # Consume stdin JSON (ignored per contract)
        sys.stdin.read()

        # Ensure mullet package is importable when run from project root
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
        if project_dir and project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        from mullet.core.presets import PresetManager

        preset = PresetManager().active()

        allowed = (
            ",".join(preset.allowed_providers)
            if preset.allowed_providers is not None
            else "all"
        )
        opencode = (
            "native"
            if preset.native_only or preset.opencode_model is None
            else preset.opencode_model
        )

        text = (
            f"[Mullet preset] Active provider preset: '{preset.name}' ({preset.label}) — "
            f"allowed_providers={allowed}, opencode={opencode}. "
            f"If the user has not chosen a preset this session, ask them which preset to use "
            f"(local-only, claude-only, balanced-mix, cheap-cloud) before delegating, "
            f"then run `mullet preset use <name>`."
        )

        output = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": text,
            }
        }

        # Print exactly one JSON object
        print(json.dumps(output))
    except Exception:
        # Fail-open: empty stdout, exit 0 — a broken hook never blocks a session
        pass


if __name__ == "__main__":
    main()
