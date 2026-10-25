#!/usr/bin/env python3
"""Claude Code SessionStart hook: report active Mullet preset and nudge confirmation.

Contract: Claude Code passes a JSON object on stdin (ignored).
The hook must print EXACTLY one JSON object of this shape:
    {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "<text>"}}
Fail-open: on an error, print nothing on stdout, say what went wrong on stderr
and exit 0, so a session always starts. If Mullet's presets can't be imported
at all, exit 1, which Claude Code shows as a non-blocking error.

Run as `mullet hook preset-prompt` (SessionStart).
"""

import json
import sys


def main():
    try:
        from mullet.core.presets import PresetManager
    except ImportError as exc:
        print(f"[hook:preset_prompt] can't import Mullet's presets: {exc}", file=sys.stderr)
        return 1
    try:
        # Consume stdin JSON (ignored per contract)
        sys.stdin.read()

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
    except Exception as exc:
        # Fail-open: empty stdout, exit 0, so a broken hook never blocks a session.
        print(f"[hook:preset_prompt] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
