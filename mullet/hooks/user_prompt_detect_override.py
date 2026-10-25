"""Claude Code UserPromptSubmit hook: detect natural-language "use Opus" overrides.

Reads the user prompt from stdin JSON, applies regex patterns to detect requests
to enable or disable an Opus provider override, then calls set_override /
clear_override from mullet.core.session_state. Prints a systemMessage hint on
state change. Never blocks the prompt: if Mullet's session state can't be
imported it says so on stderr and exits 1.

Run as `mullet hook opus-override` (UserPromptSubmit).
"""

import json
import os
import re
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Matches action-verb + "opus" (optionally with version).
# More specific patterns (4.6, 4.7) are checked before the generic catch-all.
_RE_ENABLE_48 = re.compile(
    r"\b(?:use|switch\s+to|enable|activate)\s+opus(?:\s+4\.?8)?\b",
    re.IGNORECASE,
)
_RE_ENABLE_47 = re.compile(
    r"\b(?:use|switch\s+to|enable|activate)\s+opus\s+4\.?7\b",
    re.IGNORECASE,
)
_RE_ENABLE_46 = re.compile(
    r"\b(?:use|switch\s+to|enable|activate)\s+opus\s+4\.?6\b",
    re.IGNORECASE,
)
_RE_DISABLE = re.compile(
    r"\b(?:stop\s+using|disable|deactivate|turn\s+off)\s+opus\b"
    r"|"
    r"\b(?:back\s+to|return\s+to)\s+(?:default|normal|routing)\b",
    re.IGNORECASE,
)


def _emit_runtime(hook_start: float) -> None:
    elapsed_ms = int((time.time() - hook_start) * 1000)
    if elapsed_ms > 10:
        print(f"[hook:{Path(__file__).stem}] {elapsed_ms}ms", file=sys.stderr)


def main() -> None:
    _hook_start = time.time()
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw)
        prompt: str = input_data.get("prompt", input_data.get("user_prompt", ""))

        state_dir_env = os.environ.get("MULLET_STATE_DIR", "")

        try:
            from mullet.core.session_state import clear_override, set_override
        except ImportError as exc:
            print(f"[hook:user_prompt_detect_override] can't import Mullet's session state: {exc}",
                  file=sys.stderr)
            _emit_runtime(_hook_start)
            sys.exit(1)

        state_dir = Path(state_dir_env) if state_dir_env else None

        # --- Disable check first (more specific than enable) ---
        if _RE_DISABLE.search(prompt):
            clear_override(state_dir=state_dir)
            msg = "[Opus override] cleared — returning to default routing"
            print(json.dumps({"systemMessage": msg}))
            _emit_runtime(_hook_start)
            sys.exit(0)

        # --- Enable opus-4-6 (explicit version pin) ---
        if _RE_ENABLE_46.search(prompt):
            set_override("opus-4-6", model="claude-opus-4-6", state_dir=state_dir)
            model = "claude-opus-4-6"
            msg = (
                f"[Opus override] activated — routing {model} until 5h reset "
                "or `mullet override off`"
            )
            print(json.dumps({"systemMessage": msg}))
            _emit_runtime(_hook_start)
            sys.exit(0)

        # --- Enable opus-4-7 (explicit version pin) ---
        if _RE_ENABLE_47.search(prompt):
            set_override("opus-4-7", model="claude-opus-4-7", state_dir=state_dir)
            model = "claude-opus-4-7"
            msg = (
                f"[Opus override] activated — routing {model} until 5h reset "
                "or `mullet override off`"
            )
            print(json.dumps({"systemMessage": msg}))
            _emit_runtime(_hook_start)
            sys.exit(0)

        # --- Enable opus-4-8 (current best — default when no version specified) ---
        if _RE_ENABLE_48.search(prompt):
            set_override("opus", model="claude-opus-4-8", state_dir=state_dir)
            model = "claude-opus-4-8"
            msg = (
                f"[Opus override] activated — routing {model} until 5h reset "
                "or `mullet override off`"
            )
            print(json.dumps({"systemMessage": msg}))
            _emit_runtime(_hook_start)
            sys.exit(0)

    except Exception as exc:
        print(f"[hook:user_prompt_detect_override] error: {exc}", file=sys.stderr)

    _emit_runtime(_hook_start)
    sys.exit(0)


if __name__ == "__main__":
    main()
