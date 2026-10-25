#!/usr/bin/env python3
"""Claude Code PreToolUse hook — rewrite mcp__mullet__* tool input via the Mullet router.

Intercepts calls to the Mullet MCP tools (`mullet_run`, `mullet_delegate`,
`mullet_swarm`) before they dispatch, consults the project router + budget
state, and emits `updatedInput` that:

  * respects the active session override (`mullet override on`) by pinning
    Opus via `providers` + `tier="PREMIUM"`,
  * fills in a sensible `tier` when the caller left it unset, picked from the
    current budget pressure (via `recommend_provider`),
  * respects every user-set field verbatim — we only fill gaps, never
    overwrite an explicit choice,
  * fails open: any exception emits nothing and lets the original input
    dispatch unchanged. If Mullet's own modules can't be imported, it says so
    on stderr and exits 1 (a non-blocking error), still without a rewrite.

Run as `mullet hook pretooluse-rewriter` (PreToolUse, matcher
`mcp__mullet__.*`).

Protocol (Claude Code PreToolUse hook):
  stdin:  {"tool_name": "...", "tool_input": {...}, "session_id": "..."}
  stdout: {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                  "permissionDecision": "allow",
                                  "updatedInput": {...}}}
  exit:   0, or 1 when Mullet's modules can't be imported
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

TARGET_TOOLS = {
    "mcp__mullet__mullet_run",
    "mcp__mullet__mullet_delegate",
    "mcp__mullet__mullet_swarm",
}

# Pressure → tier mapping for auto-fill (matches recommend_provider ladder)
_TIER_FOR_PRESSURE = [
    (0.30, "ADVANCED"),  # low pressure → allow Sonnet
    (0.55, "STANDARD"),  # moderate → Haiku tier
    (0.80, "LOW"),       # high → cheap
]


def _emit_runtime(hook_start: float) -> None:
    """Emit a one-line runtime metric to stderr when >10ms."""
    elapsed_ms = int((time.time() - hook_start) * 1000)
    if elapsed_ms > 10:
        print(f"[hook:{Path(__file__).stem}] {elapsed_ms}ms", file=sys.stderr)


def _pressure_to_tier(pressure: float) -> str:
    for cutoff, tier in _TIER_FOR_PRESSURE:
        if pressure < cutoff:
            return tier
    return "LOW"


def _get_session_override() -> dict | None:
    """Return the active session override or None. Fail-open on IO errors.

    An ImportError is not caught: main() reports it instead of hiding it.
    """
    from mullet.core.session_state import get_override
    try:
        return get_override()
    except Exception:
        return None


def _get_budget_pressure() -> float | None:
    """Return current budget pressure (0.0-1.0) or None if unavailable.

    An ImportError is not caught: main() reports it instead of hiding it.
    """
    from mullet.hud.guard import check_budget
    try:
        b = check_budget()
        return float(b.pressure)
    except Exception:
        return None


def _rewrite_input(tool_name: str, tool_input: dict) -> dict | None:
    """Return a rewritten `tool_input`, or None to pass through unchanged.

    Only modifies fields the caller has not set. An explicit user choice
    (any truthy value on `providers` / `tier`) is preserved verbatim.
    """
    if tool_name not in TARGET_TOOLS:
        return None
    if not isinstance(tool_input, dict):
        return None

    updated: dict = {}
    reasons: list[str] = []

    has_providers = bool(tool_input.get("providers"))
    has_tier = bool(tool_input.get("tier"))

    override = _get_session_override()
    if override and override.get("provider"):
        # User has explicitly pinned Opus — promote PREMIUM tier + opus-first
        # provider list IF the caller hasn't overridden either field.
        pinned_provider = override["provider"]
        if not has_providers:
            updated["providers"] = f"{pinned_provider},sonnet,haiku"
            reasons.append(f"override:{pinned_provider}")
        if not has_tier:
            updated["tier"] = "PREMIUM"
            reasons.append("tier:PREMIUM")
        if updated:
            if reasons:
                print(f"[mullet pretooluse] {tool_name}: {','.join(reasons)}",
                      file=sys.stderr)
            merged = dict(tool_input)
            merged.update(updated)
            return merged
        return None

    # No override — auto-fill tier only (never providers, to keep default
    # Kimi-first delegation intact).
    if has_tier:
        return None

    pressure = _get_budget_pressure()
    if pressure is None:
        return None
    tier = _pressure_to_tier(pressure)

    reasons.append(f"tier:{tier}(pressure={pressure:.2f})")
    merged = dict(tool_input)
    merged["tier"] = tier
    print(f"[mullet pretooluse] {tool_name}: {','.join(reasons)}",
          file=sys.stderr)
    return merged


def _emit_decision(updated_input: dict | None) -> None:
    """Write the PreToolUse hookSpecificOutput to stdout.

    When no rewrite is needed we emit nothing and exit 0 — Claude Code
    treats absence of hookSpecificOutput as implicit allow + unchanged input.
    """
    if updated_input is None:
        return
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated_input,
        }
    }
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")


def main() -> int:
    hook_start = time.time()
    try:
        raw = sys.stdin.read() or "{}"
        data = json.loads(raw)
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input") or {}
        if os.environ.get("MULLET_DISABLE_REWRITER", "").strip() in ("1", "true", "yes"):
            return 0
        updated = _rewrite_input(tool_name, tool_input)
        _emit_decision(updated)
    except ImportError as exc:
        # Still no rewrite, so the call goes ahead unchanged, but say why.
        print(f"[mullet pretooluse] can't import Mullet's routing state: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Fail open — never block a tool dispatch because of hook error.
        print(f"[mullet pretooluse] error (fail-open): {exc}", file=sys.stderr)
    finally:
        _emit_runtime(hook_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
