"""`mullet override` subcommand — manage the session-scoped provider override.

Usage:
    mullet override on [--model MODEL]   # activate override (default opus-4-8)
    mullet override off                  # clear the override
    mullet override status               # show current override state

The override lives at <cwd>/.mullet/session.json and auto-expires on the
Anthropic 5h window reset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Sequence

from mullet.core.session_state import (
    clear_override,
    get_override,
    set_override,
)


_PROVIDER_MODEL_DEFAULTS = {
    "opus-4-8": "claude-opus-4-8",
    "opus-4-7": "claude-opus-4-7",
    "opus": "claude-opus-4-8",   # plain "opus" → current best
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


def run(argv: Sequence[str]) -> int:
    """Entry point for `mullet override <action>`. Returns a shell-style exit code."""
    parser = argparse.ArgumentParser(
        prog="mullet override",
        description="Manage the session-scoped provider override.",
    )
    parser.add_argument("action", choices=["on", "off", "status"], nargs="?", default="status")
    parser.add_argument("--provider", default="opus-4-8",
                        help="Provider alias to pin (default: opus-4-8)")
    parser.add_argument("--model", default=None,
                        help="Explicit model id — overrides the default for the chosen provider")
    args = parser.parse_args(list(argv))

    if args.action == "on":
        model = args.model or _PROVIDER_MODEL_DEFAULTS.get(args.provider)
        set_override(provider=args.provider, model=model)
        print(f"[Opus override] activated — routing {args.provider}"
              f"{' (' + model + ')' if model else ''}"
              " until 5h reset or `mullet override off`")
        return 0

    if args.action == "off":
        clear_override()
        print("[Opus override] cleared — returning to default routing")
        return 0

    state = get_override()
    if not state:
        print("[Opus override] not active — default routing in effect")
        return 0
    expires_at = state.get("expires_at", "?")
    try:
        expires_dt = datetime.fromisoformat(expires_at)
        now = datetime.now(timezone.utc)
        mins = max(int((expires_dt - now).total_seconds() / 60), 0)
        expiry_hint = f"{mins} min remaining"
    except (ValueError, TypeError):
        expiry_hint = f"expires {expires_at}"
    print(f"[Opus override] active — provider={state.get('provider')} "
          f"model={state.get('model') or 'default'} ({expiry_hint})")
    return 0
