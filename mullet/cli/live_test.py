"""Live provider visibility test.

Fires a **minimal** real LLM call per available ``(provider, model)`` pair
and holds each HUD agent row for the caller-supplied number of seconds.

Each thread:

1. Registers a HUD row via :func:`mullet.hud.agent_state.add_agent`.
2. Sends a trivial prompt (<10 output tokens) to prove the wire is live.
3. Sleeps for the remaining hold window.
4. Removes the row.

That keeps the HUD populated for the full duration the caller asked for,
while spending the absolute minimum on tokens. Use it when you want to
see every provider/model stacked on the HUD concurrently.

Usage::

    mullet live-test 60              # hold for 60 seconds
    mullet live-test 30 --no-swarm   # skip the Kimi fan-out
    mullet live-test 60 --swarm-size 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from mullet.core.providers.base import Provider
from mullet.hud.agent_state import add_agent, remove_agent

logger = logging.getLogger(__name__)

_DEFAULT_SECONDS = 60
_DEFAULT_SWARM_WORKERS = 3

_MCP_ENV_KEYS = (
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "KIMI_BASE_URL",
)

# Trimmed to one canonical model per provider — no deprecated ids, no moonshot-v1
# models that spam constraint-healing warnings on every tick.
_KIMI_MODELS: tuple[str, ...] = ("kimi-k2.7-code",)
_GEMINI_MODELS: tuple[str, ...] = ("gemini-2.5-flash",)
_ANTHROPIC_MODELS: tuple[str, ...] = (
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-8",
)

_TRIVIAL_PROMPT = "Reply with just: ok"
_TRIVIAL_MAX_TOKENS = 5


def _load_mcp_env_fallback() -> None:
    """Populate missing provider keys from ``~/.mcp.json`` when the shell env is empty."""
    path = Path.home() / ".mcp.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    servers = data.get("mcpServers") or data.get("servers") or {}
    mullet_cfg = servers.get("mullet")
    if not isinstance(mullet_cfg, dict):
        return
    env = mullet_cfg.get("env") or {}
    for key in _MCP_ENV_KEYS:
        if key not in os.environ and isinstance(env.get(key), str) and env[key]:
            os.environ[key] = env[key]


@dataclass
class Target:
    """A provider/model target for live testing."""
    provider_label: str
    model_id: str
    build: Callable[[], Provider]


def _discover_targets() -> List[Target]:
    """Return ``(provider, model, factory)`` triples for every reachable LLM."""
    _load_mcp_env_fallback()
    targets: List[Target] = []

    kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    if kimi_key:
        from mullet.core.providers.kimi import KimiProvider

        def _kimi_factory(model_id: str, key: str = kimi_key):
            return lambda: KimiProvider(api_key=key, model=model_id)

        for m in _KIMI_MODELS:
            targets.append(Target("kimi", m, _kimi_factory(m)))

    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            from mullet.core.providers.gemini import GeminiProvider

            def _gemini_factory(model_id: str, key: str = gemini_key):
                return lambda: GeminiProvider(api_key=key, model=model_id)

            for m in _GEMINI_MODELS:
                targets.append(Target("gemini", m, _gemini_factory(m)))
        except Exception as exc:
            logger.debug("live-test: Gemini provider unavailable: %s", exc)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        from mullet.core.providers.anthropic import AnthropicProvider

        def _anthropic_factory(model_id: str, key: str = anthropic_key):
            return lambda: AnthropicProvider(api_key=key, model=model_id)

        for m in _ANTHROPIC_MODELS:
            targets.append(Target("anthropic", m, _anthropic_factory(m)))

    try:
        from mullet.core.providers.local_detection import detect_local_provider
        probe = detect_local_provider()
        if probe is not None:
            local_model = getattr(probe, "model", "") or "default"
            def _local_factory(p=probe):
                return lambda: p
            targets.append(Target("local", local_model, _local_factory()))
    except Exception as exc:
        logger.debug("live-test: local detection failed: %s", exc)

    return targets


def _fire_one(target: Target, hold_seconds: int, results: dict) -> None:
    """Register HUD row, ping the wire with a trivial prompt, hold, then clear.

    Writes a ``{provider/model: {'ok': bool, 'error_class': str, 'error': str,
    'duration': float}}`` entry into ``results`` so the caller can emit a
    health summary.
    """
    agent_id = uuid.uuid4().hex[:12]
    description = f"live-test on {target.model_id}"
    key = f"{target.provider_label}/{target.model_id}"
    add_agent(agent_id, description, target.provider_label, target.model_id)
    start = time.monotonic()
    status_line = "?"
    error_class = ""
    error_detail = ""
    ok = False
    try:
        try:
            provider = target.build()
            provider.create(
                messages=[{"role": "user", "content": _TRIVIAL_PROMPT}],
                max_tokens=_TRIVIAL_MAX_TOKENS,
                temperature=0.0,
            )
            ok = True
            status_line = "\u2713"
        except Exception as exc:
            error_class = exc.__class__.__name__
            # Keep the first line of the exception message — that's the useful
            # bit. SDK errors often include a big JSON body on subsequent lines.
            raw = str(exc)
            error_detail = raw.splitlines()[0][:220] if raw else ""
            status_line = f"\u2717 {error_class}"
        elapsed = time.monotonic() - start
        remaining = hold_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
    finally:
        try:
            remove_agent(agent_id)
        except Exception:
            pass
        dur = time.monotonic() - start
        results[key] = {
            "ok": ok,
            "error_class": error_class,
            "error": error_detail,
            "duration": dur,
        }
        print(f"  {key} {status_line} {dur:.1f}s")
        if error_detail:
            print(f"      └ {error_detail}")


def _fire_kimi_swarm_holds(seconds: int, size: int) -> None:
    """Register ``size`` Kimi swarm-worker HUD rows (no real fan-out call)."""
    threads: List[threading.Thread] = []
    for i in range(size):

        def _hold(idx: int = i):
            agent_id = uuid.uuid4().hex[:12]
            try:
                add_agent(agent_id, f"swarm-worker-{idx + 1}", "kimi", "kimi-k2.7-code")
                time.sleep(seconds)
            finally:
                try:
                    remove_agent(agent_id)
                except Exception:
                    pass

        t = threading.Thread(target=_hold, daemon=False, name=f"live-swarm-{i}")
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_live_test(
    seconds: int = _DEFAULT_SECONDS,
    *,
    include_swarm: bool = True,
    swarm_size: int = _DEFAULT_SWARM_WORKERS,
    include_local: bool = True,
) -> int:
    """Spawn one thread per provider/model and hold the HUD for ``seconds``."""
    targets = _discover_targets()
    if not include_local:
        targets = [t for t in targets if t.provider_label != "local"]
    if not targets:
        print(
            "live-test: no providers configured — set KIMI_API_KEY / GOOGLE_API_KEY /"
            " ANTHROPIC_API_KEY, sign in to Claude Code, or start LM Studio."
        )
        return 1

    results: dict = {}
    threads: List[threading.Thread] = []
    for target in targets:
        threads.append(
            threading.Thread(
                target=_fire_one,
                args=(target, seconds, results),
                daemon=False,
                name=f"live-{target.provider_label}-{target.model_id}",
            )
        )

    kimi_available = any(t.provider_label == "kimi" for t in targets)
    swarm_thread: Optional[threading.Thread] = None
    if include_swarm and kimi_available and swarm_size > 0:
        swarm_thread = threading.Thread(
            target=_fire_kimi_swarm_holds,
            args=(seconds, swarm_size),
            daemon=False,
            name="live-swarm",
        )
        threads.append(swarm_thread)

    providers = sorted({t.provider_label for t in targets})
    print(f"live-test: {len(targets)} real pings + {swarm_size if swarm_thread else 0} swarm-worker holds, hold={seconds}s")
    print(f"  providers: {', '.join(providers)}")
    print(f"  prompt:    {_TRIVIAL_PROMPT!r}  max_tokens={_TRIVIAL_MAX_TOKENS}")
    print("  (watch the statusline now)")
    print()

    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start
    print()
    print(f"live-test: done in {elapsed:.1f}s wall-clock")

    failures = {k: v for k, v in results.items() if not v["ok"]}
    successes = len(results) - len(failures)
    total = len(results)
    print()
    if failures:
        print(f"\u26a0  health: {len(failures)}/{total} provider(s) failed")
        for key, info in sorted(failures.items()):
            print(f"   \u2717 {key}  {info['error_class']}")
            if info["error"]:
                print(f"       {info['error']}")
        print()
        print("  hints:")
        print("   - RateLimitError on API-key providers: check quota / wait for reset")
        print("   - BadRequestError: model not exposed on this plan, or needs a beta header")
        print("   - APIConnectionError / TimeoutError: network / endpoint reachability")
        return 1

    print(f"\u2713  health: all {successes}/{total} providers responded")
    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mullet live-test",
        description="Hold one HUD row per provider/model for N seconds; trivial ping per row.",
    )
    parser.add_argument(
        "seconds",
        nargs="?",
        type=int,
        default=_DEFAULT_SECONDS,
        help=f"hold duration in seconds (default {_DEFAULT_SECONDS})",
    )
    parser.add_argument("--no-swarm", action="store_true", help="skip Kimi swarm-worker rows")
    parser.add_argument(
        "--swarm-size",
        type=int,
        default=_DEFAULT_SWARM_WORKERS,
        help=f"number of Kimi swarm workers (default {_DEFAULT_SWARM_WORKERS})",
    )
    parser.add_argument("--no-local", action="store_true", help="skip the local LLM row")
    return parser.parse_args(argv)


def run_cli(argv: List[str]) -> int:
    """Entry point for the live-test CLI."""
    args = _parse_args(argv)
    if args.seconds < 1:
        print("live-test: seconds must be >= 1", file=sys.stderr)
        return 2
    return run_live_test(
        seconds=args.seconds,
        include_swarm=not args.no_swarm,
        swarm_size=args.swarm_size,
        include_local=not args.no_local,
    )


if __name__ == "__main__":
    sys.exit(run_cli(sys.argv[1:]))
