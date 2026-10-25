#!/usr/bin/env python3
"""Mullet HUD — Claude Code statusline renderer.

Reads JSON from stdin (piped by Claude Code) and renders a statusline.

Structure: ``main(deps=HudDeps())`` — every external boundary (stdin,
process scan, render, now, print) is a callable on :class:`HudDeps` so tests
drive the entire entry point with in-memory stubs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace as _replace_dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from mullet.hud import render_hud
from mullet.hud.agent_state import cleanup_stale_agents
from mullet.hud.state import build_context
from mullet.hud.stdin_limits import fill_missing_windows, parse_stdin_limits
from mullet.hud.transcript import (
    TranscriptSnapshot,
    _infer_provider_from_model,
    parse_transcript,
)
from mullet.hud.types import HudConfig, HudRenderContext
from mullet.hud.usage import RateLimits, fetch_usage, parse_resets_at, record_usage

logger = logging.getLogger(__name__)

_DEFAULT_COLS = 120


def _default_read_stdin() -> dict:
    try:
        if sys.stdin.isatty():
            return {}
        raw = sys.stdin.read().strip()
        if not raw:
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return {}


def _default_get_terminal_cols() -> int:
    # shutil.get_terminal_size tries fd 1 then fd 2 (stderr stays TTY even when
    # stdout is piped, which is the case when running as a Claude Code hook).
    return shutil.get_terminal_size((_DEFAULT_COLS, 24)).columns


def _desc_from_subcommand(cmd: str) -> str:
    """Build a human-readable description from a mullet subcommand line."""
    # Strip leading path so we match on bare subcommand tokens
    tokens = cmd.split()
    # Find the index of the 'mullet' token so we can read what follows
    for i, tok in enumerate(tokens):
        if re.search(r"(?:^|/)mullet$", tok):
            sub = tokens[i + 1:] if i + 1 < len(tokens) else []
            break
    else:
        sub = tokens

    if not sub:
        return "running"

    # Strip leading flags to find the first real subcommand word
    first = next((t for t in sub if not t.startswith("-")), sub[0])

    _VERB_MAP = {
        "reindex": ("reindexing", "--pool"),
        "sync": ("syncing", "--pool"),
        "ingest": ("ingesting", None),
        "crawl": ("crawling", None),
        "build": ("building", None),
        "prune": ("pruning", "--pool"),
        "export": ("exporting", "--pool"),
        "status": ("status", None),
        "budget": ("budget check", None),
        "routing": ("routing", None),
        "run": ("running", None),
        "delegate": ("delegating", None),
        "swarm": ("swarming", None),
        "search": ("searching", None),
        "trawl": ("trawl", None),
        "observations": ("observations", None),
        "decisions": ("decisions", None),
        "review-session": ("review-session", None),
    }

    verb, pool_flag = _VERB_MAP.get(first, (first, None))

    # Try to extract a pool/target name
    target = None
    if pool_flag:
        m = re.search(rf"{re.escape(pool_flag)}\s+(\S+)", cmd)
        if m:
            target = m.group(1)
    if target is None:
        # Fall back to the first non-flag argument after the subcommand
        remaining = [t for t in sub[1:] if not t.startswith("-")]
        if remaining:
            target = remaining[0][:120]

    return f"{verb} {target}" if target else verb


def _default_scan_processes() -> list[dict]:
    """Scan ``ps aux`` for running Mullet CLI invocations.

    Returns a list of ``{provider, model, description}`` dicts. Swallows all
    subprocess errors — the HUD must never crash because ``ps`` is slow.
    """
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=2)
    except Exception as exc:
        logger.warning("hud_cli: failed to detect running agents: %s", exc)
        return []

    agents: list[dict] = []
    for line in ps.stdout.splitlines():
        if "mullet" not in line:
            continue
        if any(skip in line for skip in ("ps aux", "/bin/bash", "mullet-hud", "mullet mcp", "pip install")):
            continue
        cols = line.split(None, 10)
        if len(cols) < 10:
            continue
        pid_str, stat = cols[1], cols[7]
        if stat.startswith("Z"):  # zombie — parent hasn't reaped it yet
            continue
        try:
            os.kill(int(pid_str), 0)
        except ProcessLookupError:
            continue  # process exited between ps and liveness check
        except (ValueError, PermissionError):
            pass  # EPERM → exists but unowned; include it
        cmd_part = cols[-1]
        if not re.search(r"\bmullet\b", cmd_part):
            continue
        is_mullet_cmd = bool(
            re.search(r"(?:^|/)mullet(?:\s|$)", cmd_part)
            or ("python" in cmd_part and re.search(r"\bmullet\b(?!\S*[/.])", cmd_part))
        )
        if not is_mullet_cmd:
            continue
        m_prov = re.search(r"(?:--provider|-p)\s+(\S+)", line)
        m_mod = re.search(r"--model\s+(\S+)", line)
        desc = ""
        for quote in ('"', "'"):
            match = re.search(rf"{quote}([^{quote}]{{5,80}}){quote}", line)
            if match:
                desc = match.group(1)[:160]
                break
        if not desc:
            desc = _desc_from_subcommand(cmd_part)
        agents.append({
            "provider": m_prov.group(1) if m_prov else "local",
            "model": m_mod.group(1) if m_mod else "default",
            "description": desc,
        })
    return agents


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class HudDeps:
    """Injected boundaries for :func:`main`. Each field defaults to the real
    production implementation; tests pass stubs.
    """
    read_stdin: Callable[[], dict] = field(default=_default_read_stdin)
    parse_stdin_limits: Callable[[dict], Optional[RateLimits]] = field(default=parse_stdin_limits)
    fetch_usage_fallback: Callable[[], Optional[RateLimits]] = field(default=fetch_usage)
    record_usage: Callable[[RateLimits], None] = field(default=record_usage)
    scan_processes: Callable[[], list[dict]] = field(default=_default_scan_processes)
    parse_transcript: Callable[[str], TranscriptSnapshot] = field(default=parse_transcript)
    build_context: Callable[[str], HudRenderContext] = field(default=build_context)
    render: Callable[[HudRenderContext, HudConfig], str] = field(default=render_hud)
    now: Callable[[], datetime] = field(default=_default_now)
    get_terminal_cols: Callable[[], int] = field(default=_default_get_terminal_cols)
    emit: Callable[[str], None] = field(default=print)


def _compute_context_pct(ctx_window: dict) -> float:
    """Prefer stdin's ``used_percentage``; fall back to ``current_usage/size``.

    Mirrors claude-hud's getContextPercent so the HUD matches CC's own display.
    """
    pct = ctx_window.get("used_percentage", 0.0) or 0.0
    if pct > 0:
        return float(pct)
    size = ctx_window.get("context_window_size", 0) or 0
    current = ctx_window.get("current_usage", {}) or {}
    total = sum(int(v) for v in current.values() if isinstance(v, (int, float)))
    if size and total:
        return min(100.0, round(total / size * 100, 1))
    return 0.0


def _sum_current_usage(ctx_window: dict) -> int:
    current = ctx_window.get("current_usage", {}) or {}
    return sum(int(v) for v in current.values() if isinstance(v, (int, float)))


def _apply_limits(ctx: HudRenderContext, limits: RateLimits, now: datetime) -> None:
    ctx.five_hour_pct = limits.five_hour_pct
    ctx.weekly_pct = limits.weekly_pct
    ctx.sonnet_weekly_pct = limits.sonnet_weekly_pct
    reset_dt = parse_resets_at(limits.five_hour_resets)
    if reset_dt is not None:
        delta = reset_dt - now
        ctx.rate_limit_reset_s = max(0, int(delta.total_seconds()))


def build_hud_output(stdin_data: dict, deps: HudDeps) -> str:
    """Pure function: stdin payload → rendered HUD string. No I/O except via deps."""
    # Purge stale entries from agents.json so crashed/killed agents don't linger
    try:
        cleanup_stale_agents(max_age_seconds=600)
    except Exception:
        pass
    cwd = stdin_data.get("cwd", os.getcwd())
    model_raw = stdin_data.get("model", {}) or {}
    model = {"id": model_raw} if isinstance(model_raw, str) else model_raw
    model_name = model.get("display_name") or model.get("id") or ""
    # stdin may carry provider explicitly (Mullet's own hook stack does) but CC
    # itself does not — infer from the model id in that case.
    model_provider = model.get("provider") or _infer_provider_from_model(model_name) or ""
    ctx_window = stdin_data.get("context_window", {}) or {}

    ctx = deps.build_context(cwd)
    ctx.model_name = model_name
    ctx.main_provider = model_provider
    ctx.context_pct = _compute_context_pct(ctx_window)

    if ctx.total_tokens == 0:
        total = _sum_current_usage(ctx_window)
        if total:
            ctx.total_tokens = total

    limits: Optional[RateLimits] = None
    try:
        limits = deps.parse_stdin_limits(stdin_data)
    except Exception as exc:
        logger.warning("hud_cli: stdin rate-limit parse failed: %s", exc)
    if limits is not None:
        # The budget guard, router and swarm read these back with fetch_usage().
        try:
            deps.record_usage(fill_missing_windows(
                _replace_dataclass(limits), stdin_data, deps.fetch_usage_fallback()))
        except Exception as exc:
            logger.warning("hud_cli: failed to record usage limits: %s", exc)
    else:
        try:
            limits = deps.fetch_usage_fallback()
        except Exception as exc:
            logger.warning("hud_cli: failed to fetch usage limits: %s", exc)
    if limits is not None:
        _apply_limits(ctx, limits, deps.now())

    # --- Transcript wiring ---------------------------------------------------
    transcript_path = stdin_data.get("transcript_path")
    snapshot: Optional[TranscriptSnapshot] = None
    if isinstance(transcript_path, str):
        try:
            snapshot = deps.parse_transcript(transcript_path)
        except Exception as exc:
            logger.warning("hud_cli: transcript parse failed: %s", exc)

    if snapshot is not None:
        # Running agents from Task/Agent tool_use blocks. Prefer the provider
        # inferred from the subagent's own model (AgentCall.provider) over the
        # default "cc" tag — a Kimi-delegated subagent should read "kimi" not
        # "cc", matching what Mullet's scan_processes path already does.
        for a in snapshot.agents:
            if a.status == "running":
                ctx.running_agents.append({
                    "provider": a.provider or "cc",
                    "model": a.model or "?",
                    "description": a.description or a.agent_type,
                })

        # Running tools — carry provider/model so every pill renders the badge.
        ctx.running_tools = [
            {
                "name": t.name,
                "target": t.target,
                "started_at": t.start_time.isoformat(),
                "provider": t.provider,
                "model": t.model,
            }
            for t in snapshot.running_tools
        ]

        # Last 5 completed tools with duration
        last5 = snapshot.completed_tools[-5:]
        ctx.completed_tools = []
        for t in last5:
            if t.end_time is not None:
                duration_ms = int((t.end_time - t.start_time).total_seconds() * 1000)
            else:
                duration_ms = 0
            ctx.completed_tools.append({
                "name": t.name,
                "target": t.target,
                "duration_ms": duration_ms,
                "provider": t.provider,
                "model": t.model,
            })

        # Todos
        if snapshot.todos:
            ctx.todos_total = len(snapshot.todos)
            ctx.todos_done = sum(1 for td in snapshot.todos if td.status == "completed")

        # Session tokens
        ctx.session_tokens_in = snapshot.session_tokens_in
        ctx.session_tokens_out = snapshot.session_tokens_out

        # total_tokens — only if still 0 after stdin wiring
        if ctx.total_tokens == 0:
            total = (
                snapshot.session_tokens_in
                + snapshot.session_tokens_out
                + snapshot.session_cache_creation_tokens
                + snapshot.session_cache_read_tokens
            )
            if total:
                ctx.total_tokens = total

        # session_minutes — only if still 0
        if ctx.session_minutes == 0.0 and snapshot.session_started_at is not None:
            delta = deps.now() - snapshot.session_started_at
            ctx.session_minutes = delta.total_seconds() / 60.0

    # Process scan — only fall through when transcript provides no running agents
    if snapshot is None or not any(a.status == "running" for a in snapshot.agents):
        for agent in deps.scan_processes():
            ctx.running_agents.append(agent)

    config = HudConfig(preset="statusline", max_width=deps.get_terminal_cols())
    return deps.render(ctx, config)


def main(deps: Optional[HudDeps] = None) -> None:
    """Entry point for the ``mullet-hud`` CLI — reads stdin and emits a status line."""
    if deps is None:
        deps = HudDeps()
    stdin_data = deps.read_stdin()
    output = build_hud_output(stdin_data, deps)
    deps.emit(output)


if __name__ == "__main__":
    main()
