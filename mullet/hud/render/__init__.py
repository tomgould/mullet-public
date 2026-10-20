"""Compositional HUD renderer.

Public API
----------
render_hud(ctx, config)   -> str         Back-compat single-string API.
render_lines(ctx, config) -> list[str]   One string per logical HUD line.
"""
from typing import Optional
import pathlib
import time

from mullet.hud.types import HudConfig, HudRenderContext, HudElementConfig, PRESETS
from mullet.hud.colors import cyan, dim, green, red, yellow, magenta

from mullet.hud.render.wrap import wcwidth_truncate, split_wrap_parts
from mullet.hud.render.lines.project import render_project_line
from mullet.hud.render.lines.usage import render_usage_line
from mullet.hud.render.lines.tools import render_tools_line
from mullet.hud.render.lines.agents import render_agents_line
from mullet.hud.render.lines.todos import render_todos_line
from mullet.hud.render.lines.env import render_env_line

# Legacy element renderers — kept for back-compat with callers that import them
from mullet.hud.elements import (
    render_context, render_agents, render_mode, render_todos,
    render_session, render_git, render_model, render_tokens,
    render_rate_limits, render_reset_time, render_speed,
)

SEP = dim(" | ")
_PLAIN_SEP = " | "


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds compactly."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m{secs:02d}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h{minutes:02d}m"


def render_lines(
    ctx: HudRenderContext,
    config: Optional[HudConfig] = None,
) -> list[str]:
    """Compositional API — returns one string per logical HUD line.

    In ``full`` layout, returns up to 8 distinct lines (Nones filtered out).
    In ``minimal`` layout, collapses lines 1-5 into a single truncated line.
    """
    if config is None:
        config = HudConfig()

    elcfg = PRESETS.get(config.preset, PRESETS["focused"])
    max_w = config.max_width or 120
    # Derive layout: explicit config.layout wins; "minimal" preset forces "minimal"
    layout = getattr(config, "layout", "full")
    if layout == "full" and config.preset in ("minimal", "dense"):
        layout = "minimal"

    # --- Build section lines ---
    show_git = getattr(elcfg, "show_git", True)
    section_lines: list[Optional[str]] = [
        render_project_line(ctx, show_git=show_git) if (elcfg.model or elcfg.git) else None,
        render_usage_line(ctx, compact=elcfg.compact_usage) if elcfg.context_bar or elcfg.rate_limits else None,
        render_todos_line(ctx) if elcfg.todos else None,
        render_tools_line(ctx),
        render_env_line(ctx),  # Phase 5 stub — always None for now
    ]

    sections = [s for s in section_lines if s is not None]

    if layout == "minimal":
        # Collapse sections into a single truncated line
        collapsed = _PLAIN_SEP.join(sections)
        output_lines = [wcwidth_truncate(collapsed, max_w)]
    else:
        # Full mode: each section on its own line, wrapped if necessary
        output_lines = []
        for s in sections:
            wrapped = split_wrap_parts(s, max_w, sep=_PLAIN_SEP)
            output_lines.extend(wrapped)

    # Directory line
    if ctx.cwd:
        cwd_display = ctx.cwd.replace(str(pathlib.Path.home()), "~")
        output_lines.append(dim(f"Directory: {cwd_display}"))

    # Per-agent detail lines always render when agents are running (preset doesn't suppress them)
    if ctx.running_agents:
        now = time.time()
        for a in ctx.running_agents:
            desc = a.get("description", a.get("task", ""))
            provider = a.get("provider", "?")
            model = a.get("model", "?")
            status = a.get("status", "running")
            icon = {"running": green("●"), "failed": red("✗"), "timed_out": yellow("⚠")}.get(status, green("●"))
            short_desc = (desc[:160] + "…") if len(desc) > 160 else desc
            elapsed = now - float(a.get("started_at", now))
            elapsed_str = _fmt_elapsed(elapsed)
            line = f"  {icon} {provider}/{model}  {dim(short_desc)}  [{elapsed_str}]"
            cost_usd = a.get("cost_usd")
            if cost_usd:
                if cost_usd >= 0.005:
                    line += f"  ${cost_usd:.2f}"
                else:
                    line += "  <$0.01"
            output_lines.append(line)
    elif elcfg.agents and ctx.agent_count > 0:
        # Summary chip only when no per-agent detail is available and preset allows it
        output_lines.append(f"Agents: {cyan(str(ctx.agent_count))}")

    return output_lines


def render_hud(
    ctx: HudRenderContext,
    config: Optional[HudConfig] = None,
) -> str:
    """Back-compat single-string API.  Equivalent to ``'\\n'.join(render_lines(...))``.

    Existing callers pass ``config.max_width`` for truncation; that is still
    honoured — the first section line is truncated to ``max_width`` columns via
    the grapheme-aware :func:`wcwidth_truncate`.
    """
    return "\n".join(render_lines(ctx, config))


__all__ = [
    "HudElementConfig",
    "magenta",
    "render_agents",
    "render_agents_line",
    "render_context",
    "render_git",
    "render_hud",
    "render_lines",
    "render_mode",
    "render_model",
    "render_rate_limits",
    "render_reset_time",
    "render_session",
    "render_speed",
    "render_todos",
    "render_tokens",
    "split_wrap_parts",
    "wcwidth_truncate",
]
