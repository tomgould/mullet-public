"""HUD element renderers."""
from typing import Optional
from mullet.hud.colors import bold, dim, green, yellow, red, cyan, magenta, bright_blue, ctx_color, bar, RESET
from mullet.hud.types import HudRenderContext

def render_context(ctx: HudRenderContext) -> Optional[str]:
    """Render the context usage percentage as a colored bar."""
    if ctx.context_pct <= 0:
        return None
    c = ctx_color(ctx.context_pct)
    return f"Ctx {c}{ctx.context_pct:.0f}%{RESET} {bar(ctx.context_pct)}"

def render_agents(ctx: HudRenderContext) -> Optional[str]:
    """Render information about running agents and their providers."""
    if not ctx.running_agents:
        if ctx.agent_count <= 0:
            return None
        return f"Agents: {cyan(str(ctx.agent_count))}"

    agent_count = len(ctx.running_agents)

    if agent_count == 1:
        a = ctx.running_agents[0]
        provider = a.get("provider", "?")
        model = a.get("model", "?")
        return f"Agent: {cyan(provider + '/' + model)}"

    # Group by provider, show counts
    provider_counts: dict[str, int] = {}
    for a in ctx.running_agents:
        p = a.get("provider", "?")
        provider_counts[p] = provider_counts.get(p, 0) + 1

    parts = [
        (f"{p}\u00d7{n}" if n > 1 else p)
        for p, n in sorted(provider_counts.items())
    ]
    summary = " ".join(parts)
    return f"Agents: {cyan(str(agent_count))} {dim('(' + summary + ')')}"

def render_mode(ctx: HudRenderContext) -> Optional[str]:
    """Render the current operational mode (Ralph, Autopilot, Ultrawork)."""
    if ctx.ralph_active:
        return f"Ralph: {yellow(f'{ctx.ralph_iter}/{ctx.ralph_max}')}"
    if ctx.autopilot_active:
        return f"Autopilot: {cyan(ctx.autopilot_phase or 'active')}"
    if ctx.ultrawork_active:
        return f"Ultrawork: {magenta(str(ctx.ultrawork_tasks))} tasks"
    return None

def render_todos(ctx: HudRenderContext) -> Optional[str]:
    """Render todo completion progress."""
    if ctx.todos_total <= 0:
        return None
    c = green if ctx.todos_done == ctx.todos_total else (yellow if ctx.todos_done > 0 else dim)
    return f"Todos: {c(f'{ctx.todos_done}/{ctx.todos_total}')}"

def render_session(ctx: HudRenderContext) -> Optional[str]:
    """Render current session duration in minutes."""
    if ctx.session_minutes <= 0:
        return None
    m = int(ctx.session_minutes)
    c = red if m > 60 else (yellow if m > 30 else green)
    return f"Session: {c(f'{m}m')}"

def render_git(ctx: HudRenderContext) -> Optional[str]:
    """Render git branch name and dirty status."""
    if not ctx.git_branch:
        return None
    dirty = f" {yellow('*')}" if ctx.git_dirty else ""
    return f"Git: {bright_blue(ctx.git_branch)}{dirty}"

def render_model(ctx: HudRenderContext) -> Optional[str]:
    """Render the current model name."""
    if not ctx.model_name:
        return None
    return f"Model: {bold(ctx.model_name)}"

def render_tokens(ctx: HudRenderContext) -> Optional[str]:
    """Render token usage statistics."""
    if ctx.total_tokens <= 0:
        return None
    parts = [f"Tok: {dim(str(ctx.total_tokens))}"]
    if ctx.last_in or ctx.last_out:
        parts.append(f"({ctx.last_in}/{ctx.last_out})")
    return " ".join(parts)

def render_rate_limits(ctx: HudRenderContext) -> Optional[str]:
    """Render Anthropic rate limit usage percentages."""
    if ctx.five_hour_pct is None:
        return f"5h: {dim('?')} 7d: {dim('?')}"
    c5 = red if ctx.five_hour_pct >= 80 else (yellow if ctx.five_hour_pct >= 60 else green)
    parts = [f"5h: {c5(f'{ctx.five_hour_pct:.0f}%')}"]
    if ctx.weekly_pct is not None:
        cw = red if ctx.weekly_pct >= 80 else (yellow if ctx.weekly_pct >= 60 else green)
        parts.append(f"7d: {cw(f'{ctx.weekly_pct:.0f}%')}")
    else:
        parts.append(f"7d: {dim('?')}")
    sonnet_pct = getattr(ctx, 'sonnet_weekly_pct', None)
    if sonnet_pct is not None and sonnet_pct > 0:
        cs = red if sonnet_pct >= 80 else (yellow if sonnet_pct >= 60 else green)
        parts.append(f"Sonnet: {cs(f'{sonnet_pct:.0f}%')}")
    return " ".join(parts)

def format_reset_duration(seconds: int) -> str:
    """Format a reset countdown the same way claude-hud does: minutes < 60,
    ``Xh Ym`` < 24h, ``Xd Yh`` otherwise.  Ceiling minutes so a 1-second
    remainder still reads as ``1m`` instead of ``0m``.
    """
    if seconds <= 0:
        return ""
    mins = (seconds + 59) // 60
    if mins < 60:
        return f"{mins}m"
    hours, rem_mins = divmod(mins, 60)
    if hours < 24:
        return f"{hours}h {rem_mins}m" if rem_mins else f"{hours}h"
    days, rem_hours = divmod(hours, 24)
    return f"{days}d {rem_hours}h" if rem_hours else f"{days}d"


def render_reset_time(ctx: HudRenderContext) -> Optional[str]:
    """Render rate limit reset countdown."""
    reset_s = getattr(ctx, 'rate_limit_reset_s', 0)
    if not reset_s or reset_s <= 0:
        return None
    # Colour based on minutes remaining — thresholds unchanged.
    mins_left = reset_s // 60
    c = red if mins_left < 5 else (yellow if mins_left < 15 else green)
    return f"Resets: {c(format_reset_duration(reset_s))}"

def render_speed(ctx: HudRenderContext) -> Optional[str]:
    """Render current speed mode badge if not default economy."""
    mode = getattr(ctx, 'speed_mode', 'economy')
    if not mode or mode == 'economy':
        return None
    from mullet.core.router.complexity import speed_mode_badge
    badge = speed_mode_badge(mode)
    c = yellow if mode == 'balanced' else (cyan if mode == 'fast' else magenta)
    return c(badge)

def _model_badge(provider: Optional[str], model: Optional[str]) -> str:
    """Format a ``provider/model`` suffix for a HUD row.

    Returns an empty string when both are missing so callers can unconditionally
    concatenate without checking. Kept short and dim because the HUD is
    line-capped; the goal is attribution, not emphasis.
    """
    if not provider and not model:
        return ""
    if provider and model:
        tag = f"{provider}/{model}"
    else:
        tag = provider or model or ""
    return dim(f" {tag}")


def render_thinking(ctx: HudRenderContext) -> Optional[str]:
    """Render thinking indicator with provider/model badge."""
    if not getattr(ctx, 'thinking_active', False):
        return None
    badge = _model_badge(
        getattr(ctx, 'main_provider', '') or None,
        getattr(ctx, 'model_name', '') or None,
    )
    return magenta("Thinking...") + badge

def render_search_quality(result_count: int, time_ms: float, score_min: float, score_max: float, source_count: int) -> str:
    """Format search quality indicators for display."""
    return f"Search: {result_count} results in {time_ms:.0f}ms (scores: {score_min:.2f}-{score_max:.2f}, {source_count} sources)"
