"""Usage line renderer: Ctx XX% • 5h: XX% • 7d: XX% • Sonnet: XX% • Resets: ..."""
from typing import Optional

from mullet.hud.types import HudRenderContext
from mullet.hud.colors import dim, green, yellow, red, ctx_color, bar, RESET
from mullet.hud.elements import format_reset_duration


_BUDGET_BADGE = {
    "go": ("●", green),
    "throttle": ("⚠", yellow),
    "pause": ("✗", red),
}


def render_usage_line(ctx: HudRenderContext, compact: bool = False) -> Optional[str]:
    """Return a combined usage line or None when no usage data is present."""
    parts: list[str] = []

    status = getattr(ctx, "budget_status", "")
    if status and status in _BUDGET_BADGE:
        icon, color = _BUDGET_BADGE[status]
        parts.append(color(f"{icon} {status.upper()}"))

    if ctx.context_pct > 0:
        c = ctx_color(ctx.context_pct)
        bar_str = "" if compact else f" {bar(ctx.context_pct)}"
        parts.append(f"Ctx {c}{ctx.context_pct:.0f}%{RESET}{bar_str}")

    # Rate-limit percentages
    if ctx.five_hour_pct is not None:
        c5 = red if ctx.five_hour_pct >= 80 else (yellow if ctx.five_hour_pct >= 60 else green)
        parts.append(f"5h: {c5(f'{ctx.five_hour_pct:.0f}%')}")
    else:
        parts.append(f"5h: {dim('?')}")

    if ctx.weekly_pct is not None:
        cw = red if ctx.weekly_pct >= 80 else (yellow if ctx.weekly_pct >= 60 else green)
        parts.append(f"7d: {cw(f'{ctx.weekly_pct:.0f}%')}")
    else:
        parts.append(f"7d: {dim('?')}")

    sonnet_pct = getattr(ctx, "sonnet_weekly_pct", None)
    if sonnet_pct is not None and sonnet_pct > 0:
        cs = red if sonnet_pct >= 80 else (yellow if sonnet_pct >= 60 else green)
        parts.append(f"Sonnet: {cs(f'{sonnet_pct:.0f}%')}")

    reset_s = getattr(ctx, "rate_limit_reset_s", 0)
    if reset_s and reset_s > 0:
        mins_left = reset_s // 60
        cr = red if mins_left < 5 else (yellow if mins_left < 15 else green)
        parts.append(f"Resets: {cr(format_reset_duration(reset_s))}")

    p5h = getattr(ctx, "pressure_5h", 0.0)
    p7d = getattr(ctx, "pressure_7d", 0.0)
    if p5h > 0 or p7d > 0:
        def _pc(v: float) -> str:
            pct = v * 100
            c = red if pct >= 80 else (yellow if pct >= 60 else green)
            return c(f"{pct:.0f}%")
        parts.append(f"bpd: {_pc(p5h)} • bpw: {_pc(p7d)}")

    # Per-provider spend in last 5h — skip free (local) and zero-spend providers
    _SKIP_PROVIDERS = {"local", "unknown"}
    costs = getattr(ctx, "provider_costs_5h", {})
    spend_parts = []
    for provider, cost in sorted(costs.items()):
        if provider.lower() in _SKIP_PROVIDERS:
            continue
        if cost < 0.001:
            continue
        c = red if cost > 5.0 else (yellow if cost > 1.0 else green)
        spend_parts.append(f"{provider}: {c(f'${cost:.2f}')}")
    if spend_parts:
        parts.append(" ".join(spend_parts))

    if not parts:  # pragma: no cover
        return None

    SEP = " • "
    return SEP.join(parts)
