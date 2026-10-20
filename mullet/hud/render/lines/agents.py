"""Agents line renderer: per-agent status rows or summary chip."""
from typing import Optional

from mullet.hud.types import HudRenderContext
from mullet.hud.colors import cyan, dim, green, red, yellow

_STATUS_ICON = {
    "running":   lambda: green("●"),
    "failed":    lambda: red("✗"),
    "timed_out": lambda: yellow("⚠"),
}


def _agent_chip(a: dict) -> str:
    status = a.get("status", "running")
    icon = _STATUS_ICON.get(status, lambda: green("●"))()
    provider = a.get("provider", "?")
    model = a.get("model", "?")
    desc = a.get("description", a.get("task", ""))
    label = f"{provider}/{model}"
    if desc:
        short = desc[:160] + ("…" if len(desc) > 160 else "")
        label = f"{label}  {dim(short)}"
    return f"{icon} {label}"


def render_agents_line(ctx: HudRenderContext) -> Optional[str]:
    """Return a compact agents summary line, or None when no agents are running."""
    agents = getattr(ctx, "running_agents", []) or []

    if not agents:
        if ctx.agent_count > 0:
            return f"Agents: {cyan(str(ctx.agent_count))}"
        return None

    count = len(agents)

    # ≤4 agents: per-row detail with status icon
    if count <= 4:
        return "\n".join(_agent_chip(a) for a in agents)

    # >4 agents: summary chip to avoid wall of text
    provider_counts: dict[str, int] = {}
    for a in agents:
        p = a.get("provider", "?")
        provider_counts[p] = provider_counts.get(p, 0) + 1
    summary_parts = [
        (f"{p}×{n}" if n > 1 else p)
        for p, n in sorted(provider_counts.items())
    ]
    summary = " ".join(summary_parts)
    return f"Agents: {cyan(str(count))} {dim('(' + summary + ')')}"
