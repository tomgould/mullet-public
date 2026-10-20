"""Todos line renderer: Todos: X/Y with mini-progress bar."""
from typing import Optional

from mullet.hud.types import HudRenderContext
from mullet.hud.colors import green, yellow, dim


def render_todos_line(ctx: HudRenderContext) -> Optional[str]:
    """Return a todos progress line or None when no todos are tracked."""
    if ctx.todos_total <= 0:
        return None

    if ctx.todos_done == ctx.todos_total:
        c = green
    elif ctx.todos_done > 0:
        c = yellow
    else:
        c = dim

    return f"Todos: {c(f'{ctx.todos_done}/{ctx.todos_total}')}"
