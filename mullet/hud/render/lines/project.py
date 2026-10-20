"""Project line renderer: [Model | cwd-basename | Git | Mode | Speed]."""
from typing import Optional
import pathlib

from mullet.hud.types import HudRenderContext
from mullet.hud.colors import bold, dim, bright_blue, yellow, cyan, magenta
from mullet.hud.elements import render_mode


def render_project_line(ctx: HudRenderContext, show_git: bool = True) -> Optional[str]:
    """Return ``[Model | cwd-basename | Git]`` with optional mode/speed appended.

    Returns None if all parts are absent.
    show_git: if False, skip git branch/dirty status.
    """
    parts: list[str] = []

    if ctx.model_name:
        parts.append(bold(ctx.model_name))

    if ctx.cwd:
        basename = pathlib.Path(ctx.cwd).name or ctx.cwd
        parts.append(dim(basename))

    if show_git and ctx.git_branch:
        dirty = yellow(" *") if ctx.git_dirty else ""
        parts.append(bright_blue(ctx.git_branch) + dirty)

    # Mode (Ralph / Autopilot / Ultrawork)
    mode_str = render_mode(ctx)
    if mode_str:
        parts.append(mode_str)

    # Speed badge (non-economy)
    speed_mode = getattr(ctx, "speed_mode", "economy")
    if speed_mode and speed_mode != "economy":
        try:
            from mullet.core.router.complexity import speed_mode_badge
            badge = speed_mode_badge(speed_mode)
            c = yellow if speed_mode == "balanced" else (cyan if speed_mode == "fast" else magenta)
            parts.append(c(badge))
        except ImportError:
            parts.append(dim(f"[{speed_mode}]"))

    if not parts:
        return None

    inner = dim(" | ").join(parts)
    return f"[{inner}]"
