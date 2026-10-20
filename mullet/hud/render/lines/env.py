"""Env line renderer: MCP / Hooks / Rules / CLAUDE.md counts."""
from typing import Optional

from mullet.hud.types import HudRenderContext
from mullet.hud.colors import dim, bold, cyan
from mullet.hud.config_counts import get_config_counts, ConfigCounts


def render_env_line(ctx: HudRenderContext) -> Optional[str]:
    """Render config counts: ``MCP:N | Hooks:N | Rules:N | CLAUDE.md:N | Style:name``."""
    counts = get_config_counts(ctx.cwd) if ctx.cwd else ConfigCounts()
    parts = []

    if counts.mcp_count:
        parts.append(f"MCP:{cyan(str(counts.mcp_count))}")
    if counts.hooks_count:
        parts.append(f"Hooks:{cyan(str(counts.hooks_count))}")
    if counts.rules_count:
        parts.append(f"Rules:{cyan(str(counts.rules_count))}")
    if counts.claude_md_count:
        parts.append(f"CLAUDE.md:{cyan(str(counts.claude_md_count))}")
    if counts.output_style:
        parts.append(f"Style:{bold(counts.output_style)}")

    if not parts:
        return None

    return dim(" | ").join(parts)
