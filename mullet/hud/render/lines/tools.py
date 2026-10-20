"""Tools line renderer: running tool pills only, one per line.

Only tools currently IN FLIGHT (unmatched tool_use in the transcript) are
rendered. Completed-tool history is derivable from the transcript if anyone
wants it later, but lingering ``✓ Read 0ms`` rows add visual noise without
conveying anything live — the transcript itself is already the authoritative
log of what finished.

Each pill carries a ``provider/model`` badge when the parent assistant message
carried a model id (always true from Phase 2 transcripts). Missing attribution
suppresses the badge rather than printing ``?/?``.
"""
from typing import List, Optional

from mullet.hud.types import HudRenderContext
from mullet.hud.colors import cyan
from mullet.hud.elements import _model_badge


def render_tools_line(ctx: HudRenderContext) -> Optional[str]:
    """Return one running-tool pill per line, or ``None`` when nothing is running."""
    running = getattr(ctx, "running_tools", []) or []
    if not running:
        return None

    lines: List[str] = []
    for t in running:
        name = t.get("name", "?")
        target = t.get("target", "")
        label = f"{name}({target})" if target else name
        badge = _model_badge(t.get("provider"), t.get("model"))
        lines.append(cyan(f"  \u21bb {label}") + badge)
    return "\n".join(lines)
