"""Claude-style HUD — matches Anthropic Claude Code plugin format.

Integrates with existing Mullet infrastructure (:class:`~mullet.hud.types.HudRenderContext`,
:func:`~mullet.hud.state.build_context`, usage APIs, config counts) to provide a dense
5-line status display with real-time budget tracking, MCP status, active provider,
and permission state.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from mullet.hud.colors import (
    RESET,
    BOLD,
    DIM,
    RED,
    GREEN,
    YELLOW,
    CYAN,
    MAGENTA,
    ctx_color,
)
from mullet.hud.config_counts import get_config_counts
from mullet.hud.elements import format_reset_duration
from mullet.hud.state import build_context
from mullet.hud.types import HudRenderContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _budget_color(pct: float) -> str:
    """Return an ANSI colour for a percentage value (0-100)."""
    if pct >= 80:
        return RED
    if pct >= 60:
        return YELLOW
    return GREEN


def _status_color(status: str) -> str:
    """Map a budget-status string to an ANSI colour."""
    if status == "go":
        return GREEN
    if status == "throttle":
        return YELLOW
    if status == "pause":
        return RED
    return DIM


def _speed_color(mode: str) -> str:
    """Return an ANSI colour for the current speed mode."""
    if mode == "max":
        return MAGENTA
    if mode == "fast":
        return CYAN
    if mode == "balanced":
        return YELLOW
    return DIM


# ---------------------------------------------------------------------------
# Fallback data helpers (used when HudRenderContext is sparse)
# ---------------------------------------------------------------------------


def _get_provider_from_log() -> Tuple[Optional[str], Optional[str]]:
    """Read the most recent provider/model from ``~/.mullet/routing.log``."""
    try:
        log_path = Path.home() / ".mullet" / "routing.log"
        if not log_path.exists():
            return None, None
        with open(log_path, encoding="utf-8") as fh:
            lines = fh.readlines()
        if not lines:
            return None, None
        last = lines[-1].strip()
        if not last:
            return None, None
        # Expected: 2026-04-28 22:03:15  provider/model  TIER  reasons  |  task
        parts = [p for p in last.split("  ") if p]
        if len(parts) >= 2:
            provider_model = parts[1]
            if "/" in provider_model:
                provider, model = provider_model.split("/", 1)
                return provider, model
            return provider_model, None
        return None, None
    except Exception as exc:
        logger.debug("claude_style: provider read failed: %s", exc)
        return None, None


def _get_permission_mode(cwd: str) -> str:
    """Return current permission mode (auto/plan/default)."""
    return "default"


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class ClaudeStyleHUD:
    """Render a 5-line Claude Code plugin-style HUD."""

    # Context usage threshold where auto-compact warning appears
    AUTO_COMPACT_THRESHOLD = 90.0

    def __init__(
        self,
        ctx: Optional[HudRenderContext] = None,
        no_color: bool = False,
    ) -> None:
        self.no_color = no_color
        if ctx is None:
            try:
                self.ctx = build_context(cwd=os.getcwd())
            except Exception as exc:
                logger.debug("claude_style: build_context failed: %s", exc)
                self.ctx = HudRenderContext(cwd=os.getcwd())
        else:
            self.ctx = ctx

    def _c(self, text: str, color: str) -> str:
        """Wrap *text* in *color* unless ``no_color`` is set."""
        if self.no_color:
            return text
        return f"{color}{text}{RESET}"

    def _bold(self, text: str) -> str:
        if self.no_color:
            return text
        return f"{BOLD}{text}{RESET}"

    def _dim(self, text: str) -> str:
        if self.no_color:
            return text
        return f"{DIM}{text}{RESET}"

    # ------------------------------------------------------------------
    # Public render
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return the 5-line HUD as a single string."""
        lines: list[str] = []

        # --- Line 1: Budget / context / usage ---
        lines.append(self._render_budget_line())

        # --- Line 2: MCP, hooks, CLAUDE.md, git, speed, todos ---
        lines.append(self._render_config_line())

        # --- Line 3: Directory ---
        lines.append(self._render_directory_line())

        # --- Line 4: Provider / model / tokens ---
        lines.append(self._render_provider_line())

        # --- Line 5: Permission bypass ---
        lines.append(self._render_permission_line())

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Individual lines
    # ------------------------------------------------------------------

    def _render_budget_line(self) -> str:
        """Line 1: ● GO • Ctx 79% • 5h: 3% • 7d: 19% • Resets: 22m • bpd: 0% • bpw: 15% 5% until auto-compact"""
        ctx = self.ctx
        status = getattr(ctx, "budget_status", "go") or "go"

        status_symbol = "●"
        if status == "throttle":
            status_symbol = "⚠"
        elif status == "pause":
            status_symbol = "✗"

        parts: list[str] = []
        parts.append(
            self._c(status_symbol, _status_color(status))
            + " "
            + self._c(status.upper(), _status_color(status))
        )

        # Context usage — always shown when available
        context_pct = ctx.context_pct or 0.0
        parts.append(
            "Ctx " + self._c(f"{context_pct:.0f}%", ctx_color(context_pct))
        )

        # 5h / 7d percentages
        if ctx.five_hour_pct is not None:
            parts.append(
                "5h: " + self._c(f"{ctx.five_hour_pct:.0f}%", _budget_color(ctx.five_hour_pct))
            )
        else:
            parts.append("5h: " + self._dim("?"))

        if ctx.weekly_pct is not None:
            parts.append(
                "7d: " + self._c(f"{ctx.weekly_pct:.0f}%", _budget_color(ctx.weekly_pct))
            )
        else:
            parts.append("7d: " + self._dim("?"))

        # Reset timer
        reset_s = getattr(ctx, "rate_limit_reset_s", 0)
        if reset_s and reset_s > 0:
            mins_left = reset_s // 60
            color = RED if mins_left < 5 else (YELLOW if mins_left < 15 else GREEN)
            parts.append("Resets: " + self._c(format_reset_duration(reset_s), color))
        elif ctx.five_hour_pct is not None:
            parts.append("Resets: " + self._dim("now"))

        # Pressure components
        p5h = getattr(ctx, "pressure_5h", 0.0)
        p7d = getattr(ctx, "pressure_7d", 0.0)
        parts.append(
            "bpd: " + self._c(f"{p5h * 100:.0f}%", _budget_color(p5h * 100))
        )
        parts.append(
            "bpw: " + self._c(f"{p7d * 100:.0f}%", _budget_color(p7d * 100))
        )

        # Auto-compact hint appended without a bullet separator
        until_compact = self.AUTO_COMPACT_THRESHOLD - context_pct
        auto_compact = ""
        if until_compact <= 10 and context_pct > 0:
            auto_compact = self._dim(f" {max(0, until_compact):.0f}% until auto-compact")

        return " • ".join(parts) + auto_compact

    def _render_config_line(self) -> str:
        """Line 2: MCP:1 | Hooks:6 | CLAUDE.md:3"""
        ctx = self.ctx

        try:
            counts = get_config_counts(str(Path(ctx.cwd)))
            mcp = counts.mcp_count
            hooks = counts.hooks_count
            claude_md = counts.claude_md_count
        except Exception as exc:
            logger.debug("claude_style: config_counts failed: %s", exc)
            mcp, hooks, claude_md = 0, 0, 0

        parts: list[str] = [
            "MCP:" + self._c(str(mcp), CYAN),
            "Hooks:" + self._c(str(hooks), CYAN),
            "CLAUDE.md:" + self._c(str(claude_md), CYAN),
        ]

        return " | ".join(parts)

    def _render_directory_line(self) -> str:
        """Line 3: Directory: ~/projects/my-project"""
        cwd = Path(self.ctx.cwd)
        home = Path.home()
        try:
            cwd_relative = cwd.relative_to(home)
            display = f"~/{cwd_relative}"
        except ValueError:
            display = str(cwd)
        return self._dim("Directory: ") + display

    def _render_provider_line(self) -> str:
        """Line 4: ● local/default cli [0s]"""
        ctx = self.ctx

        provider = getattr(ctx, "main_provider", "") or "local"
        model = ctx.model_name or "default"

        # Fallback to routing.log when context has no provider info
        if provider == "local" and model == "default":
            log_provider, log_model = _get_provider_from_log()
            if log_provider:
                provider = log_provider
            if log_model:
                model = log_model

        # Elapsed since last routing.log entry
        elapsed_str = "0s"
        try:
            log_path = Path.home() / ".mullet" / "routing.log"
            if log_path.exists():
                mtime = log_path.stat().st_mtime
                elapsed = int(time.time() - mtime)
                if elapsed < 60:
                    elapsed_str = f"{elapsed}s"
                elif elapsed < 3600:
                    elapsed_str = f"{elapsed // 60}m"
                else:
                    elapsed_str = f"{elapsed // 3600}h"
        except Exception:
            pass

        return f"{self._c('●', GREEN)} {provider}/{model} cli [{elapsed_str}]"

    def _render_permission_line(self) -> str:
        """Line 5: ⏵⏵ bypass permissions on (shift+tab to cycle)"""
        mode = _get_permission_mode(self.ctx.cwd)
        if mode == "plan":
            return (
                f"{self._c('⏵⏵', YELLOW)} plan mode — read-only (shift+tab to cycle)"
            )
        return f"{self._c('⏵⏵', YELLOW)} bypass permissions on (shift+tab to cycle)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_claude_style_hud(
    no_color: bool = False,
    ctx: Optional[HudRenderContext] = None,
) -> str:
    """Render the Claude-style HUD.

    Args:
        no_color: Strip all ANSI escape sequences from the output.
        ctx: Optional pre-built render context. When ``None``, a fresh context
            is assembled from the current working directory via
            :func:`~mullet.hud.state.build_context`.
    """
    return ClaudeStyleHUD(ctx=ctx, no_color=no_color).render()


def main() -> None:
    """Print the HUD to stdout (entry point for ``mullet-claude-hud``)."""
    no_color = os.environ.get("NO_COLOR") is not None or not os.isatty(1)
    print(render_claude_style_hud(no_color=no_color))


if __name__ == "__main__":
    main()
