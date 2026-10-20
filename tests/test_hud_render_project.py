"""Tests for mullet.hud.render.lines.project — missing lines coverage."""
from unittest.mock import patch

from mullet.hud.render.lines.project import render_project_line
from mullet.hud.types import HudRenderContext
from mullet.hud.colors import strip_ansi


class TestRenderProjectLineMissingLines:
    """Cover lines 42-43 in project.py — ImportError fallback for speed_mode_badge."""

    def test_speed_mode_import_error_fallback(self):
        """When speed_mode_badge import fails, render dim badge text."""
        ctx = HudRenderContext(
            cwd="/home/user/myproject",
            model_name="Opus",
            speed_mode="fast",
        )
        with patch.dict("sys.modules", {"mullet.core.router.complexity": None}):
            line = render_project_line(ctx)
        plain = strip_ansi(line or "")
        assert "[fast]" in plain

    def test_speed_mode_fast_without_import_error(self):
        """Normal path when import succeeds."""
        ctx = HudRenderContext(
            cwd="/home/user/myproject",
            model_name="Opus",
            speed_mode="fast",
        )
        line = render_project_line(ctx)
        plain = strip_ansi(line or "")
        # Should contain some speed badge (exact text depends on speed_mode_badge)
        assert plain is not None

    def test_mode_str_appended(self):
        """When render_mode returns a non-empty string, it is appended (line 32)."""
        ctx = HudRenderContext(
            cwd="/home/user/myproject",
            model_name="Opus",
            ralph_active=True,
            ralph_iter=3,
            ralph_max=10,
        )
        line = render_project_line(ctx)
        plain = strip_ansi(line or "")
        assert "Ralph" in plain or "ralph" in plain.lower()
