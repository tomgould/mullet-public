"""Tests for mullet.hud.claude_style covering missing branches."""
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.hud.claude_style import (
    _budget_color,
    _get_permission_mode,
    _get_provider_from_log,
    _speed_color,
    _status_color,
    ClaudeStyleHUD,
    render_claude_style_hud,
    main,
)
from mullet.hud.colors import RED, YELLOW, GREEN, DIM, MAGENTA, CYAN, BOLD
from mullet.hud.types import HudRenderContext


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

class TestBudgetColor:
    def test_red_at_80(self):
        assert _budget_color(80.0) == RED
        assert _budget_color(95.0) == RED

    def test_yellow_at_60(self):
        assert _budget_color(60.0) == YELLOW
        assert _budget_color(79.0) == YELLOW

    def test_green_below_60(self):
        assert _budget_color(0.0) == GREEN
        assert _budget_color(59.0) == GREEN


class TestStatusColor:
    def test_go(self):
        assert _status_color("go") == GREEN

    def test_throttle(self):
        assert _status_color("throttle") == YELLOW

    def test_pause(self):
        assert _status_color("pause") == RED

    def test_unknown(self):
        assert _status_color("unknown") == DIM
        assert _status_color("") == DIM


class TestSpeedColor:
    def test_max(self):
        assert _speed_color("max") == MAGENTA

    def test_fast(self):
        assert _speed_color("fast") == CYAN

    def test_balanced(self):
        assert _speed_color("balanced") == YELLOW

    def test_other(self):
        assert _speed_color("economy") == DIM


# ---------------------------------------------------------------------------
# _get_provider_from_log
# ---------------------------------------------------------------------------

class TestGetProviderFromLog:
    def test_no_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _get_provider_from_log() == (None, None)

    def test_empty_log_file(self, tmp_path, monkeypatch):
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        (log_dir / "routing.log").write_text("")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _get_provider_from_log() == (None, None)

    def test_last_line_empty(self, tmp_path, monkeypatch):
        """Line 87 — last line is empty after strip."""
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        (log_dir / "routing.log").write_text("some line\n   \n")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _get_provider_from_log() == (None, None)

    def test_parts_less_than_two(self, tmp_path, monkeypatch):
        """Line 96 — not enough parts in log line."""
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        (log_dir / "routing.log").write_text("only_one_part\n")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _get_provider_from_log() == (None, None)

    def test_provider_model_split(self, tmp_path, monkeypatch):
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        (log_dir / "routing.log").write_text(
            "2026-04-28 22:03:15  openrouter/gpt-4  TIER  reasons  |  task\n"
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _get_provider_from_log() == ("openrouter", "gpt-4")

    def test_provider_no_model(self, tmp_path, monkeypatch):
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        (log_dir / "routing.log").write_text(
            "2026-04-28 22:03:15  local  TIER  reasons  |  task\n"
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _get_provider_from_log() == ("local", None)

    def test_exception_handling(self, tmp_path, monkeypatch):
        """Lines 97-99 — exception while reading log."""
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        log_file = log_dir / "routing.log"
        log_file.write_text("data\n")

        def boom(*args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
        assert _get_provider_from_log() == (None, None)


# ---------------------------------------------------------------------------
# _get_permission_mode
# ---------------------------------------------------------------------------

class TestGetPermissionMode:
    def test_returns_default(self):
        """_get_permission_mode always returns 'default'."""
        assert _get_permission_mode("/any/path") == "default"


# ---------------------------------------------------------------------------
# ClaudeStyleHUD
# ---------------------------------------------------------------------------

class TestClaudeStyleHUDInit:
    def test_build_context_failure(self, monkeypatch):
        """Lines 134-136 — build_context raises, falls back to empty context."""
        monkeypatch.setattr(
            "mullet.hud.claude_style.build_context",
            lambda **kw: (_ for _ in ()).raise_(RuntimeError("fail")),
        )
        hud = ClaudeStyleHUD()
        assert hud.ctx.cwd == os.getcwd()

    def test_explicit_ctx(self):
        ctx = HudRenderContext(cwd="/tmp")
        hud = ClaudeStyleHUD(ctx=ctx)
        assert hud.ctx is ctx


class TestNoColorMethods:
    def test_bold_no_color(self):
        hud = ClaudeStyleHUD(no_color=True)
        assert hud._bold("text") == "text"

    def test_bold_with_color(self):
        """Line 149 — _bold applies BOLD when color is enabled."""
        hud = ClaudeStyleHUD(no_color=False)
        result = hud._bold("text")
        assert BOLD in result
        assert "text" in result

    def test_dim_no_color(self):
        hud = ClaudeStyleHUD(no_color=True)
        assert hud._dim("text") == "text"

    def test_c_no_color(self):
        hud = ClaudeStyleHUD(no_color=True)
        assert hud._c("text", RED) == "text"


class TestRenderBudgetLineMissingBranches:
    def test_five_hour_pct_none(self):
        """Line 215 — five_hour_pct is None shows '?'."""
        ctx = HudRenderContext(five_hour_pct=None, weekly_pct=10.0)
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_budget_line()
        assert "5h: ?" in line

    def test_weekly_pct_none(self):
        """Line 222 — weekly_pct is None shows '?'."""
        ctx = HudRenderContext(five_hour_pct=10.0, weekly_pct=None)
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_budget_line()
        assert "7d: ?" in line

    def test_reset_s_positive(self):
        """Lines 227-229 — positive reset_s with color formatting."""
        ctx = HudRenderContext(
            five_hour_pct=10.0,
            weekly_pct=10.0,
            rate_limit_reset_s=600,  # 10 minutes
        )
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_budget_line()
        assert "Resets:" in line

    def test_reset_s_zero_no_five_hour(self):
        """When reset_s is 0 and five_hour_pct is None, no Resets shown."""
        ctx = HudRenderContext(
            five_hour_pct=None,
            weekly_pct=None,
            rate_limit_reset_s=0,
        )
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_budget_line()
        assert "Resets:" not in line

    def test_throttle_status(self):
        """Line 192 — throttle status uses ⚠ symbol."""
        ctx = HudRenderContext(five_hour_pct=10.0, weekly_pct=10.0, budget_status="throttle")
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_budget_line()
        assert "⚠" in line
        assert "THROTTLE" in line

    def test_pause_status(self):
        """Line 194 — pause status uses ✗ symbol."""
        ctx = HudRenderContext(five_hour_pct=10.0, weekly_pct=10.0, budget_status="pause")
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_budget_line()
        assert "✗" in line
        assert "PAUSE" in line

    def test_auto_compact_hint(self):
        """Line 247 — context_pct near threshold shows auto-compact hint."""
        ctx = HudRenderContext(
            five_hour_pct=10.0,
            weekly_pct=10.0,
            context_pct=85.0,
        )
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_budget_line()
        assert "until auto-compact" in line


class TestRenderConfigLineMissingBranch:
    def test_get_config_counts_failure(self, monkeypatch):
        """Lines 260-262 — get_config_counts raises, defaults to 0."""
        monkeypatch.setattr(
            "mullet.hud.claude_style.get_config_counts",
            lambda p: (_ for _ in ()).throw(RuntimeError("fail")),
        )
        ctx = HudRenderContext(cwd="/tmp")
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_config_line()
        assert "MCP:0" in line
        assert "Hooks:0" in line
        assert "CLAUDE.md:0" in line


class TestRenderProviderLineMissingBranches:
    def test_elapsed_hours(self, monkeypatch, tmp_path):
        """Lines 307-310 — elapsed >= 3600 formats as hours."""
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        log_file = log_dir / "routing.log"
        log_file.write_text("data\n")
        # Set mtime to 2 hours ago
        two_hours_ago = time.time() - 7200
        os.utime(log_file, (two_hours_ago, two_hours_ago))

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        ctx = HudRenderContext()
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_provider_line()
        assert "2h" in line or "1h" in line  # depending on rounding

    def test_log_stat_exception(self, monkeypatch, tmp_path):
        """Lines 311-312 — exception in stat/mtime handling."""
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        log_file = log_dir / "routing.log"
        log_file.write_text("data\n")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(
            "mullet.hud.claude_style.Path.stat",
            lambda self: (_ for _ in ()).throw(OSError("stat fail")),
        )
        ctx = HudRenderContext()
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_provider_line()
        assert "0s" in line

    def test_elapsed_seconds(self, monkeypatch, tmp_path):
        """Line 306 — elapsed < 60 formats as seconds."""
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        log_file = log_dir / "routing.log"
        log_file.write_text("data\n")
        # Set mtime to 30 seconds ago
        thirty_secs_ago = time.time() - 30
        os.utime(log_file, (thirty_secs_ago, thirty_secs_ago))

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        ctx = HudRenderContext()
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_provider_line()
        assert "30s" in line or "29s" in line or "31s" in line

    def test_elapsed_minutes(self, monkeypatch, tmp_path):
        """Lines 307-308 — 60 <= elapsed < 3600 formats as minutes."""
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        log_file = log_dir / "routing.log"
        log_file.write_text("data\n")
        # Set mtime to 15 minutes ago
        fifteen_mins_ago = time.time() - 900
        os.utime(log_file, (fifteen_mins_ago, fifteen_mins_ago))

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        ctx = HudRenderContext()
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_provider_line()
        assert "15m" in line


class TestRenderDirectoryLineMissingBranch:
    def test_directory_under_home(self):
        """Line 278 — cwd under home uses ~/ prefix."""
        home = Path.home()
        cwd = str(home / "projects" / "my-project")
        ctx = HudRenderContext(cwd=cwd)
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_directory_line()
        assert "~/projects/my-project" in line


class TestRenderPermissionLineMissingBranch:
    def test_plan_mode(self, monkeypatch):
        """Line 320 — plan mode renders specific message."""
        monkeypatch.setattr(
            "mullet.hud.claude_style._get_permission_mode",
            lambda cwd: "plan",
        )
        ctx = HudRenderContext(cwd="/tmp")
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        line = hud._render_permission_line()
        assert "plan mode" in line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestRenderClaudeStyleHud:
    def test_renders_string(self):
        ctx = HudRenderContext(cwd="/tmp", five_hour_pct=10.0, weekly_pct=10.0)
        result = render_claude_style_hud(ctx=ctx, no_color=True)
        assert isinstance(result, str)
        assert "\n" in result

    def test_no_color_false(self):
        ctx = HudRenderContext(cwd="/tmp", five_hour_pct=10.0, weekly_pct=10.0)
        result = render_claude_style_hud(ctx=ctx, no_color=False)
        assert isinstance(result, str)


class TestMain:
    def test_main_prints(self, monkeypatch, capsys):
        """Lines 348-349 — main() prints HUD."""
        monkeypatch.setattr(os, "isatty", lambda fd: False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(
            "mullet.hud.claude_style.render_claude_style_hud",
            lambda no_color: "HUD_OUTPUT",
        )
        main()
        captured = capsys.readouterr()
        assert "HUD_OUTPUT" in captured.out

    def test_main_no_color_env(self, monkeypatch, capsys):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setattr(os, "isatty", lambda fd: True)
        monkeypatch.setattr(
            "mullet.hud.claude_style.render_claude_style_hud",
            lambda no_color: f"no_color={no_color}",
        )
        main()
        captured = capsys.readouterr()
        assert "no_color=True" in captured.out

    def test_main_tty_no_env(self, monkeypatch, capsys):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(os, "isatty", lambda fd: True)
        monkeypatch.setattr(
            "mullet.hud.claude_style.render_claude_style_hud",
            lambda no_color: f"no_color={no_color}",
        )
        main()
        captured = capsys.readouterr()
        assert "no_color=False" in captured.out


class TestModuleEntryPoint:
    def test_name_main_guard(self, monkeypatch):
        """Line 353 — __name__ == '__main__' branch executes main()."""
        import runpy
        claude_style_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../mullet/hud/claude_style.py")
        )
        monkeypatch.setattr(os, "isatty", lambda fd: False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(
            "mullet.hud.state.build_context",
            lambda cwd: HudRenderContext(cwd=cwd or "/tmp"),
        )
        with patch("builtins.print") as mock_print:
            runpy.run_path(claude_style_path, run_name="__main__")
            mock_print.assert_called_once()


# ---------------------------------------------------------------------------
# Render method integration
# ---------------------------------------------------------------------------

class TestClaudeStyleMissingLines:
    """Integration tests exercising all missing branches together."""

    def test_full_render_with_none_pcts(self):
        ctx = HudRenderContext(
            cwd="/tmp",
            five_hour_pct=None,
            weekly_pct=None,
        )
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        out = hud.render()
        assert "5h: ?" in out
        assert "7d: ?" in out

    def test_full_render_with_plan_mode(self, monkeypatch):
        monkeypatch.setattr(
            "mullet.hud.claude_style._get_permission_mode",
            lambda cwd: "plan",
        )
        ctx = HudRenderContext(cwd="/tmp", five_hour_pct=10.0, weekly_pct=10.0)
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        out = hud.render()
        assert "plan mode" in out

    def test_full_render_with_config_counts_failure(self, monkeypatch):
        monkeypatch.setattr(
            "mullet.hud.claude_style.get_config_counts",
            lambda p: (_ for _ in ()).throw(RuntimeError("fail")),
        )
        ctx = HudRenderContext(cwd="/tmp", five_hour_pct=10.0, weekly_pct=10.0)
        hud = ClaudeStyleHUD(ctx=ctx, no_color=True)
        out = hud.render()
        assert "MCP:0" in out
