from unittest.mock import patch, MagicMock

from mullet.hud.claude_style import (
    _budget_color,
    _status_color,
    _speed_color,
    _get_provider_from_log,
    ClaudeStyleHUD,
)
from mullet.hud.types import HudRenderContext
from mullet.hud.colors import RED, YELLOW, GREEN, CYAN, MAGENTA, DIM


def _ctx(**kw):
    """Build a minimal HudRenderContext for testing."""
    defaults = dict(
        cwd="/tmp",
        budget_status="go",
        five_hour_pct=10.0,
        weekly_pct=20.0,
        context_pct=50.0,
        pressure_5h=0.1,
        pressure_7d=0.2,
    )
    defaults.update(kw)
    return HudRenderContext(**defaults)


class TestBudgetColor:
    def test_zero(self):
        assert _budget_color(0) == GREEN

    def test_fifty_nine(self):
        assert _budget_color(59) == GREEN

    def test_sixty(self):
        assert _budget_color(60) == YELLOW

    def test_seventy_nine(self):
        assert _budget_color(79) == YELLOW

    def test_eighty(self):
        assert _budget_color(80) == RED

    def test_one_hundred(self):
        assert _budget_color(100) == RED


class TestStatusColor:
    def test_go(self):
        assert _status_color("go") == GREEN

    def test_throttle(self):
        assert _status_color("throttle") == YELLOW

    def test_pause(self):
        assert _status_color("pause") == RED

    def test_unknown(self):
        assert _status_color("unknown") == DIM

    def test_empty(self):
        assert _status_color("") == DIM


class TestSpeedColor:
    def test_max(self):
        assert _speed_color("max") == MAGENTA

    def test_fast(self):
        assert _speed_color("fast") == CYAN

    def test_balanced(self):
        assert _speed_color("balanced") == YELLOW

    def test_economy(self):
        assert _speed_color("economy") == DIM

    def test_other(self):
        assert _speed_color("other") == DIM


class TestGetProviderFromLog:
    def test_absent_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _get_provider_from_log() == (None, None)

    def test_empty_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        (log_dir / "routing.log").write_text("")
        assert _get_provider_from_log() == (None, None)

    def test_valid_line(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        (log_dir / "routing.log").write_text(
            "2026-04-28 22:03:15  kimi/kimi-k2.7-code  STANDARD  cost  |  task\n"
        )
        assert _get_provider_from_log() == ("kimi", "kimi-k2.7-code")

    def test_provider_only(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        (log_dir / "routing.log").write_text(
            "2026-04-28 22:03:15  openrouter  STANDARD\n"
        )
        assert _get_provider_from_log() == ("openrouter", None)

    def test_multiple_lines(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        log_dir = tmp_path / ".mullet"
        log_dir.mkdir()
        (log_dir / "routing.log").write_text(
            "2026-04-28 22:03:15  first/provider  STANDARD\n"
            "2026-04-28 22:03:16  second/model  FAST\n"
        )
        assert _get_provider_from_log() == ("second", "model")


class TestClaudeStyleHUDHelpers:
    def test_c_with_color(self):
        hud = ClaudeStyleHUD(no_color=False)
        result = hud._c("text", RED)
        assert "text" in result
        assert RED in result

    def test_c_no_color(self):
        hud = ClaudeStyleHUD(no_color=True)
        assert hud._c("text", RED) == "text"

    def test_bold_no_color(self):
        hud = ClaudeStyleHUD(no_color=True)
        assert hud._bold("text") == "text"

    def test_dim_no_color(self):
        hud = ClaudeStyleHUD(no_color=True)
        assert hud._dim("text") == "text"


class TestRenderBudgetLine:
    def test_status_go(self):
        hud = ClaudeStyleHUD(ctx=_ctx(budget_status="go"), no_color=True)
        output = hud._render_budget_line()
        assert "GO" in output

    def test_status_throttle(self):
        hud = ClaudeStyleHUD(ctx=_ctx(budget_status="throttle"), no_color=True)
        output = hud._render_budget_line()
        assert "THROTTLE" in output
        assert "⚠" in output

    def test_status_pause(self):
        hud = ClaudeStyleHUD(ctx=_ctx(budget_status="pause"), no_color=True)
        output = hud._render_budget_line()
        assert "PAUSE" in output
        assert "✗" in output

    def test_five_hour_pct(self):
        hud = ClaudeStyleHUD(ctx=_ctx(five_hour_pct=30.0), no_color=True)
        output = hud._render_budget_line()
        assert "5h: 30%" in output

    def test_weekly_pct(self):
        hud = ClaudeStyleHUD(ctx=_ctx(weekly_pct=45.0), no_color=True)
        output = hud._render_budget_line()
        assert "7d: 45%" in output

    def test_context_pct(self):
        hud = ClaudeStyleHUD(ctx=_ctx(context_pct=85.0), no_color=True)
        output = hud._render_budget_line()
        assert "Ctx 85%" in output

    def test_auto_compact_hint(self):
        hud = ClaudeStyleHUD(ctx=_ctx(context_pct=82.0), no_color=True)
        output = hud._render_budget_line()
        assert "8% until auto-compact" in output

    def test_no_auto_compact_hint(self):
        hud = ClaudeStyleHUD(ctx=_ctx(context_pct=50.0), no_color=True)
        output = hud._render_budget_line()
        assert "until auto-compact" not in output


class TestRenderConfigLine:
    def test_config_counts(self):
        mock_counts = MagicMock()
        mock_counts.mcp_count = 3
        mock_counts.hooks_count = 6
        mock_counts.claude_md_count = 2

        with patch("mullet.hud.claude_style.get_config_counts", return_value=mock_counts):
            hud = ClaudeStyleHUD(ctx=_ctx(), no_color=True)
            output = hud._render_config_line()
            assert "MCP:3" in output
            assert "Hooks:6" in output
            assert "CLAUDE.md:2" in output


class TestRenderDirectoryLine:
    def test_cwd_is_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        hud = ClaudeStyleHUD(ctx=_ctx(cwd=str(tmp_path)), no_color=True)
        output = hud._render_directory_line()
        assert "~/" in output

    def test_cwd_not_home(self):
        hud = ClaudeStyleHUD(ctx=_ctx(cwd="/tmp/somewhere"), no_color=True)
        output = hud._render_directory_line()
        assert "/tmp/somewhere" in output


class TestRender:
    def test_returns_string(self):
        hud = ClaudeStyleHUD(ctx=_ctx(), no_color=True)
        result = hud.render()
        assert isinstance(result, str)

    def test_contains_newlines(self):
        hud = ClaudeStyleHUD(ctx=_ctx(), no_color=True)
        result = hud.render()
        assert "\n" in result

    def test_contains_go(self):
        hud = ClaudeStyleHUD(ctx=_ctx(budget_status="go"), no_color=True)
        result = hud.render()
        assert "GO" in result
