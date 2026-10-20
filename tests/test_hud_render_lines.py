"""Tests for the compositional render_lines / render_hud API."""
import pytest
from mullet.hud.types import HudRenderContext, HudConfig
from mullet.hud.render import render_hud, render_lines
from mullet.hud.colors import strip_ansi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_ctx():
    return HudRenderContext()


@pytest.fixture
def full_ctx():
    return HudRenderContext(
        cwd="/home/user/myproject",
        model_name="Opus 4.7",
        context_pct=55.0,
        five_hour_pct=40.0,
        weekly_pct=20.0,
        git_branch="main",
        git_dirty=False,
        todos_done=3,
        todos_total=5,
        running_agents=[{"provider": "kimi", "model": "k2", "description": "rewrite tests"}],
        running_tools=[{"name": "Read", "target": "foo.py"}],
        completed_tools=[{"name": "Edit", "duration_ms": 42}],
        rate_limit_reset_s=300,
    )


# ---------------------------------------------------------------------------
# render_lines — individual section tests
# ---------------------------------------------------------------------------

class TestRenderProjectLine:
    def test_model_appears(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Opus 4.7" in combined

    def test_cwd_basename_appears(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "myproject" in combined

    def test_git_branch_appears(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "main" in combined

    def test_no_model_no_cwd_no_git_omits_project_line(self, empty_ctx):
        lines = render_lines(empty_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        # No bracketed block expected
        assert "[" not in combined.replace("Directory", "DIR")


class TestRenderUsageLine:
    def test_context_pct_shown(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "55%" in combined

    def test_five_hour_pct_shown(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "5h: 40%" in combined

    def test_weekly_pct_shown(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "7d: 20%" in combined

    def test_reset_shown(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Resets:" in combined


class TestRenderTodosLine:
    def test_todos_shown(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Todos: 3/5" in combined

    def test_no_todos_absent(self, empty_ctx):
        lines = render_lines(empty_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Todos" not in combined


class TestRenderAgentsLine:
    def test_agent_shown(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Agent:" in combined or "kimi" in combined

    def test_no_agents_absent(self, empty_ctx):
        lines = render_lines(empty_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Agent" not in combined


class TestRenderToolsLine:
    def test_running_tool_shown(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Read" in combined

    def test_completed_tools_not_rendered(self, full_ctx):
        """Completed tools intentionally don't render — see lines/tools.py docstring.

        The running ``Read(foo.py)`` pill is the only live tool signal; the
        ``Edit`` entry in ``completed_tools`` fixture should NOT appear.
        """
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        # Read (running) should appear
        assert "Read(foo.py)" in combined
        # Edit (completed) should NOT appear as a tool pill
        assert "✓ Edit" not in combined
        assert "Edit 0ms" not in combined

    def test_no_tools_absent(self, empty_ctx):
        lines = render_lines(empty_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Read" not in combined


class TestDirectoryLine:
    def test_directory_shown(self, full_ctx):
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Directory:" in combined

    def test_no_cwd_no_directory(self, empty_ctx):
        lines = render_lines(empty_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "Directory" not in combined


class TestPerAgentDetailLines:
    def test_per_agent_bullet_line(self, full_ctx):
        lines = render_lines(full_ctx)
        # The per-agent detail line starts with ● (green)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "●" in combined
        assert "kimi" in combined

    def test_per_agent_elapsed_default(self, full_ctx):
        """When started_at is absent, elapsed defaults to ~0s."""
        lines = render_lines(full_ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "[0s]" in combined

    def test_per_agent_elapsed_formatted(self, monkeypatch):
        """Elapsed time is formatted based on started_at."""
        now = 1000000.0
        monkeypatch.setattr("mullet.hud.render.time.time", lambda: now)
        ctx = HudRenderContext(
            running_agents=[{
                "provider": "kimi",
                "model": "k2",
                "description": "rewrite tests",
                "started_at": now - 125,
            }],
        )
        lines = render_lines(ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "[2m05s]" in combined


class TestFmtElapsed:
    def test_seconds(self):
        from mullet.hud.render import _fmt_elapsed
        assert _fmt_elapsed(42) == "42s"
        assert _fmt_elapsed(0) == "0s"
        assert _fmt_elapsed(59) == "59s"

    def test_minutes(self):
        from mullet.hud.render import _fmt_elapsed
        assert _fmt_elapsed(60) == "1m00s"
        assert _fmt_elapsed(125) == "2m05s"
        assert _fmt_elapsed(3599) == "59m59s"

    def test_hours(self):
        from mullet.hud.render import _fmt_elapsed
        assert _fmt_elapsed(3600) == "1h00m"
        assert _fmt_elapsed(3661) == "1h01m"
        assert _fmt_elapsed(7265) == "2h01m"


# ---------------------------------------------------------------------------
# render_lines — layout modes
# ---------------------------------------------------------------------------

class TestLayoutModes:
    def test_full_mode_multiple_lines(self, full_ctx):
        config = HudConfig(preset="focused", layout="full")
        lines = render_lines(full_ctx, config)
        assert len(lines) >= 3

    def test_minimal_mode_collapses_to_fewer_lines(self, full_ctx):
        config_full = HudConfig(preset="focused", layout="full")
        config_min = HudConfig(preset="focused", layout="minimal")
        lines_full = render_lines(full_ctx, config_full)
        lines_min = render_lines(full_ctx, config_min)
        # Minimal should not produce more section lines than full
        assert len(lines_min) <= len(lines_full)

    def test_minimal_preset_forces_minimal_layout(self, full_ctx):
        config = HudConfig(preset="minimal")
        lines = render_lines(full_ctx, config)
        # Section lines should be collapsed into one (plus directory / agents)
        # At most: 1 collapsed section + directory + per-agent
        section_count = sum(1 for ln in lines if "Directory" not in strip_ansi(ln) and "●" not in strip_ansi(ln))
        assert section_count <= 1

    def test_max_width_honoured_in_minimal(self, full_ctx):
        config = HudConfig(layout="minimal", max_width=30)
        lines = render_lines(full_ctx, config)
        first = strip_ansi(lines[0])
        assert len(first) <= 30 + 1  # +1 for ellipsis char


# ---------------------------------------------------------------------------
# render_hud — back-compat
# ---------------------------------------------------------------------------

class TestRenderHudBackCompat:
    def test_returns_string(self, full_ctx):
        result = render_hud(full_ctx)
        assert isinstance(result, str)

    def test_newline_joined(self, full_ctx):
        lines = render_lines(full_ctx)
        expected = "\n".join(lines)
        assert render_hud(full_ctx) == expected

    def test_none_config_uses_defaults(self, full_ctx):
        result = render_hud(full_ctx, None)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Per-provider spend in usage line
# ---------------------------------------------------------------------------

class TestProviderCosts5h:
    def test_kimi_spend_shown(self):
        from mullet.hud.render.lines.usage import render_usage_line
        from mullet.hud.colors import strip_ansi
        ctx = HudRenderContext(provider_costs_5h={"kimi": 0.42})
        line = render_usage_line(ctx)
        assert line is not None
        assert "kimi" in strip_ansi(line)
        assert "$0.42" in strip_ansi(line)

    def test_local_spend_not_shown(self):
        from mullet.hud.render.lines.usage import render_usage_line
        from mullet.hud.colors import strip_ansi
        ctx = HudRenderContext(provider_costs_5h={"local": 0.0, "kimi": 0.10})
        line = render_usage_line(ctx)
        assert "local" not in strip_ansi(line)
        assert "kimi" in strip_ansi(line)

    def test_zero_spend_not_shown(self):
        from mullet.hud.render.lines.usage import render_usage_line
        from mullet.hud.colors import strip_ansi
        ctx = HudRenderContext(provider_costs_5h={"kimi": 0.0, "gemini": 0.0})
        line = render_usage_line(ctx)
        # No spend means no spend section
        assert line is None or "kimi" not in strip_ansi(line or "")

    def test_multiple_providers_shown(self):
        from mullet.hud.render.lines.usage import render_usage_line
        from mullet.hud.colors import strip_ansi
        ctx = HudRenderContext(provider_costs_5h={"kimi": 0.50, "gemini": 0.03})
        line = render_usage_line(ctx)
        plain = strip_ansi(line or "")
        assert "kimi" in plain
        assert "gemini" in plain


class TestPerAgentSpendChip:
    def test_cost_chip_shown_when_cost_usd_set(self):
        ctx = HudRenderContext(
            running_agents=[{
                "provider": "swarm",
                "model": "kimi",
                "description": "refactor auth",
                "started_at": 1000000.0,
                "cost_usd": 0.04,
            }],
        )
        lines = render_lines(ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "$0.04" in combined

    def test_cost_chip_shown_as_cents_when_below_threshold(self):
        ctx = HudRenderContext(
            running_agents=[{
                "provider": "swarm",
                "model": "kimi",
                "description": "refactor auth",
                "started_at": 1000000.0,
                "cost_usd": 0.002,
            }],
        )
        lines = render_lines(ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "<$0.01" in combined

    def test_cost_chip_absent_when_cost_is_zero(self):
        ctx = HudRenderContext(
            running_agents=[{
                "provider": "swarm",
                "model": "kimi",
                "description": "refactor auth",
                "started_at": 1000000.0,
                "cost_usd": 0.0,
            }],
        )
        lines = render_lines(ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "$" not in combined

    def test_cost_chip_absent_when_cost_usd_missing(self):
        ctx = HudRenderContext(
            running_agents=[{
                "provider": "swarm",
                "model": "kimi",
                "description": "refactor auth",
                "started_at": 1000000.0,
            }],
        )
        lines = render_lines(ctx)
        combined = "\n".join(strip_ansi(ln) for ln in lines)
        assert "$" not in combined


class TestBudgetStatusBadge:
    def test_go_badge_shown_green(self):
        ctx = HudRenderContext(five_hour_pct=10.0, budget_status="go")
        from mullet.hud.render.lines.usage import render_usage_line
        line = render_usage_line(ctx)
        assert line is not None
        plain = strip_ansi(line)
        assert "● GO" in plain

    def test_throttle_badge_shown(self):
        ctx = HudRenderContext(five_hour_pct=50.0, budget_status="throttle")
        from mullet.hud.render.lines.usage import render_usage_line
        line = render_usage_line(ctx)
        assert line is not None
        plain = strip_ansi(line)
        assert "⚠ THROTTLE" in plain

    def test_pause_badge_shown(self):
        ctx = HudRenderContext(five_hour_pct=90.0, budget_status="pause")
        from mullet.hud.render.lines.usage import render_usage_line
        line = render_usage_line(ctx)
        assert line is not None
        plain = strip_ansi(line)
        assert "✗ PAUSE" in plain

    def test_no_badge_when_status_empty(self):
        ctx = HudRenderContext(five_hour_pct=30.0, budget_status="")
        from mullet.hud.render.lines.usage import render_usage_line
        line = render_usage_line(ctx)
        assert line is not None
        plain = strip_ansi(line)
        assert "GO" not in plain and "THROTTLE" not in plain and "PAUSE" not in plain
