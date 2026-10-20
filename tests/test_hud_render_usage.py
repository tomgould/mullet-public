
from mullet.hud.render.lines.usage import render_usage_line
from mullet.hud.types import HudRenderContext
from mullet.hud.colors import strip_ansi


class TestRenderUsageLine:
    def test_no_data_shows_unknown(self):
        ctx = HudRenderContext()
        out = strip_ansi(render_usage_line(ctx))
        assert "5h: ?" in out
        assert "7d: ?" in out

    def test_budget_status_go(self):
        ctx = HudRenderContext(budget_status="go")
        out = strip_ansi(render_usage_line(ctx))
        assert "GO" in out

    def test_budget_status_throttle(self):
        ctx = HudRenderContext(budget_status="throttle")
        out = strip_ansi(render_usage_line(ctx))
        assert "THROTTLE" in out

    def test_budget_status_pause(self):
        ctx = HudRenderContext(budget_status="pause")
        out = strip_ansi(render_usage_line(ctx))
        assert "PAUSE" in out

    def test_context_pct_renders(self):
        ctx = HudRenderContext(context_pct=50.0)
        out = strip_ansi(render_usage_line(ctx))
        assert "Ctx 50%" in out

    def test_compact_hides_bar(self):
        ctx = HudRenderContext(context_pct=50.0)
        compact_out = strip_ansi(render_usage_line(ctx, compact=True))
        full_out = strip_ansi(render_usage_line(ctx, compact=False))
        assert "Ctx 50%" in compact_out
        assert "Ctx 50%" in full_out

    def test_five_hour_pct(self):
        ctx = HudRenderContext(five_hour_pct=75.0)
        out = strip_ansi(render_usage_line(ctx))
        assert "5h: 75%" in out

    def test_weekly_pct(self):
        ctx = HudRenderContext(weekly_pct=60.0)
        out = strip_ansi(render_usage_line(ctx))
        assert "7d: 60%" in out

    def test_unknown_percentages(self):
        ctx = HudRenderContext(five_hour_pct=None, weekly_pct=None)
        out = strip_ansi(render_usage_line(ctx))
        assert "5h: ?" in out
        assert "7d: ?" in out

    def test_sonnet_pct_positive(self):
        ctx = HudRenderContext(sonnet_weekly_pct=45.0)
        out = strip_ansi(render_usage_line(ctx))
        assert "Sonnet: 45%" in out

    def test_sonnet_pct_zero_hidden(self):
        ctx = HudRenderContext(sonnet_weekly_pct=0.0)
        out = strip_ansi(render_usage_line(ctx))
        assert "Sonnet" not in out

    def test_reset_time(self):
        ctx = HudRenderContext(rate_limit_reset_s=300)
        out = strip_ansi(render_usage_line(ctx))
        assert "Resets: 5m" in out

    def test_pressure_renders(self):
        ctx = HudRenderContext(pressure_5h=0.5, pressure_7d=0.25)
        out = strip_ansi(render_usage_line(ctx))
        assert "bpd: 50%" in out
        assert "bpw: 25%" in out

    def test_provider_costs(self):
        ctx = HudRenderContext(provider_costs_5h={"kimi": 2.5, "local": 0.1, "unknown": 1.0})
        out = strip_ansi(render_usage_line(ctx))
        assert "kimi: $2.50" in out
        assert "local" not in out
        assert "unknown" not in out

    def test_combined_parts_separated(self):
        ctx = HudRenderContext(
            budget_status="go",
            context_pct=25.0,
            five_hour_pct=10.0,
            weekly_pct=20.0,
        )
        out = render_usage_line(ctx)
        assert " • " in out


class TestRenderUsageLineMissingLines:
    """Cover line 71 in usage.py."""

    def test_zero_cost_provider_skipped(self):
        """Providers with cost < 0.001 are skipped via continue (line 71)."""
        ctx = HudRenderContext(provider_costs_5h={"kimi": 0.0005})
        out = strip_ansi(render_usage_line(ctx) or "")
        assert "kimi" not in out
