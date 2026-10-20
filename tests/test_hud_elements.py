"""Tests for HUD agent display — render_agents, render_hud, format_reset_duration."""

from mullet.hud.types import HudRenderContext, HudConfig
from mullet.hud.elements import (
    render_agents, format_reset_duration, render_search_quality,
    render_context, render_mode, render_todos, render_session,
    render_git, render_model, render_tokens, render_rate_limits,
    render_reset_time, render_speed, _model_badge, render_thinking,
)
from mullet.hud.render import render_hud
from mullet.hud.colors import strip_ansi


# ---------------------------------------------------------------------------
# render_agents — headline element
# ---------------------------------------------------------------------------

def test_render_agents_no_agents_returns_none():
    ctx = HudRenderContext(running_agents=[], agent_count=0)
    assert render_agents(ctx) is None


def test_render_agents_no_list_falls_back_to_count():
    ctx = HudRenderContext(running_agents=[], agent_count=3)
    out = render_agents(ctx)
    assert out is not None
    assert "3" in strip_ansi(out)


def test_render_agents_single_shows_provider_and_model():
    ctx = HudRenderContext(
        running_agents=[{"provider": "kimi", "model": "kimi-k2-0711-preview", "description": "review auth"}],
        agent_count=1,
    )
    out = strip_ansi(render_agents(ctx))
    assert "kimi" in out
    assert "kimi-k2-0711-preview" in out


def test_render_agents_single_count_derived_from_list():
    ctx = HudRenderContext(
        running_agents=[{"provider": "kimi", "model": "kimi-k2-0711-preview"}],
        agent_count=99,  # stale count — should be overridden
    )
    out = strip_ansi(render_agents(ctx))
    # Should reflect 1 agent, not 99
    assert "Agent:" in out  # single-agent branch


def test_render_agents_multi_provider_grouping():
    ctx = HudRenderContext(
        running_agents=[
            {"provider": "kimi", "model": "kimi-k2-0711-preview"},
            {"provider": "kimi", "model": "kimi-k2-0711-preview"},
            {"provider": "kimi", "model": "kimi-k2-0711-preview"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ],
        agent_count=5,
    )
    out = strip_ansi(render_agents(ctx))
    assert "Agents: 5" in out
    assert "kimi×3" in out
    assert "gemini×2" in out


def test_render_agents_single_provider_multiple_agents():
    ctx = HudRenderContext(
        running_agents=[
            {"provider": "haiku", "model": "claude-haiku-4-5"},
            {"provider": "haiku", "model": "claude-haiku-4-5"},
        ],
        agent_count=2,
    )
    out = strip_ansi(render_agents(ctx))
    assert "Agents: 2" in out
    assert "haiku×2" in out


def test_render_agents_mixed_providers_single_counts():
    ctx = HudRenderContext(
        running_agents=[
            {"provider": "kimi", "model": "kimi-k2-0711-preview"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ],
        agent_count=2,
    )
    out = strip_ansi(render_agents(ctx))
    assert "Agents: 2" in out
    assert "kimi" in out
    assert "gemini" in out
    # No ×N suffix for count=1 providers
    assert "kimi×" not in out
    assert "gemini×" not in out


def test_render_agents_missing_fields_graceful():
    ctx = HudRenderContext(
        running_agents=[
            {"provider": "kimi"},         # no model
            {"model": "gpt-4"},           # no provider
            {},                            # nothing
        ],
        agent_count=3,
    )
    out = render_agents(ctx)
    assert out is not None
    assert "3" in strip_ansi(out)


# ---------------------------------------------------------------------------
# render_hud — per-agent lines
# ---------------------------------------------------------------------------

def test_render_hud_per_agent_lines_show_provider_model_task():
    ctx = HudRenderContext(
        running_agents=[
            {"provider": "kimi", "model": "kimi-k2-0711-preview", "description": "review auth.py"},
        ],
        agent_count=1,
    )
    config = HudConfig(preset="full")
    out = strip_ansi(render_hud(ctx, config))
    assert "kimi" in out
    assert "kimi-k2-0711-preview" in out
    assert "review auth.py" in out


def test_render_hud_multiple_agents_one_line_each():
    ctx = HudRenderContext(
        running_agents=[
            {"provider": "kimi", "model": "kimi-k2-0711-preview", "description": "task A"},
            {"provider": "gemini", "model": "gemini-2.5-flash", "description": "task B"},
        ],
        agent_count=2,
    )
    out = render_hud(ctx, HudConfig(preset="full"))
    lines = out.splitlines()
    agent_lines = [ln for ln in lines if "●" in ln]
    assert len(agent_lines) == 2


def test_render_hud_task_truncated_at_160_chars():
    long_task = "a" * 200
    ctx = HudRenderContext(
        running_agents=[{"provider": "kimi", "model": "kimi-k2", "description": long_task}],
        agent_count=1,
    )
    out = strip_ansi(render_hud(ctx, HudConfig(preset="full")))
    # Full 200-char task should not appear — truncated to 160 + ellipsis
    assert long_task not in out
    assert "a" * 160 in out


def test_render_hud_agents_headline_hidden_in_minimal_preset():
    # minimal preset disables the Agents: N headline, but per-agent lines still render
    ctx = HudRenderContext(
        running_agents=[{"provider": "kimi", "model": "kimi-k2", "description": "something"}],
        agent_count=1,
    )
    out = strip_ansi(render_hud(ctx, HudConfig(preset="minimal")))
    assert "Agents:" not in out  # headline suppressed
    assert "Agent:" not in out   # single-agent headline also suppressed


def test_render_hud_description_falls_back_to_task_key():
    ctx = HudRenderContext(
        running_agents=[{"provider": "kimi", "model": "kimi-k2", "task": "fallback task"}],
        agent_count=1,
    )
    out = strip_ansi(render_hud(ctx, HudConfig(preset="full")))
    assert "fallback task" in out


# ---------------------------------------------------------------------------
# format_reset_duration
# ---------------------------------------------------------------------------

class TestFormatResetDuration:
    def test_zero_returns_empty(self):
        assert format_reset_duration(0) == ""

    def test_negative_returns_empty(self):
        assert format_reset_duration(-1) == ""

    def test_1_second_rounds_to_1m(self):
        assert format_reset_duration(1) == "1m"

    def test_60_seconds_is_1m(self):
        assert format_reset_duration(60) == "1m"

    def test_59_seconds_rounds_up(self):
        assert format_reset_duration(59) == "1m"

    def test_120_seconds_is_2m(self):
        assert format_reset_duration(120) == "2m"

    def test_30_minutes(self):
        assert format_reset_duration(30 * 60) == "30m"

    def test_1_hour(self):
        assert format_reset_duration(60 * 60) == "1h"

    def test_1h_30m(self):
        assert format_reset_duration(90 * 60) == "1h 30m"

    def test_2h_exact(self):
        assert format_reset_duration(2 * 60 * 60) == "2h"

    def test_1_day(self):
        assert format_reset_duration(24 * 60 * 60) == "1d"

    def test_1d_1h(self):
        assert format_reset_duration(25 * 60 * 60) == "1d 1h"

    def test_2d_exact(self):
        assert format_reset_duration(48 * 60 * 60) == "2d"


# ---------------------------------------------------------------------------
# render_search_quality
# ---------------------------------------------------------------------------

class TestRenderSearchQuality:
    def test_contains_result_count(self):
        result = render_search_quality(10, 42.0, 0.3, 0.9, 3)
        assert "10" in result

    def test_contains_source_count(self):
        result = render_search_quality(5, 15.0, 0.1, 0.8, 4)
        assert "4" in result

    def test_returns_string(self):
        assert isinstance(render_search_quality(0, 0.0, 0.0, 0.0, 0), str)

    def test_scores_rounded_to_2dp(self):
        result = render_search_quality(1, 5.0, 0.12345, 0.98765, 1)
        assert "0.12" in result
        assert "0.99" in result

    def test_time_no_decimal(self):
        result = render_search_quality(3, 99.9, 0.5, 0.8, 2)
        # ":.0f" format → "100ms" or "99ms"
        assert "99ms" in result or "100ms" in result


# ---------------------------------------------------------------------------
# HudRenderContext defaults
# ---------------------------------------------------------------------------

class TestHudRenderContextDefaults:
    def test_numeric_defaults_zero(self):
        ctx = HudRenderContext()
        assert ctx.context_pct == 0.0
        assert ctx.agent_count == 0
        assert ctx.todos_done == 0
        assert ctx.todos_total == 0
        assert ctx.total_tokens == 0

    def test_bool_defaults_false(self):
        ctx = HudRenderContext()
        assert ctx.ralph_active is False
        assert ctx.autopilot_active is False
        assert ctx.ultrawork_active is False
        assert ctx.git_dirty is False
        assert ctx.thinking_active is False

    def test_list_defaults_empty(self):
        ctx = HudRenderContext()
        assert ctx.running_agents == []
        assert ctx.running_tools == []
        assert ctx.completed_tools == []

    def test_optional_pcts_none(self):
        ctx = HudRenderContext()
        assert ctx.five_hour_pct is None
        assert ctx.weekly_pct is None

    def test_speed_mode_default(self):
        ctx = HudRenderContext()
        assert ctx.speed_mode == "economy"


class TestRenderContext:
    def test_zero_pct_returns_none(self):
        ctx = HudRenderContext(context_pct=0)
        assert render_context(ctx) is None

    def test_positive_pct_renders(self):
        ctx = HudRenderContext(context_pct=50)
        out = render_context(ctx)
        assert out is not None
        assert "50%" in strip_ansi(out)


class TestRenderMode:
    def test_ralph_active(self):
        ctx = HudRenderContext(ralph_active=True, ralph_iter=2, ralph_max=5)
        out = strip_ansi(render_mode(ctx))
        assert "Ralph: 2/5" in out

    def test_autopilot_active(self):
        ctx = HudRenderContext(autopilot_active=True, autopilot_phase="planning")
        out = strip_ansi(render_mode(ctx))
        assert "Autopilot: planning" in out

    def test_ultrawork_active(self):
        ctx = HudRenderContext(ultrawork_active=True, ultrawork_tasks=3)
        out = strip_ansi(render_mode(ctx))
        assert "Ultrawork: 3 tasks" in out

    def test_no_mode_returns_none(self):
        ctx = HudRenderContext()
        assert render_mode(ctx) is None


class TestRenderTodos:
    def test_zero_total_returns_none(self):
        ctx = HudRenderContext(todos_total=0)
        assert render_todos(ctx) is None

    def test_all_done(self):
        ctx = HudRenderContext(todos_done=5, todos_total=5)
        out = strip_ansi(render_todos(ctx))
        assert "5/5" in out

    def test_partial(self):
        ctx = HudRenderContext(todos_done=2, todos_total=5)
        out = strip_ansi(render_todos(ctx))
        assert "2/5" in out

    def test_none_done(self):
        ctx = HudRenderContext(todos_done=0, todos_total=3)
        out = strip_ansi(render_todos(ctx))
        assert "0/3" in out


class TestRenderSession:
    def test_zero_minutes_returns_none(self):
        ctx = HudRenderContext(session_minutes=0)
        assert render_session(ctx) is None

    def test_short_session_green(self):
        ctx = HudRenderContext(session_minutes=15)
        out = strip_ansi(render_session(ctx))
        assert "15m" in out

    def test_long_session_red(self):
        ctx = HudRenderContext(session_minutes=90)
        out = strip_ansi(render_session(ctx))
        assert "90m" in out


class TestRenderGit:
    def test_no_branch_returns_none(self):
        ctx = HudRenderContext(git_branch="")
        assert render_git(ctx) is None

    def test_clean_branch(self):
        ctx = HudRenderContext(git_branch="main", git_dirty=False)
        out = strip_ansi(render_git(ctx))
        assert "main" in out
        assert "*" not in out

    def test_dirty_branch(self):
        ctx = HudRenderContext(git_branch="feat/x", git_dirty=True)
        out = strip_ansi(render_git(ctx))
        assert "feat/x" in out
        assert "*" in out


class TestRenderModel:
    def test_no_model_returns_none(self):
        ctx = HudRenderContext(model_name="")
        assert render_model(ctx) is None

    def test_model_renders(self):
        ctx = HudRenderContext(model_name="claude-sonnet")
        out = strip_ansi(render_model(ctx))
        assert "claude-sonnet" in out


class TestRenderTokens:
    def test_zero_tokens_returns_none(self):
        ctx = HudRenderContext(total_tokens=0)
        assert render_tokens(ctx) is None

    def test_total_only(self):
        ctx = HudRenderContext(total_tokens=1000)
        out = strip_ansi(render_tokens(ctx))
        assert "1000" in out

    def test_with_last_in_out(self):
        ctx = HudRenderContext(total_tokens=1000, last_in=50, last_out=30)
        out = strip_ansi(render_tokens(ctx))
        assert "1000" in out
        assert "50" in out
        assert "30" in out


class TestRenderRateLimits:
    def test_unknown(self):
        ctx = HudRenderContext(five_hour_pct=None, weekly_pct=None)
        out = strip_ansi(render_rate_limits(ctx))
        assert "?" in out

    def test_five_hour_only(self):
        ctx = HudRenderContext(five_hour_pct=50.0, weekly_pct=None)
        out = strip_ansi(render_rate_limits(ctx))
        assert "50%" in out
        assert "7d: ?" in out

    def test_both_known(self):
        ctx = HudRenderContext(five_hour_pct=30.0, weekly_pct=40.0)
        out = strip_ansi(render_rate_limits(ctx))
        assert "30%" in out
        assert "40%" in out

    def test_sonnet_shown_when_positive(self):
        ctx = HudRenderContext(five_hour_pct=10.0, weekly_pct=10.0, sonnet_weekly_pct=20.0)
        out = strip_ansi(render_rate_limits(ctx))
        assert "Sonnet: 20%" in out

    def test_sonnet_hidden_when_zero(self):
        ctx = HudRenderContext(five_hour_pct=10.0, weekly_pct=10.0, sonnet_weekly_pct=0.0)
        out = strip_ansi(render_rate_limits(ctx))
        assert "Sonnet" not in out


class TestRenderResetTime:
    def test_no_reset_returns_none(self):
        ctx = HudRenderContext(rate_limit_reset_s=0)
        assert render_reset_time(ctx) is None

    def test_positive_reset_renders(self):
        ctx = HudRenderContext(rate_limit_reset_s=300)
        out = strip_ansi(render_reset_time(ctx))
        assert "5m" in out


class TestRenderSpeed:
    def test_economy_returns_none(self):
        ctx = HudRenderContext(speed_mode="economy")
        assert render_speed(ctx) is None

    def test_fast_renders(self):
        ctx = HudRenderContext(speed_mode="fast")
        out = render_speed(ctx)
        assert out is not None


class TestModelBadge:
    def test_both_missing(self):
        assert _model_badge(None, None) == ""

    def test_provider_only(self):
        assert strip_ansi(_model_badge("kimi", None)) == " kimi"

    def test_model_only(self):
        assert strip_ansi(_model_badge(None, "k2")) == " k2"

    def test_both_present(self):
        assert strip_ansi(_model_badge("kimi", "k2")) == " kimi/k2"


class TestRenderThinking:
    def test_inactive_returns_none(self):
        ctx = HudRenderContext(thinking_active=False)
        assert render_thinking(ctx) is None

    def test_active_renders(self):
        ctx = HudRenderContext(thinking_active=True, main_provider="anthropic", model_name="opus")
        out = strip_ansi(render_thinking(ctx))
        assert "Thinking..." in out
        assert "anthropic/opus" in out
