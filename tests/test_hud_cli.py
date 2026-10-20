"""Tests for mullet.hud.cli — dependency-injected entry point.

No real stdin, no subprocess, no network. Everything flows through HudDeps.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


from mullet.hud.cli import HudDeps, build_hud_output, main
from mullet.hud.types import HudRenderContext
from mullet.hud.usage import RateLimits


def _fixed_now() -> datetime:
    return datetime(2026, 4, 24, 20, 0, 0, tzinfo=timezone.utc)


def _stub_context(cwd: str) -> HudRenderContext:
    return HudRenderContext(cwd=cwd)


def _make_deps(
    *,
    stdin: Optional[dict] = None,
    stdin_limits: Optional[RateLimits] = None,
    fallback_limits: Optional[RateLimits] = None,
    scan_result: Optional[list[dict]] = None,
    captured: Optional[list[str]] = None,
) -> HudDeps:
    captured = captured if captured is not None else []
    return HudDeps(
        read_stdin=lambda: stdin or {},
        parse_stdin_limits=lambda data: stdin_limits,
        fetch_usage_fallback=lambda: fallback_limits,
        scan_processes=lambda: scan_result or [],
        build_context=_stub_context,
        render=lambda ctx, cfg: f"CTX={ctx.context_pct} 5H={ctx.five_hour_pct} TOK={ctx.total_tokens} AGENTS={len(ctx.running_agents)} MODEL={ctx.model_name}",
        now=_fixed_now,
        get_terminal_cols=lambda: 120,
        emit=captured.append,
    )


def test_main_emits_rendered_output():
    captured: list[str] = []
    deps = _make_deps(stdin={"model": {"display_name": "Opus 4.7"}}, captured=captured)
    main(deps)
    assert captured
    assert "MODEL=Opus 4.7" in captured[0]


def test_build_hud_empty_stdin_still_renders():
    deps = _make_deps(stdin={})
    out = build_hud_output({}, deps)
    assert "CTX=" in out  # renders something even without stdin


def test_stdin_limits_preferred_over_fallback():
    stdin_rl = RateLimits(five_hour_pct=42.0, weekly_pct=12.0)
    fallback_rl = RateLimits(five_hour_pct=99.0, weekly_pct=99.0)
    deps = _make_deps(stdin={}, stdin_limits=stdin_rl, fallback_limits=fallback_rl)
    out = build_hud_output({}, deps)
    assert "5H=42.0" in out


def test_falls_back_to_fetch_usage_when_stdin_limits_none():
    fallback_rl = RateLimits(five_hour_pct=88.0, weekly_pct=55.0)
    deps = _make_deps(stdin={}, stdin_limits=None, fallback_limits=fallback_rl)
    out = build_hud_output({}, deps)
    assert "5H=88.0" in out


def test_no_limits_leaves_ctx_untouched():
    deps = _make_deps(stdin={}, stdin_limits=None, fallback_limits=None)
    out = build_hud_output({}, deps)
    assert "5H=None" in out


def test_ctx_pct_from_native_used_percentage():
    stdin = {"context_window": {"used_percentage": 37.5}}
    deps = _make_deps(stdin=stdin)
    out = build_hud_output(stdin, deps)
    assert "CTX=37.5" in out


def test_ctx_pct_fallback_from_current_usage():
    stdin = {
        "context_window": {
            "used_percentage": 0,
            "context_window_size": 1000,
            "current_usage": {"input_tokens": 100, "cache_read_input_tokens": 150},
        }
    }
    deps = _make_deps(stdin=stdin)
    out = build_hud_output(stdin, deps)
    assert "CTX=25.0" in out  # (100 + 150) / 1000 = 25%
    assert "TOK=250" in out


def test_total_tokens_only_populated_when_state_had_none():
    stdin = {"context_window": {"current_usage": {"input_tokens": 77, "cache_creation_input_tokens": 23}}}
    deps = _make_deps(stdin=stdin)
    out = build_hud_output(stdin, deps)
    assert "TOK=100" in out


def test_scan_processes_results_wired_into_running_agents():
    deps = _make_deps(stdin={}, scan_result=[
        {"provider": "kimi", "model": "k2.6", "description": "fix foo"},
        {"provider": "kimi", "model": "k2.6", "description": "fix bar"},
    ])
    out = build_hud_output({}, deps)
    assert "AGENTS=2" in out


def test_rate_limit_reset_seconds_computed_from_now():
    future_iso = datetime(2026, 4, 24, 22, 30, 0, tzinfo=timezone.utc).isoformat()
    rl = RateLimits(five_hour_pct=5.0, five_hour_resets=future_iso)
    captured_ctx: dict = {}

    def capture_render(ctx, cfg):
        captured_ctx["ctx"] = ctx
        return ""

    deps = HudDeps(
        read_stdin=lambda: {},
        parse_stdin_limits=lambda data: rl,
        fetch_usage_fallback=lambda: None,
        scan_processes=lambda: [],
        build_context=_stub_context,
        render=capture_render,
        now=_fixed_now,
        get_terminal_cols=lambda: 120,
        emit=lambda _: None,
    )
    main(deps)
    # 2h 30m from 20:00 UTC = 9000s
    assert captured_ctx["ctx"].rate_limit_reset_s == 9000


def test_stdin_parse_exception_falls_through_to_fetch():
    def boom(_data):
        raise RuntimeError("bad json")

    fallback_rl = RateLimits(five_hour_pct=11.0)
    deps = HudDeps(
        read_stdin=lambda: {},
        parse_stdin_limits=boom,
        fetch_usage_fallback=lambda: fallback_rl,
        scan_processes=lambda: [],
        build_context=_stub_context,
        render=lambda ctx, cfg: f"5H={ctx.five_hour_pct}",
        now=_fixed_now,
        get_terminal_cols=lambda: 120,
        emit=lambda _: None,
    )
    out = build_hud_output({}, deps)
    assert "5H=11.0" in out


def test_model_name_prefers_display_name():
    stdin = {"model": {"display_name": "Opus 4.7", "id": "claude-opus-4-7"}}
    deps = _make_deps(stdin=stdin)
    out = build_hud_output(stdin, deps)
    assert "MODEL=Opus 4.7" in out


def test_model_name_falls_back_to_id():
    stdin = {"model": {"id": "claude-haiku-4-5-20251001"}}
    deps = _make_deps(stdin=stdin)
    out = build_hud_output(stdin, deps)
    assert "MODEL=claude-haiku-4-5-20251001" in out
