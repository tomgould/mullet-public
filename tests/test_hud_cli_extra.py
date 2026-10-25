"""Extra tests for mullet.hud.cli to cover previously-uncovered lines.

Uses mocking for all external boundaries (subprocess, os, sys.stdin, etc.).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch


from mullet.hud.cli import (
    HudDeps,
    _apply_limits,
    _compute_context_pct,
    _default_get_terminal_cols,
    _default_now,
    _default_read_stdin,
    _default_scan_processes,
    _desc_from_subcommand,
    build_hud_output,
    main,
)
from mullet.hud.transcript import AgentCall, TodoItem, ToolCall, TranscriptSnapshot
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
        render=lambda ctx, cfg: (
            f"CTX={ctx.context_pct} 5H={ctx.five_hour_pct} "
            f"TOK={ctx.total_tokens} AGENTS={len(ctx.running_agents)} "
            f"RTOOLS={len(ctx.running_tools)} CTTOOLS={len(ctx.completed_tools)} "
            f"MODEL={ctx.model_name}"
        ),
        now=_fixed_now,
        get_terminal_cols=lambda: 120,
        emit=captured.append,
    )


# ---------------------------------------------------------------------------
# _default_read_stdin
# ---------------------------------------------------------------------------


def test_default_read_stdin_tty_returns_empty():
    with patch.object(sys.stdin, "isatty", return_value=True):
        assert _default_read_stdin() == {}


def test_default_read_stdin_empty_returns_empty():
    with patch.object(sys.stdin, "isatty", return_value=False):
        with patch.object(sys.stdin, "read", return_value="   "):
            assert _default_read_stdin() == {}


def test_default_read_stdin_valid_json():
    payload = {"model": "claude-opus", "cwd": "/tmp"}
    with patch.object(sys.stdin, "isatty", return_value=False):
        with patch.object(sys.stdin, "read", return_value=json.dumps(payload)):
            assert _default_read_stdin() == payload


def test_default_read_stdin_bad_json_returns_empty():
    with patch.object(sys.stdin, "isatty", return_value=False):
        with patch.object(sys.stdin, "read", return_value="not json"):
            assert _default_read_stdin() == {}


def test_default_read_stdin_oserror_returns_empty():
    with patch.object(sys.stdin, "isatty", return_value=False):
        with patch.object(sys.stdin, "read", side_effect=OSError("broken pipe")):
            assert _default_read_stdin() == {}


# ---------------------------------------------------------------------------
# _default_get_terminal_cols
# ---------------------------------------------------------------------------


def test_default_get_terminal_cols():
    with patch("mullet.hud.cli.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
        assert _default_get_terminal_cols() == 80


# ---------------------------------------------------------------------------
# _desc_from_subcommand
# ---------------------------------------------------------------------------


def test_desc_empty_command():
    assert _desc_from_subcommand("") == "running"


def test_desc_no_mullet_token():
    assert _desc_from_subcommand("python foo.py") == "python foo.py"


def test_desc_known_verbs():
    assert _desc_from_subcommand("mullet reindex") == "reindexing"
    assert _desc_from_subcommand("mullet sync") == "syncing"
    assert _desc_from_subcommand("mullet ingest") == "ingesting"
    assert _desc_from_subcommand("mullet crawl") == "crawling"
    assert _desc_from_subcommand("mullet build") == "building"
    assert _desc_from_subcommand("mullet prune") == "pruning"
    assert _desc_from_subcommand("mullet export") == "exporting"
    assert _desc_from_subcommand("mullet status") == "status"
    assert _desc_from_subcommand("mullet budget") == "budget check"
    assert _desc_from_subcommand("mullet routing") == "routing"
    assert _desc_from_subcommand("mullet run") == "running"
    assert _desc_from_subcommand("mullet delegate") == "delegating"
    assert _desc_from_subcommand("mullet swarm") == "swarming"
    assert _desc_from_subcommand("mullet search") == "searching"
    assert _desc_from_subcommand("mullet trawl") == "trawl"
    assert _desc_from_subcommand("mullet observations") == "observations"
    assert _desc_from_subcommand("mullet decisions") == "decisions"
    assert _desc_from_subcommand("mullet review-session") == "review-session"


def test_desc_pool_extraction():
    assert _desc_from_subcommand("mullet reindex --pool mypool") == "reindexing mypool"
    assert _desc_from_subcommand("mullet sync --pool other") == "syncing other"
    assert _desc_from_subcommand("mullet prune -p fast") == "pruning fast"
    assert _desc_from_subcommand('mullet export --pool "big pool"') == 'exporting "big'


def test_desc_fallback_target():
    assert _desc_from_subcommand("mullet ingest src/") == "ingesting src/"
    # remaining filters out flags, so openrouter (first non-flag after subcmd) is target
    assert _desc_from_subcommand("mullet run --provider openrouter script.py") == "running openrouter"


def test_desc_unknown_verb():
    assert _desc_from_subcommand("mullet foo bar") == "foo bar"


def test_desc_with_path_prefix():
    assert _desc_from_subcommand("/usr/bin/mullet run task") == "running task"
    assert _desc_from_subcommand("/home/user/.local/bin/mullet status") == "status"


# ---------------------------------------------------------------------------
# _default_scan_processes
# ---------------------------------------------------------------------------


def _make_ps_line(pid: str, stat: str, command: str) -> str:
    # ps aux columns: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
    return f"user   {pid}  0.0  0.1 1234 567 pts/0  {stat} 10:00 0:00 {command}"


def test_scan_processes_subprocess_exception_returns_empty():
    with patch("mullet.hud.cli.subprocess.run", side_effect=FileNotFoundError("no ps")):
        assert _default_scan_processes() == []


def test_scan_processes_no_mullet_lines():
    stdout = "user   1  0.0  0.1 1234 567 pts/0  S 10:00 0:00 bash\n"
    mock_ps = MagicMock(stdout=stdout)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        assert _default_scan_processes() == []


def test_scan_processes_skips_ps_aux_and_bash():
    lines = [
        _make_ps_line("1", "S", "ps aux"),
        _make_ps_line("2", "S", "/bin/bash mullet run"),
        _make_ps_line("3", "S", "mullet-hud"),
        _make_ps_line("4", "S", "mullet mcp"),
        _make_ps_line("5", "S", "pip install mullet"),
    ]
    mock_ps = MagicMock(stdout="\n".join(lines))
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        assert _default_scan_processes() == []


def test_scan_processes_skips_short_lines():
    # Must contain "mullet" to get past the first filter, but be short enough
    # to fail the column count check.
    stdout = "user   1  0.0  0.1 1234\n"
    mock_ps = MagicMock(stdout=stdout)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        assert _default_scan_processes() == []


def test_scan_processes_skips_short_cols_with_mullet():
    # Contains "mullet" but < 10 columns after split
    stdout = "user   1  0.0  mullet\n"
    mock_ps = MagicMock(stdout=stdout)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        assert _default_scan_processes() == []


def test_scan_processes_skips_cmd_part_without_mullet_word():
    # Full line contains "mullet" (in "mullettest") but cmd_part has no \bmullet\b
    line = _make_ps_line("1", "S", "python -m mullettest run")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", return_value=None):
            assert _default_scan_processes() == []


def test_scan_processes_skips_non_mullet_cmd():
    # cmd_part contains \bmullet\b but is_mullet_cmd is False
    line = _make_ps_line("1", "S", "foo mullet.bar")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", return_value=None):
            assert _default_scan_processes() == []


def test_scan_processes_skips_zombies():
    line = _make_ps_line("1", "Z", "python -m mullet run")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        assert _default_scan_processes() == []


def test_scan_processes_skips_process_lookup_error():
    line = _make_ps_line("999", "S", "python -m mullet run")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", side_effect=ProcessLookupError):
            assert _default_scan_processes() == []


def test_scan_processes_includes_on_permission_error():
    line = _make_ps_line("999", "S", "python -m mullet run")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", side_effect=PermissionError):
            result = _default_scan_processes()
    assert len(result) == 1
    assert result[0]["description"] == "running"


def test_scan_processes_includes_on_value_error():
    line = _make_ps_line("badpid", "S", "python -m mullet run")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", side_effect=ValueError):
            result = _default_scan_processes()
    assert len(result) == 1


def test_scan_processes_skips_cmd_without_mullet():
    line = _make_ps_line("1", "S", "python -m something_else run")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", return_value=None):
            assert _default_scan_processes() == []


def test_scan_processes_python_mullet_path():
    line = _make_ps_line("1", "S", "python -m mullet run task")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", return_value=None):
            result = _default_scan_processes()
    assert len(result) == 1
    assert result[0]["description"] == "running task"


def test_scan_processes_extracts_provider_and_model():
    line = _make_ps_line(
        "1", "S", "python -m mullet run --provider openrouter --model gpt-4 task"
    )
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", return_value=None):
            result = _default_scan_processes()
    assert result[0]["provider"] == "openrouter"
    assert result[0]["model"] == "gpt-4"


def test_scan_processes_extracts_quoted_description():
    line = _make_ps_line(
        "1", "S", 'python -m mullet run "fix the bug in parser"'
    )
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", return_value=None):
            result = _default_scan_processes()
    assert result[0]["description"] == "fix the bug in parser"


def test_scan_processes_single_quoted_description():
    line = _make_ps_line(
        "1", "S", "python -m mullet run 'do something great'"
    )
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", return_value=None):
            result = _default_scan_processes()
    assert result[0]["description"] == "do something great"


def test_scan_processes_fallback_desc_when_no_quote():
    line = _make_ps_line("1", "S", "python -m mullet search pattern")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", return_value=None):
            result = _default_scan_processes()
    assert result[0]["description"] == "searching pattern"


def test_scan_processes_default_provider_model():
    line = _make_ps_line("1", "S", "python -m mullet run task")
    mock_ps = MagicMock(stdout=line)
    with patch("mullet.hud.cli.subprocess.run", return_value=mock_ps):
        with patch("mullet.hud.cli.os.kill", return_value=None):
            result = _default_scan_processes()
    assert result[0]["provider"] == "local"
    assert result[0]["model"] == "default"


# ---------------------------------------------------------------------------
# _default_now
# ---------------------------------------------------------------------------


def test_default_now():
    before = datetime.now(timezone.utc)
    result = _default_now()
    after = datetime.now(timezone.utc)
    assert before <= result <= after


# ---------------------------------------------------------------------------
# build_hud_output — exception resilience
# ---------------------------------------------------------------------------


def test_cleanup_stale_agents_exception_swallowed():
    """Lines 228-229: exception in cleanup_stale_agents is ignored."""
    from mullet.hud import cli as cli_module

    with patch.object(cli_module, "cleanup_stale_agents", side_effect=RuntimeError("boom")):
        deps = _make_deps(stdin={})
        out = build_hud_output({}, deps)
    assert "CTX=" in out


def test_fetch_usage_fallback_exception_swallowed():
    """Lines 257-258: exception in fetch_usage_fallback is logged and ignored."""

    def boom():
        raise RuntimeError("network down")

    deps = _make_deps(stdin={}, fallback_limits=None)
    deps.fetch_usage_fallback = boom
    out = build_hud_output({}, deps)
    assert "5H=None" in out


def test_parse_transcript_exception_swallowed():
    """Lines 268-269: exception in parse_transcript is logged and ignored."""

    def boom(_path: str):
        raise RuntimeError("bad file")

    deps = _make_deps(stdin={"transcript_path": "/fake/path.jsonl"})
    deps.parse_transcript = boom
    out = build_hud_output({"transcript_path": "/fake/path.jsonl"}, deps)
    assert "CTX=" in out


# ---------------------------------------------------------------------------
# build_hud_output — completed tools with end_time
# ---------------------------------------------------------------------------


def test_completed_tools_duration_with_end_time():
    """Lines 300-304: completed tools calculate duration_ms when end_time exists."""
    start = datetime(2026, 4, 24, 19, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 24, 19, 0, 5, tzinfo=timezone.utc)  # 5 seconds
    snap = TranscriptSnapshot(
        completed_tools=[
            ToolCall(
                id="t1",
                name="bash",
                status="completed",
                start_time=start,
                end_time=end,
                target="echo hi",
                provider="openrouter",
                model="gpt-4",
            )
        ]
    )

    captured_ctx: dict = {}

    def capture_render(ctx, cfg):
        captured_ctx["ctx"] = ctx
        return f"CTTOOLS={len(ctx.completed_tools)}"

    deps = HudDeps(
        read_stdin=lambda: {"transcript_path": "/fake/t.jsonl"},
        parse_stdin_limits=lambda _: None,
        fetch_usage_fallback=lambda: None,
        scan_processes=lambda: [],
        parse_transcript=lambda _: snap,
        build_context=_stub_context,
        render=capture_render,
        now=_fixed_now,
        get_terminal_cols=lambda: 120,
        emit=lambda _: None,
    )
    build_hud_output({"transcript_path": "/fake/t.jsonl"}, deps)
    ctx = captured_ctx["ctx"]
    assert len(ctx.completed_tools) == 1
    assert ctx.completed_tools[0]["duration_ms"] == 5000


def test_completed_tools_duration_without_end_time():
    """Line 303: completed tools get duration_ms=0 when end_time is None."""
    start = datetime(2026, 4, 24, 19, 0, 0, tzinfo=timezone.utc)
    snap = TranscriptSnapshot(
        completed_tools=[
            ToolCall(
                id="t1",
                name="bash",
                status="completed",
                start_time=start,
                end_time=None,
                target="echo hi",
            )
        ]
    )

    captured_ctx: dict = {}

    def capture_render(ctx, cfg):
        captured_ctx["ctx"] = ctx
        return f"CTTOOLS={len(ctx.completed_tools)}"

    deps = HudDeps(
        read_stdin=lambda: {"transcript_path": "/fake/t.jsonl"},
        parse_stdin_limits=lambda _: None,
        fetch_usage_fallback=lambda: None,
        scan_processes=lambda: [],
        parse_transcript=lambda _: snap,
        build_context=_stub_context,
        render=capture_render,
        now=_fixed_now,
        get_terminal_cols=lambda: 120,
        emit=lambda _: None,
    )
    build_hud_output({"transcript_path": "/fake/t.jsonl"}, deps)
    ctx = captured_ctx["ctx"]
    assert len(ctx.completed_tools) == 1
    assert ctx.completed_tools[0]["duration_ms"] == 0


def test_session_minutes_from_transcript_snapshot():
    """Lines 334-335: session_minutes computed from snapshot.session_started_at."""
    start = datetime(2026, 4, 24, 18, 30, 0, tzinfo=timezone.utc)  # 90 min before now
    snap = TranscriptSnapshot(session_started_at=start)

    captured_ctx: dict = {}

    def capture_render(ctx, cfg):
        captured_ctx["ctx"] = ctx
        return "ok"

    deps = HudDeps(
        read_stdin=lambda: {"transcript_path": "/fake/t.jsonl"},
        parse_stdin_limits=lambda _: None,
        fetch_usage_fallback=lambda: None,
        scan_processes=lambda: [],
        parse_transcript=lambda _: snap,
        build_context=_stub_context,
        render=capture_render,
        now=_fixed_now,
        get_terminal_cols=lambda: 120,
        emit=lambda _: None,
    )
    build_hud_output({"transcript_path": "/fake/t.jsonl"}, deps)
    ctx = captured_ctx["ctx"]
    assert ctx.session_minutes == 90.0


# ---------------------------------------------------------------------------
# main() default deps instantiation
# ---------------------------------------------------------------------------


def test_main_with_none_deps_instantiates_hud_deps():
    """Line 349: main() creates HudDeps when deps is None."""
    mock_deps = MagicMock()
    mock_deps.read_stdin.return_value = {}
    mock_deps.render.return_value = "rendered"

    with patch("mullet.hud.cli.HudDeps", return_value=mock_deps):
        main(deps=None)

    mock_deps.read_stdin.assert_called_once()
    mock_deps.emit.assert_called_once_with("rendered")


# ---------------------------------------------------------------------------
# Missing-line coverage
# ---------------------------------------------------------------------------


class TestHudCliMissingLines:
    """Cover specific previously-uncovered lines in mullet/hud/cli.py."""

    def test_compute_context_pct_used_percentage(self):
        """Line 199: return float(pct) when used_percentage > 0."""
        assert _compute_context_pct({"used_percentage": 42.5}) == 42.5

    def test_compute_context_pct_size_and_total(self):
        """Line 204: return min(100.0, ...) when size and total are present."""
        assert _compute_context_pct(
            {"context_window_size": 1000, "current_usage": {"a": 500}}
        ) == 50.0

    def test_apply_limits_with_reset(self):
        """Lines 219-220: parse_resets_at returns a valid datetime."""
        ctx = HudRenderContext()
        limits = RateLimits(
            five_hour_pct=50.0,
            five_hour_resets="2026-04-24T21:00:00+00:00",
        )
        _apply_limits(ctx, limits, _fixed_now())
        assert ctx.five_hour_pct == 50.0
        assert ctx.rate_limit_reset_s == 3600

    def test_total_tokens_from_current_usage(self):
        """Line 247: ctx.total_tokens set from _sum_current_usage when > 0."""
        captured_ctx: dict = {}

        def capture_render(ctx, cfg):
            captured_ctx["ctx"] = ctx
            return "ok"

        deps = HudDeps(
            read_stdin=lambda: {},
            parse_stdin_limits=lambda _: None,
            fetch_usage_fallback=lambda: None,
            scan_processes=lambda: [],
            build_context=_stub_context,
            render=capture_render,
            now=_fixed_now,
            get_terminal_cols=lambda: 120,
            emit=lambda _: None,
        )
        build_hud_output({"context_window": {"current_usage": {"input": 100}}}, deps)
        assert captured_ctx["ctx"].total_tokens == 100

    def test_parse_stdin_limits_exception(self):
        """Lines 252-253: exception in parse_stdin_limits is logged and ignored."""
        captured_ctx: dict = {}

        def capture_render(ctx, cfg):
            captured_ctx["ctx"] = ctx
            return "ok"

        def boom(_data: dict):
            raise RuntimeError("parse error")

        deps = HudDeps(
            read_stdin=lambda: {},
            parse_stdin_limits=boom,
            fetch_usage_fallback=lambda: None,
            scan_processes=lambda: [],
            build_context=_stub_context,
            render=capture_render,
            now=_fixed_now,
            get_terminal_cols=lambda: 120,
            emit=lambda _: None,
        )
        build_hud_output({}, deps)
        assert "ctx" in captured_ctx

    def test_running_agent_from_snapshot(self):
        """Lines 277-278: running agents from transcript snapshot."""
        captured_ctx: dict = {}

        def capture_render(ctx, cfg):
            captured_ctx["ctx"] = ctx
            return "ok"

        snap = TranscriptSnapshot(
            agents=[
                AgentCall(
                    id="a1",
                    agent_type="general-purpose",
                    model="kimi-k2",
                    description="task",
                    status="running",
                    start_time=_fixed_now(),
                    provider="kimi",
                )
            ]
        )

        deps = HudDeps(
            read_stdin=lambda: {"transcript_path": "/fake/t.jsonl"},
            parse_stdin_limits=lambda _: None,
            fetch_usage_fallback=lambda: None,
            scan_processes=lambda: [],
            parse_transcript=lambda _: snap,
            build_context=_stub_context,
            render=capture_render,
            now=_fixed_now,
            get_terminal_cols=lambda: 120,
            emit=lambda _: None,
        )
        build_hud_output({"transcript_path": "/fake/t.jsonl"}, deps)
        assert len(captured_ctx["ctx"].running_agents) == 1
        assert captured_ctx["ctx"].running_agents[0]["provider"] == "kimi"

    def test_todos_from_snapshot(self):
        """Lines 314-315: todo counts from transcript snapshot."""
        captured_ctx: dict = {}

        def capture_render(ctx, cfg):
            captured_ctx["ctx"] = ctx
            return "ok"

        snap = TranscriptSnapshot(
            todos=[
                TodoItem(content="todo1", status="completed"),
                TodoItem(content="todo2", status="pending"),
            ]
        )

        deps = HudDeps(
            read_stdin=lambda: {"transcript_path": "/fake/t.jsonl"},
            parse_stdin_limits=lambda _: None,
            fetch_usage_fallback=lambda: None,
            scan_processes=lambda: [],
            parse_transcript=lambda _: snap,
            build_context=_stub_context,
            render=capture_render,
            now=_fixed_now,
            get_terminal_cols=lambda: 120,
            emit=lambda _: None,
        )
        build_hud_output({"transcript_path": "/fake/t.jsonl"}, deps)
        assert captured_ctx["ctx"].todos_total == 2
        assert captured_ctx["ctx"].todos_done == 1

    def test_total_tokens_from_snapshot_session_tokens(self):
        """Line 330: total_tokens from snapshot session tokens when ctx.total_tokens == 0."""
        captured_ctx: dict = {}

        def capture_render(ctx, cfg):
            captured_ctx["ctx"] = ctx
            return "ok"

        snap = TranscriptSnapshot(session_tokens_in=50)

        deps = HudDeps(
            read_stdin=lambda: {"transcript_path": "/fake/t.jsonl"},
            parse_stdin_limits=lambda _: None,
            fetch_usage_fallback=lambda: None,
            scan_processes=lambda: [],
            parse_transcript=lambda _: snap,
            build_context=_stub_context,
            render=capture_render,
            now=_fixed_now,
            get_terminal_cols=lambda: 120,
            emit=lambda _: None,
        )
        build_hud_output({"transcript_path": "/fake/t.jsonl"}, deps)
        assert captured_ctx["ctx"].total_tokens == 50

    def test_process_scan_appends_agents(self):
        """Line 340: process scan appends agents when no running transcript agents."""
        captured_ctx: dict = {}

        def capture_render(ctx, cfg):
            captured_ctx["ctx"] = ctx
            return "ok"

        deps = HudDeps(
            read_stdin=lambda: {},
            parse_stdin_limits=lambda _: None,
            fetch_usage_fallback=lambda: None,
            scan_processes=lambda: [
                {"provider": "local", "model": "default", "description": "running task"}
            ],
            build_context=_stub_context,
            render=capture_render,
            now=_fixed_now,
            get_terminal_cols=lambda: 120,
            emit=lambda _: None,
        )
        build_hud_output({}, deps)
        assert len(captured_ctx["ctx"].running_agents) == 1
        assert captured_ctx["ctx"].running_agents[0]["description"] == "running task"

    def test_main_guard(self):
        """Line 356: if __name__ == '__main__': main()."""
        import inspect
        import mullet.hud.cli as cli_mod
        source_lines = inspect.getsource(cli_mod).splitlines()
        # Pad with empty lines so compile() preserves original line numbers.
        guard = "\n".join([""] * (len(source_lines) - 2) + source_lines[-2:])
        with patch.object(cli_mod, "__name__", "__main__"):
            with patch.object(cli_mod, "main") as mock_main:
                code = compile(guard, cli_mod.__file__, "exec")
                exec(code, cli_mod.__dict__)
                mock_main.assert_called_once()
