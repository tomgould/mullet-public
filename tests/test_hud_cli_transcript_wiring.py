"""Tests for build_hud_output transcript wiring in mullet.hud.cli.

All tests use a stub deps.parse_transcript — no real filesystem parsing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


from mullet.hud.cli import HudDeps, build_hud_output
from mullet.hud.transcript import AgentCall, TodoItem, TranscriptSnapshot
from mullet.hud.types import HudRenderContext


def _fixed_now() -> datetime:
    return datetime(2026, 4, 24, 20, 0, 0, tzinfo=timezone.utc)


def _stub_context(cwd: str) -> HudRenderContext:
    return HudRenderContext(cwd=cwd)


def _make_deps(
    *,
    stdin: Optional[dict] = None,
    snapshot: Optional[TranscriptSnapshot] = None,
    scan_called: Optional[list] = None,
) -> HudDeps:
    """Return a minimal HudDeps with stubs for snapshot and render."""
    called = scan_called if scan_called is not None else []

    def _scan():
        called.append(True)
        return [{"provider": "ps", "model": "default", "description": "from_ps"}]

    def _parse_transcript(_path: str) -> TranscriptSnapshot:
        if snapshot is None:
            return TranscriptSnapshot()
        return snapshot

    captured_ctx: dict = {}

    def _render(ctx, cfg):
        captured_ctx["ctx"] = ctx
        return (
            f"AGENTS={len(ctx.running_agents)}"
            f" TODOS={ctx.todos_done}/{ctx.todos_total}"
            f" SIN={ctx.session_tokens_in}"
            f" SOUT={ctx.session_tokens_out}"
            f" TOK={ctx.total_tokens}"
            f" RTOOLS={len(ctx.running_tools)}"
        )

    deps = HudDeps(
        read_stdin=lambda: stdin or {},
        parse_stdin_limits=lambda _: None,
        fetch_usage_fallback=lambda: None,
        scan_processes=_scan,
        parse_transcript=_parse_transcript,
        build_context=_stub_context,
        render=_render,
        now=_fixed_now,
        get_terminal_cols=lambda: 120,
        emit=lambda _: None,
    )
    deps._captured = captured_ctx  # type: ignore[attr-defined]
    return deps


# ---------------------------------------------------------------------------
# Test 1 — running agents from snapshot wired into ctx.running_agents
# ---------------------------------------------------------------------------

def test_running_agents_from_snapshot_wired_into_ctx():
    snap = TranscriptSnapshot(
        agents=[
            AgentCall(
                id="a1",
                agent_type="Explore",
                model="claude-opus-4-7",
                description="explore repo",
                status="running",
                start_time=datetime(2026, 4, 24, 19, 0, tzinfo=timezone.utc),
            ),
            AgentCall(
                id="a2",
                agent_type="general-purpose",
                model=None,
                description=None,
                status="completed",
                start_time=datetime(2026, 4, 24, 18, 0, tzinfo=timezone.utc),
            ),
        ]
    )
    deps = _make_deps(stdin={"transcript_path": "/fake/session.jsonl"}, snapshot=snap)
    out = build_hud_output({"transcript_path": "/fake/session.jsonl"}, deps)
    # Only the running agent should be wired
    assert "AGENTS=1" in out
    ctx = deps._captured["ctx"]  # type: ignore[attr-defined]
    assert ctx.running_agents[0]["provider"] == "cc"
    assert ctx.running_agents[0]["model"] == "claude-opus-4-7"
    assert ctx.running_agents[0]["description"] == "explore repo"


# ---------------------------------------------------------------------------
# Test 2 — todos counted correctly from snapshot
# ---------------------------------------------------------------------------

def test_todos_counted_correctly_from_snapshot():
    snap = TranscriptSnapshot(
        todos=[
            TodoItem(content="A", status="completed"),
            TodoItem(content="B", status="completed"),
            TodoItem(content="C", status="pending"),
            TodoItem(content="D", status="in_progress"),
        ]
    )
    deps = _make_deps(stdin={"transcript_path": "/fake/session.jsonl"}, snapshot=snap)
    out = build_hud_output({"transcript_path": "/fake/session.jsonl"}, deps)
    assert "TODOS=2/4" in out


# ---------------------------------------------------------------------------
# Test 3 — session_tokens wired + total_tokens fallback when state was 0
# ---------------------------------------------------------------------------

def test_session_tokens_wired_and_total_tokens_fallback():
    snap = TranscriptSnapshot(
        session_tokens_in=500,
        session_tokens_out=200,
        session_cache_creation_tokens=30,
        session_cache_read_tokens=70,
    )
    deps = _make_deps(stdin={"transcript_path": "/fake/session.jsonl"}, snapshot=snap)
    out = build_hud_output({"transcript_path": "/fake/session.jsonl"}, deps)
    assert "SIN=500" in out
    assert "SOUT=200" in out
    # total_tokens = 500+200+30+70 = 800
    assert "TOK=800" in out


# ---------------------------------------------------------------------------
# Test 4 — when snapshot has running agents, scan_processes is NOT called
# ---------------------------------------------------------------------------

def test_scan_processes_not_called_when_snapshot_has_running_agents():
    snap = TranscriptSnapshot(
        agents=[
            AgentCall(
                id="a1",
                agent_type="Explore",
                model=None,
                description="do stuff",
                status="running",
                start_time=datetime(2026, 4, 24, 19, 0, tzinfo=timezone.utc),
            ),
        ]
    )
    scan_called: list = []
    deps = _make_deps(
        stdin={"transcript_path": "/fake/session.jsonl"},
        snapshot=snap,
        scan_called=scan_called,
    )
    build_hud_output({"transcript_path": "/fake/session.jsonl"}, deps)
    assert scan_called == [], "scan_processes should NOT be called when snapshot has running agents"


# ---------------------------------------------------------------------------
# Test 5 — when transcript_path missing, scan_processes IS called
# ---------------------------------------------------------------------------

def test_scan_processes_called_when_no_transcript_path():
    scan_called: list = []
    deps = _make_deps(
        stdin={},
        snapshot=None,
        scan_called=scan_called,
    )
    out = build_hud_output({}, deps)
    assert scan_called, "scan_processes SHOULD be called when no transcript_path"
    # The ps-stub returns one agent
    assert "AGENTS=1" in out
