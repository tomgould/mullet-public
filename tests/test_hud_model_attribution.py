"""HUD per-task model attribution — C2+C3+C4 regression tests.

Locks the contract that every live HUD row (parent "Thinking" row, tool pills,
subagent rows) carries a `provider/model` badge. Data source: the enclosing
assistant message's `model` field from the Claude Code transcript JSONL.

Phases under test:
  C2 — transcript parser extracts provider/model into ToolCall / AgentCall
  C3 — build_hud_output propagates them into HudRenderContext dicts
  C4 — render_thinking / render_tools_line append the badge
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _assistant_entry(
    content: list,
    model: str | None = None,
    timestamp: str = "2026-04-24T10:00:00+00:00",
) -> dict:
    msg: dict = {"content": content}
    if model is not None:
        msg["model"] = model
    return {"type": "assistant", "timestamp": timestamp, "message": msg}


def _tool_use_block(id: str, name: str, input: dict | None = None) -> dict:
    return {"type": "tool_use", "id": id, "name": name, "input": input or {}}


def _write_jsonl(path: Path, lines: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# C2 — transcript parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected_provider",
    [
        ("claude-opus-4-7", "anthropic"),
        ("claude-sonnet-4-6", "anthropic"),
        ("claude-haiku-4-5-20251001", "anthropic"),
        ("kimi-k2.6", "kimi"),
        ("kimi-k2-thinking-turbo", "kimi"),
        ("moonshot-v1-128k", "kimi"),
        ("gemini-2.5-flash", "gemini"),
        ("gpt-4o", "openai"),
        ("o1-mini", "openai"),
        ("mistral-nemo-instruct-2407", "local"),
        ("qwen2.5-coder-7b", "local"),
        ("", None),
        (None, None),
    ],
)
def test_infer_provider_from_model(model, expected_provider):
    from mullet.hud.transcript import _infer_provider_from_model
    assert _infer_provider_from_model(model) == expected_provider


def test_tool_call_carries_provider_and_model_from_assistant_message(tmp_path):
    from mullet.hud.transcript import parse_transcript

    entry = _assistant_entry(
        content=[_tool_use_block("t1", "Read", {"file_path": "/foo.py"})],
        model="claude-opus-4-7",
    )
    p = tmp_path / "transcript.jsonl"
    _write_jsonl(p, [entry])

    snap = parse_transcript(str(p))
    assert len(snap.running_tools) == 1
    tool = snap.running_tools[0]
    assert tool.model == "claude-opus-4-7"
    assert tool.provider == "anthropic"


def test_agent_call_carries_provider_from_assistant_message(tmp_path):
    from mullet.hud.transcript import parse_transcript

    entry = _assistant_entry(
        content=[_tool_use_block("a1", "Task", {
            "subagent_type": "Explore",
            "description": "explore codebase",
        })],
        model="kimi-k2.6",
    )
    p = tmp_path / "transcript.jsonl"
    _write_jsonl(p, [entry])

    snap = parse_transcript(str(p))
    assert len(snap.agents) == 1
    agent = snap.agents[0]
    # Agent's own model (from Task input) takes priority if present;
    # otherwise falls back to the enclosing assistant message's model.
    assert agent.model == "kimi-k2.6"
    assert agent.provider == "kimi"


def test_tool_call_has_none_provider_when_no_assistant_model(tmp_path):
    """Transcript without a model field (older CC versions) → None, not crash."""
    from mullet.hud.transcript import parse_transcript

    entry = _assistant_entry(
        content=[_tool_use_block("t2", "Bash", {"command": "ls"})],
        model=None,
    )
    p = tmp_path / "transcript.jsonl"
    _write_jsonl(p, [entry])

    snap = parse_transcript(str(p))
    assert len(snap.running_tools) == 1
    tool = snap.running_tools[0]
    assert tool.model is None
    assert tool.provider is None


def test_mid_session_model_change_attaches_to_correct_tools(tmp_path):
    """Each tool_use inherits its own enclosing message's model —
    not the session-initial one. Simulates a constraint self-heal
    that swaps model mid-session."""
    from mullet.hud.transcript import parse_transcript

    first = _assistant_entry(
        content=[_tool_use_block("t1", "Read", {"file_path": "/a.py"})],
        model="claude-opus-4-7",
        timestamp="2026-04-24T10:00:00+00:00",
    )
    second = _assistant_entry(
        content=[_tool_use_block("t2", "Bash", {"command": "pytest"})],
        model="claude-sonnet-4-6",
        timestamp="2026-04-24T10:00:05+00:00",
    )
    p = tmp_path / "transcript.jsonl"
    _write_jsonl(p, [first, second])

    snap = parse_transcript(str(p))
    by_id = {t.id: t for t in snap.running_tools}
    assert by_id["t1"].model == "claude-opus-4-7"
    assert by_id["t2"].model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# C3 — cli.py → HudRenderContext propagation
# ---------------------------------------------------------------------------

def test_build_hud_output_sets_main_provider_from_stdin_model():
    from mullet.hud.cli import HudDeps, build_hud_output
    from mullet.hud.types import HudRenderContext

    captured: dict = {}

    def fake_render(ctx: HudRenderContext, cfg) -> str:
        captured["ctx"] = ctx
        return ""

    deps = HudDeps(
        read_stdin=lambda: {"model": {"id": "kimi-k2.6"}, "cwd": "/tmp"},
        parse_stdin_limits=lambda _d: None,
        fetch_usage_fallback=lambda: None,
        scan_processes=lambda: [],
        parse_transcript=lambda _p: None,
        build_context=lambda _cwd: HudRenderContext(),
        render=fake_render,
        emit=lambda _s: None,
    )
    build_hud_output({"model": {"id": "kimi-k2.6"}, "cwd": "/tmp"}, deps)
    assert captured["ctx"].main_provider == "kimi"
    assert captured["ctx"].model_name == "kimi-k2.6"


def test_build_hud_output_propagates_tool_provider_model_into_dicts(tmp_path):
    from mullet.hud.cli import HudDeps, build_hud_output
    from mullet.hud.transcript import parse_transcript
    from mullet.hud.types import HudRenderContext

    entry = _assistant_entry(
        content=[_tool_use_block("t1", "Grep", {"pattern": "foo"})],
        model="claude-opus-4-7",
    )
    tpath = tmp_path / "transcript.jsonl"
    _write_jsonl(tpath, [entry])

    captured: dict = {}

    def fake_render(ctx: HudRenderContext, cfg) -> str:
        captured["ctx"] = ctx
        return ""

    deps = HudDeps(
        read_stdin=lambda: {},
        parse_stdin_limits=lambda _d: None,
        fetch_usage_fallback=lambda: None,
        scan_processes=lambda: [],
        parse_transcript=parse_transcript,
        build_context=lambda _cwd: HudRenderContext(),
        render=fake_render,
        emit=lambda _s: None,
    )
    build_hud_output(
        {"model": {"id": "claude-opus-4-7"}, "cwd": "/tmp", "transcript_path": str(tpath)},
        deps,
    )
    tools = captured["ctx"].running_tools
    assert len(tools) == 1
    assert tools[0]["provider"] == "anthropic"
    assert tools[0]["model"] == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# C4 — renderer badges
# ---------------------------------------------------------------------------

def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# --- _model_badge helper (peer-review finding: needs direct coverage) -------

def test_model_badge_full_attribution():
    from mullet.hud.elements import _model_badge
    assert "anthropic/claude-opus-4-7" in _strip_ansi(_model_badge("anthropic", "claude-opus-4-7"))


def test_model_badge_only_provider_no_slash():
    from mullet.hud.elements import _model_badge
    out = _strip_ansi(_model_badge("anthropic", None))
    assert "anthropic" in out
    assert "/" not in out


def test_model_badge_only_model_no_slash():
    from mullet.hud.elements import _model_badge
    out = _strip_ansi(_model_badge(None, "claude-opus-4-7"))
    assert "claude-opus-4-7" in out
    assert "/" not in out


def test_model_badge_both_missing_returns_empty():
    from mullet.hud.elements import _model_badge
    assert _model_badge(None, None) == ""
    assert _model_badge("", "") == ""


def test_render_thinking_appends_provider_model_badge():
    from mullet.hud.elements import render_thinking
    from mullet.hud.types import HudRenderContext

    ctx = HudRenderContext(
        thinking_active=True,
        model_name="kimi-k2.6",
        main_provider="kimi",
    )
    out = _strip_ansi(render_thinking(ctx) or "")
    assert "Thinking" in out
    assert "kimi/kimi-k2.6" in out


def test_render_thinking_no_badge_when_model_missing():
    from mullet.hud.elements import render_thinking
    from mullet.hud.types import HudRenderContext

    ctx = HudRenderContext(thinking_active=True)  # no model_name, no main_provider
    out = _strip_ansi(render_thinking(ctx) or "")
    assert "Thinking" in out
    assert "/" not in out  # no empty "?/?" or "/" fragment


def test_render_thinking_returns_none_when_not_thinking():
    from mullet.hud.elements import render_thinking
    from mullet.hud.types import HudRenderContext

    assert render_thinking(HudRenderContext(thinking_active=False)) is None


def test_render_tools_line_appends_per_pill_badge():
    from mullet.hud.render.lines.tools import render_tools_line
    from mullet.hud.types import HudRenderContext

    ctx = HudRenderContext(running_tools=[
        {"name": "Read", "target": "foo.py", "provider": "anthropic", "model": "claude-opus-4-7"},
        {"name": "Bash", "target": "ls", "provider": "kimi", "model": "kimi-k2.6"},
    ])
    out = _strip_ansi(render_tools_line(ctx) or "")
    assert "Read(foo.py)" in out
    assert "anthropic/claude-opus-4-7" in out
    assert "Bash(ls)" in out
    assert "kimi/kimi-k2.6" in out


def test_render_tools_line_no_badge_when_model_missing():
    from mullet.hud.render.lines.tools import render_tools_line
    from mullet.hud.types import HudRenderContext

    ctx = HudRenderContext(running_tools=[
        {"name": "Read", "target": "foo.py"},  # no provider/model keys
    ])
    out = _strip_ansi(render_tools_line(ctx) or "")
    assert "Read(foo.py)" in out
    assert "/" not in out.split("Read(foo.py)")[1]  # nothing trailing the label
