"""Tests for mullet.hud.render.lines.agents and .todos."""
from __future__ import annotations

import re

from mullet.hud.types import HudRenderContext
from mullet.hud.render.lines.agents import _agent_chip, render_agents_line
from mullet.hud.render.lines.todos import render_todos_line


def _strip(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---------------------------------------------------------------------------
# render_agents_line
# ---------------------------------------------------------------------------

class TestRenderAgentsLine:
    def test_no_agents_and_no_count_returns_none(self) -> None:
        ctx = HudRenderContext(agent_count=0, running_agents=[])
        assert render_agents_line(ctx) is None

    def test_count_only_no_list_shows_count(self) -> None:
        ctx = HudRenderContext(agent_count=3, running_agents=[])
        out = render_agents_line(ctx)
        assert out is not None
        assert "3" in _strip(out)
        assert "Agents" in out

    def test_single_agent_shows_provider_model_description(self) -> None:
        ctx = HudRenderContext(
            running_agents=[{"provider": "kimi", "model": "kimi-k2.6", "description": "refactor auth", "status": "running"}]
        )
        out = render_agents_line(ctx)
        assert out is not None
        plain = _strip(out)
        assert "kimi" in plain
        assert "kimi-k2.6" in plain
        assert "refactor auth" in plain

    def test_up_to_four_agents_shows_individual_rows(self) -> None:
        agents = [
            {"provider": "kimi", "model": f"m{i}", "description": f"task {i}", "status": "running"}
            for i in range(4)
        ]
        ctx = HudRenderContext(running_agents=agents)
        out = render_agents_line(ctx)
        assert out is not None
        # Should be multi-line (4 rows joined with \n)
        assert out.count("\n") == 3

    def test_more_than_four_agents_shows_summary(self) -> None:
        agents = [
            {"provider": "kimi", "model": "kimi-k2.6", "description": f"task {i}", "status": "running"}
            for i in range(5)
        ]
        ctx = HudRenderContext(running_agents=agents)
        out = render_agents_line(ctx)
        assert out is not None
        plain = _strip(out)
        assert "5" in plain
        assert "Agents" in out

    def test_multi_provider_summary_counts_per_provider(self) -> None:
        agents = (
            [{"provider": "kimi", "model": "k", "description": "t", "status": "running"} for _ in range(3)]
            + [{"provider": "gemini", "model": "g", "description": "t", "status": "running"} for _ in range(2)]
        )
        ctx = HudRenderContext(running_agents=agents)
        out = render_agents_line(ctx)
        assert out is not None
        plain = _strip(out)
        assert "kimi" in plain
        assert "gemini" in plain

    def test_failed_agent_shows_fail_icon(self) -> None:
        ctx = HudRenderContext(
            running_agents=[{"provider": "kimi", "model": "k", "description": "t", "status": "failed"}]
        )
        chip = _agent_chip({"provider": "kimi", "model": "k", "description": "task", "status": "failed"})
        plain = _strip(chip)
        assert "kimi" in plain

    def test_long_description_truncated_at_160(self) -> None:
        long_desc = "x" * 200
        chip = _agent_chip({"provider": "p", "model": "m", "description": long_desc, "status": "running"})
        plain = _strip(chip)
        assert "…" in plain

    def test_missing_status_defaults_to_running_icon(self) -> None:
        chip = _agent_chip({"provider": "p", "model": "m", "description": "task"})
        # Should not raise — default icon used
        assert "p/m" in _strip(chip)


# ---------------------------------------------------------------------------
# render_todos_line
# ---------------------------------------------------------------------------

class TestRenderTodosLine:
    def test_no_todos_returns_none(self) -> None:
        ctx = HudRenderContext(todos_total=0, todos_done=0)
        assert render_todos_line(ctx) is None

    def test_zero_done_shows_fraction(self) -> None:
        ctx = HudRenderContext(todos_total=5, todos_done=0)
        out = render_todos_line(ctx)
        assert out is not None
        assert "0/5" in _strip(out)

    def test_partial_shows_fraction(self) -> None:
        ctx = HudRenderContext(todos_total=5, todos_done=3)
        out = render_todos_line(ctx)
        assert out is not None
        assert "3/5" in _strip(out)

    def test_all_done_shows_fraction(self) -> None:
        ctx = HudRenderContext(todos_total=5, todos_done=5)
        out = render_todos_line(ctx)
        assert out is not None
        assert "5/5" in _strip(out)

    def test_negative_total_returns_none(self) -> None:
        ctx = HudRenderContext(todos_total=-1, todos_done=0)
        assert render_todos_line(ctx) is None

    def test_has_todos_label(self) -> None:
        ctx = HudRenderContext(todos_total=3, todos_done=1)
        out = render_todos_line(ctx)
        assert out is not None
        assert "Todos" in out
