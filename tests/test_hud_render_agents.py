
from mullet.hud.render.lines.agents import _agent_chip, render_agents_line
from mullet.hud.types import HudRenderContext
from mullet.hud.colors import strip_ansi


class TestAgentChip:
    def test_running_status(self):
        a = {"status": "running", "provider": "kimi", "model": "k2", "description": "task"}
        out = strip_ansi(_agent_chip(a))
        assert "kimi/k2" in out
        assert "task" in out

    def test_failed_status(self):
        a = {"status": "failed", "provider": "openai", "model": "gpt-4"}
        out = strip_ansi(_agent_chip(a))
        assert "openai/gpt-4" in out

    def test_timed_out_status(self):
        a = {"status": "timed_out", "provider": "anthropic", "model": "sonnet"}
        out = strip_ansi(_agent_chip(a))
        assert "anthropic/sonnet" in out

    def test_unknown_status_defaults_running(self):
        a = {"status": "unknown", "provider": "local", "model": "llama"}
        out = strip_ansi(_agent_chip(a))
        assert "local/llama" in out

    def test_missing_provider_model(self):
        a = {}
        out = strip_ansi(_agent_chip(a))
        assert "?/?" in out

    def test_description_truncated(self):
        long_desc = "x" * 200
        a = {"provider": "kimi", "model": "k2", "description": long_desc}
        out = _agent_chip(a)
        assert "x" * 160 in strip_ansi(out)
        assert "…" in strip_ansi(out)
        assert long_desc not in strip_ansi(out)

    def test_fallback_to_task(self):
        a = {"provider": "kimi", "model": "k2", "task": "fallback"}
        out = strip_ansi(_agent_chip(a))
        assert "fallback" in out

    def test_no_description_no_task(self):
        a = {"provider": "kimi", "model": "k2"}
        out = strip_ansi(_agent_chip(a))
        assert "kimi/k2" in out


class TestRenderAgentsLine:
    def test_no_agents_zero_count(self):
        ctx = HudRenderContext(running_agents=[], agent_count=0)
        assert render_agents_line(ctx) is None

    def test_no_agents_positive_count(self):
        ctx = HudRenderContext(running_agents=[], agent_count=3)
        out = strip_ansi(render_agents_line(ctx))
        assert "Agents: 3" in out

    def test_four_or_fewer_details(self):
        agents = [
            {"provider": "kimi", "model": "k2", "description": "a"},
            {"provider": "kimi", "model": "k2", "description": "b"},
            {"provider": "gemini", "model": "flash", "description": "c"},
            {"provider": "anthropic", "model": "sonnet", "description": "d"},
        ]
        ctx = HudRenderContext(running_agents=agents, agent_count=4)
        out = render_agents_line(ctx)
        lines = out.split("\n")
        assert len(lines) == 4

    def test_more_than_four_summary(self):
        agents = [
            {"provider": "kimi", "model": "k2"},
            {"provider": "kimi", "model": "k2"},
            {"provider": "gemini", "model": "flash"},
            {"provider": "gemini", "model": "flash"},
            {"provider": "anthropic", "model": "sonnet"},
        ]
        ctx = HudRenderContext(running_agents=agents, agent_count=5)
        out = strip_ansi(render_agents_line(ctx))
        assert "Agents: 5" in out
        assert "kimi×2" in out
        assert "gemini×2" in out

    def test_single_agent_detail(self):
        agents = [{"provider": "kimi", "model": "k2", "description": "review"}]
        ctx = HudRenderContext(running_agents=agents, agent_count=1)
        out = strip_ansi(render_agents_line(ctx))
        assert "kimi/k2" in out
        assert "review" in out

    def test_missing_running_agents_attr(self):
        ctx = HudRenderContext()
        delattr(ctx, "running_agents")
        assert render_agents_line(ctx) is None
