from unittest.mock import patch


from mullet.hud.render.lines.env import render_env_line
from mullet.hud.types import HudRenderContext
from mullet.hud.config_counts import ConfigCounts
from mullet.hud.colors import strip_ansi


def test_no_cwd():
    ctx = HudRenderContext(cwd="")
    assert render_env_line(ctx) is None


def test_all_counts_zero():
    ctx = HudRenderContext(cwd="/tmp")
    with patch("mullet.hud.render.lines.env.get_config_counts", return_value=ConfigCounts()):
        assert render_env_line(ctx) is None


def test_mcp_count():
    ctx = HudRenderContext(cwd="/tmp")
    with patch("mullet.hud.render.lines.env.get_config_counts", return_value=ConfigCounts(mcp_count=3)):
        line = render_env_line(ctx)
    assert "MCP:3" in strip_ansi(line)


def test_hooks_count():
    ctx = HudRenderContext(cwd="/tmp")
    with patch("mullet.hud.render.lines.env.get_config_counts", return_value=ConfigCounts(hooks_count=2)):
        line = render_env_line(ctx)
    assert "Hooks:2" in strip_ansi(line)


def test_rules_count():
    ctx = HudRenderContext(cwd="/tmp")
    with patch("mullet.hud.render.lines.env.get_config_counts", return_value=ConfigCounts(rules_count=5)):
        line = render_env_line(ctx)
    assert "Rules:5" in strip_ansi(line)


def test_claude_md_count():
    ctx = HudRenderContext(cwd="/tmp")
    with patch("mullet.hud.render.lines.env.get_config_counts", return_value=ConfigCounts(claude_md_count=1)):
        line = render_env_line(ctx)
    assert "CLAUDE.md:1" in strip_ansi(line)


def test_output_style():
    ctx = HudRenderContext(cwd="/tmp")
    with patch("mullet.hud.render.lines.env.get_config_counts", return_value=ConfigCounts(output_style="concise")):
        line = render_env_line(ctx)
    assert "Style:concise" in strip_ansi(line)


def test_multiple_parts():
    ctx = HudRenderContext(cwd="/tmp")
    with patch("mullet.hud.render.lines.env.get_config_counts", return_value=ConfigCounts(mcp_count=1, hooks_count=1, output_style="full")):
        line = render_env_line(ctx)
    assert "MCP:1" in strip_ansi(line)
    assert "Hooks:1" in strip_ansi(line)
    assert "Style:full" in strip_ansi(line)
