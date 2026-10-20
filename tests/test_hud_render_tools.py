from unittest.mock import patch


from mullet.hud.render.lines.tools import render_tools_line
from mullet.hud.types import HudRenderContext


def test_no_tools():
    ctx = HudRenderContext(running_tools=[])
    assert render_tools_line(ctx) is None


def test_none_tools():
    ctx = HudRenderContext()
    assert render_tools_line(ctx) is None


def test_single_no_target():
    ctx = HudRenderContext(running_tools=[{"name": "Read"}])
    with patch("mullet.hud.render.lines.tools._model_badge", return_value=""):
        line = render_tools_line(ctx)
    assert "Read" in line
    assert "\n" not in line


def test_single_with_target():
    ctx = HudRenderContext(running_tools=[{"name": "Read", "target": "/tmp/foo.py"}])
    with patch("mullet.hud.render.lines.tools._model_badge", return_value=""):
        line = render_tools_line(ctx)
    assert "Read(/tmp/foo.py)" in line


def test_multiple_tools():
    ctx = HudRenderContext(running_tools=[{"name": "Read", "target": "a.py"}, {"name": "Write", "target": "b.py"}])
    with patch("mullet.hud.render.lines.tools._model_badge", return_value=""):
        line = render_tools_line(ctx)
    lines = line.split("\n")
    assert len(lines) == 2
    assert "a.py" in lines[0]
    assert "b.py" in lines[1]


def test_tool_with_badge():
    ctx = HudRenderContext(running_tools=[{"name": "Search", "provider": "kimi", "model": "k2"}])
    with patch("mullet.hud.render.lines.tools._model_badge", return_value=" kimi/k2"):
        line = render_tools_line(ctx)
    assert "kimi/k2" in line


def test_missing_name():
    ctx = HudRenderContext(running_tools=[{"target": "x"}])
    with patch("mullet.hud.render.lines.tools._model_badge", return_value=""):
        line = render_tools_line(ctx)
    assert "?(x)" in line
