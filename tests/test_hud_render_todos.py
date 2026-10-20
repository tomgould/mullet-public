
from mullet.hud.render.lines.todos import render_todos_line
from mullet.hud.types import HudRenderContext


def test_no_todos():
    ctx = HudRenderContext(todos_done=0, todos_total=0)
    assert render_todos_line(ctx) is None


def test_zero_done():
    ctx = HudRenderContext(todos_done=0, todos_total=5)
    line = render_todos_line(ctx)
    assert line is not None
    assert "0/5" in line


def test_partial_done():
    ctx = HudRenderContext(todos_done=2, todos_total=5)
    line = render_todos_line(ctx)
    assert "2/5" in line


def test_all_done():
    ctx = HudRenderContext(todos_done=5, todos_total=5)
    line = render_todos_line(ctx)
    assert "5/5" in line


def test_negative_total():
    ctx = HudRenderContext(todos_done=0, todos_total=-1)
    assert render_todos_line(ctx) is None


def test_single_todo():
    ctx = HudRenderContext(todos_done=1, todos_total=1)
    line = render_todos_line(ctx)
    assert "1/1" in line
