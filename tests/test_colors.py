"""Tests for HUD color utilities."""
from mullet.hud.colors import strip_ansi, bold, dim, red, green, yellow, bar, ctx_color, bright_blue, BRIGHT_BLUE, RESET


def test_strip_ansi():
    assert strip_ansi("\033[31mhello\033[0m") == "hello"
    assert strip_ansi("plain") == "plain"


def test_color_functions():
    assert "hello" in strip_ansi(bold("hello"))
    assert "hello" in strip_ansi(dim("hello"))
    assert "hello" in strip_ansi(red("hello"))
    assert "hello" in strip_ansi(green("hello"))
    assert "hello" in strip_ansi(yellow("hello"))


def test_bar():
    b = strip_ansi(bar(50))
    assert len(b) == 10  # Default width
    b100 = strip_ansi(bar(100))
    assert len(b100) == 10


def test_ctx_color():
    c = ctx_color(30)  # Low usage — should be green
    assert "\033[" in c  # Contains ANSI escape
    c2 = ctx_color(80)  # High usage — should be red
    assert "\033[" in c2


def test_bar_zero():
    b = strip_ansi(bar(0))
    assert len(b) == 10


def test_bright_blue_wraps_text():
    result = bright_blue("hello")
    assert result == f"{BRIGHT_BLUE}hello{RESET}"
    assert "hello" in strip_ansi(result)


def test_ctx_color_threshold_85():
    from mullet.hud.colors import RED
    assert ctx_color(85) == RED
    assert ctx_color(100) == RED


def test_ctx_color_threshold_70():
    from mullet.hud.colors import YELLOW
    assert ctx_color(70) == YELLOW
    assert ctx_color(84) == YELLOW


def test_ctx_color_below_70():
    from mullet.hud.colors import GREEN
    assert ctx_color(0) == GREEN
    assert ctx_color(69) == GREEN
