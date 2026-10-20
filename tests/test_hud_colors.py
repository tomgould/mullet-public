"""Tests for mullet.hud.colors — ANSI color helpers."""

from mullet.hud.colors import (
    BLUE, BOLD, BRIGHT_BLUE, CYAN, DIM, GREEN, MAGENTA, RED, RESET, WHITE, YELLOW,
    bar, blue, bold, bright_blue, ctx_color, cyan, dim,
    green, magenta, red, strip_ansi, white, yellow,
)


def test_strip_ansi_removes_codes():
    assert strip_ansi("\033[1mhello\033[0m") == "hello"


def test_strip_ansi_no_codes():
    assert strip_ansi("plain text") == "plain text"


def test_bold():
    result = bold("x")
    assert BOLD in result and "x" in result


def test_dim():
    result = dim("x")
    assert DIM in result and "x" in result


def test_red():
    result = red("x")
    assert RED in result and "x" in result


def test_green():
    result = green("x")
    assert GREEN in result and "x" in result


def test_yellow():
    result = yellow("x")
    assert YELLOW in result and "x" in result


def test_blue():
    result = blue("hello")
    assert BLUE in result and "hello" in result


def test_magenta():
    result = magenta("x")
    assert MAGENTA in result and "x" in result


def test_cyan():
    result = cyan("x")
    assert CYAN in result and "x" in result


def test_white():
    result = white("hi")
    assert WHITE in result and "hi" in result


def test_bright_blue():
    result = bright_blue("x")
    assert BRIGHT_BLUE in result and "x" in result


def test_ctx_color_red():
    assert ctx_color(85) == RED


def test_ctx_color_yellow():
    assert ctx_color(70) == YELLOW


def test_ctx_color_green():
    assert ctx_color(50) == GREEN


def test_bar_renders():
    b = bar(50, width=10)
    assert len(strip_ansi(b)) == 10


def test_bar_full():
    b = bar(100, width=5)
    assert len(strip_ansi(b)) == 5


def test_bar_empty():
    b = bar(0, width=4)
    assert len(strip_ansi(b)) == 4
