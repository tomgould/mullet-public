"""Tests for the CLI spinner module."""

import time
from io import StringIO
from mullet.cli.spinner import Spinner, SpinnerStyle, ToolSpinner, TOOL_EMOJI


# ── SpinnerStyle ─────────────────────────────────────────────────────────────

def test_spinner_style_has_frames():
    for style in SpinnerStyle:
        assert len(style.value) >= 2


def test_spinner_style_dots():
    assert "⠋" in SpinnerStyle.DOTS.value


def test_spinner_style_line():
    assert "-" in SpinnerStyle.LINE.value


# ── Spinner ──────────────────────────────────────────────────────────────────

def test_spinner_init():
    s = Spinner(SpinnerStyle.DOTS, "Loading...", stderr=StringIO())
    assert s.message == "Loading..."
    assert s.running is False


def test_spinner_start_stop():
    buf = StringIO()
    s = Spinner(SpinnerStyle.DOTS, "Working", stderr=buf)
    s.start()
    time.sleep(0.3)
    s.stop()
    assert s.running is False
    output = buf.getvalue()
    assert "Working" in output


def test_spinner_update():
    buf = StringIO()
    s = Spinner(SpinnerStyle.DOTS, "Phase 1", stderr=buf)
    s.start()
    time.sleep(0.15)
    s.update("Phase 2")
    time.sleep(0.15)
    s.stop()
    output = buf.getvalue()
    assert "Phase 2" in output


def test_spinner_context_manager():
    buf = StringIO()
    with Spinner(SpinnerStyle.LINE, "Loading", stderr=buf) as s:
        time.sleep(0.2)
    assert s.running is False
    assert "Loading" in buf.getvalue()


def test_spinner_no_colorize():
    buf = StringIO()
    s = Spinner(SpinnerStyle.DOTS, "Test", colorize=False, stderr=buf)
    s.start()
    time.sleep(0.15)
    s.stop()
    output = buf.getvalue()
    assert "\033[96m" not in output


def test_spinner_colorize():
    buf = StringIO()
    s = Spinner(SpinnerStyle.DOTS, "Test", colorize=True, stderr=buf)
    s.start()
    time.sleep(0.15)
    s.stop()
    output = buf.getvalue()
    assert "\033[96m" in output


def test_spinner_final_message():
    buf = StringIO()
    s = Spinner(SpinnerStyle.DOTS, "Loading", stderr=buf)
    s.start()
    time.sleep(0.15)
    s.stop(final_message="Done!")
    assert "Done!" in buf.getvalue()


# ── TOOL_EMOJI ───────────────────────────────────────────────────────────────

def test_tool_emoji_has_common_tools():
    assert "read_file" in TOOL_EMOJI
    assert "terminal" in TOOL_EMOJI
    assert "search" in TOOL_EMOJI
    assert "write_file" in TOOL_EMOJI


def test_tool_emoji_values_are_strings():
    for name, emoji in TOOL_EMOJI.items():
        assert isinstance(emoji, str)
        assert len(emoji) > 0


# ── ToolSpinner ──────────────────────────────────────────────────────────────

def test_tool_spinner_known_tool():
    buf = StringIO()
    ts = ToolSpinner("terminal", "Running...", stderr=buf)
    assert "⚡" in ts.message
    assert "terminal" in ts.message


def test_tool_spinner_unknown_tool():
    buf = StringIO()
    ts = ToolSpinner("custom_tool", "Working...", stderr=buf)
    assert "🛠️" in ts.message
    assert "custom_tool" in ts.message


def test_tool_spinner_update():
    buf = StringIO()
    ts = ToolSpinner("terminal", "Phase 1", stderr=buf)
    ts.start()
    time.sleep(0.15)
    ts.update("Phase 2")
    time.sleep(0.15)
    ts.stop()
    output = buf.getvalue()
    assert "Phase 2" in output


def test_tool_spinner_different_style():
    buf = StringIO()
    ts = ToolSpinner("write_file", "Saving", style=SpinnerStyle.LINE, stderr=buf)
    ts.start()
    time.sleep(0.2)
    ts.stop()
    assert "write_file" in buf.getvalue()
