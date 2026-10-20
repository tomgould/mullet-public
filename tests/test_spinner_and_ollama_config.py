"""Tests for spinner/ToolSpinner and OllamaConfig dataclass."""

import io
from mullet.cli.spinner import SpinnerStyle, Spinner, ToolSpinner, TOOL_EMOJI
from mullet.core.providers.ollama import OllamaConfig


# ---------------------------------------------------------------------------
# SpinnerStyle
# ---------------------------------------------------------------------------

class TestSpinnerStyle:
    def test_all_styles_have_frames(self):
        for style in SpinnerStyle:
            assert len(style.value) > 0

    def test_frames_are_strings(self):
        for style in SpinnerStyle:
            for frame in style.value:
                assert isinstance(frame, str)

    def test_dots_style_exists(self):
        assert SpinnerStyle.DOTS is not None

    def test_line_style_exists(self):
        assert SpinnerStyle.LINE is not None


# ---------------------------------------------------------------------------
# TOOL_EMOJI constant
# ---------------------------------------------------------------------------

class TestToolEmoji:
    def test_is_dict(self):
        assert isinstance(TOOL_EMOJI, dict)

    def test_common_tools_present(self):
        for tool in ("read_file", "write_file", "terminal", "search"):
            assert tool in TOOL_EMOJI

    def test_values_are_strings(self):
        for emoji in TOOL_EMOJI.values():
            assert isinstance(emoji, str)


# ---------------------------------------------------------------------------
# Spinner (state-only tests, no thread start)
# ---------------------------------------------------------------------------

class TestSpinnerInit:
    def test_message_stored(self):
        s = Spinner(SpinnerStyle.DOTS, "working...")
        assert s.message == "working..."

    def test_not_running_on_init(self):
        s = Spinner(SpinnerStyle.DOTS, "msg")
        assert s.running is False

    def test_colorize_false_no_color_codes(self):
        s = Spinner(SpinnerStyle.DOTS, "msg", colorize=False)
        assert s.color_code == ""
        assert s.reset_code == ""

    def test_colorize_true_has_color_codes(self):
        s = Spinner(SpinnerStyle.DOTS, "msg", colorize=True)
        assert "\033[" in s.color_code


class TestSpinnerUpdate:
    def test_update_changes_message(self):
        s = Spinner(SpinnerStyle.DOTS, "initial")
        s.update("updated")
        assert s.message == "updated"

    def test_update_tracks_max_len(self):
        s = Spinner(SpinnerStyle.DOTS, "short")
        s.update("a much longer message here")
        assert s._max_len >= len("a much longer message here")

    def test_shorter_update_preserves_max_len(self):
        s = Spinner(SpinnerStyle.DOTS, "long initial message here")
        prev_max = s._max_len
        s.update("short")
        assert s._max_len == prev_max


class TestSpinnerContextManager:
    def test_start_and_stop_via_context(self):
        buf = io.StringIO()
        with Spinner(SpinnerStyle.LINE, "test", colorize=False, stderr=buf):
            pass  # Should not raise
        # After exit, spinner clears line (no final message)
        output = buf.getvalue()
        assert "\r" in output  # Clear sequence written


# ---------------------------------------------------------------------------
# ToolSpinner
# ---------------------------------------------------------------------------

class TestToolSpinner:
    def test_known_tool_gets_emoji_prefix(self):
        ts = ToolSpinner("read_file", "loading auth.py")
        assert TOOL_EMOJI["read_file"] in ts.message
        assert "read_file" in ts.message
        assert "loading auth.py" in ts.message

    def test_unknown_tool_gets_default_emoji(self):
        ts = ToolSpinner("unknown_tool_xyz", "doing something")
        assert "unknown_tool_xyz" in ts.message
        assert "🛠️" in ts.message

    def test_update_includes_emoji(self):
        ts = ToolSpinner("search", "initial")
        ts.update("new query")
        assert TOOL_EMOJI["search"] in ts.message
        assert "new query" in ts.message

    def test_tool_name_stored(self):
        ts = ToolSpinner("terminal", "running ls")
        assert ts.tool_name == "terminal"


# ---------------------------------------------------------------------------
# OllamaConfig
# ---------------------------------------------------------------------------

class TestOllamaConfig:
    def test_defaults(self):
        cfg = OllamaConfig()
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.model == "llama2"
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.max_tokens == 4096

    def test_custom_url(self):
        cfg = OllamaConfig(base_url="http://192.168.1.1:11434")
        assert cfg.base_url == "http://192.168.1.1:11434"

    def test_custom_model(self):
        cfg = OllamaConfig(model="mistral")
        assert cfg.model == "mistral"

    def test_request_timeout_default(self):
        cfg = OllamaConfig()
        assert cfg.request_timeout == 300
