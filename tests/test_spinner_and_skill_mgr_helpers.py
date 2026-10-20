"""Tests for spinner pure helpers and skill_mgr schema/handler."""

import io
import json

from mullet.cli.spinner import (
    SpinnerStyle,
    Spinner,
    ToolSpinner,
    TOOL_EMOJI,
)
from mullet.core.tools.skill_mgr import (
    skill_mgr_handler,
    SKILL_MGR_SCHEMA,
)


# ---------------------------------------------------------------------------
# SpinnerStyle enum
# ---------------------------------------------------------------------------

class TestSpinnerStyle:
    def test_dots_has_frames(self):
        assert len(SpinnerStyle.DOTS.value) > 0

    def test_line_has_frames(self):
        assert len(SpinnerStyle.LINE.value) > 0

    def test_line_has_four_frames(self):
        assert len(SpinnerStyle.LINE.value) == 4

    def test_all_frames_are_strings(self):
        for style in SpinnerStyle:
            assert all(isinstance(f, str) for f in style.value)

    def test_expected_styles_exist(self):
        names = {s.name for s in SpinnerStyle}
        assert "DOTS" in names
        assert "LINE" in names
        assert "BRAILLE" in names


# ---------------------------------------------------------------------------
# Spinner init and state
# ---------------------------------------------------------------------------

def _make_spinner(message="Loading", colorize=True):
    return Spinner(SpinnerStyle.DOTS, message, colorize=colorize, stderr=io.StringIO())


class TestSpinnerInit:
    def test_message_stored(self):
        s = _make_spinner("Test msg")
        assert s.message == "Test msg"

    def test_running_false_initially(self):
        s = _make_spinner()
        assert s.running is False

    def test_colorize_true_sets_codes(self):
        s = _make_spinner(colorize=True)
        assert s.color_code != ""
        assert s.reset_code != ""

    def test_colorize_false_no_codes(self):
        s = _make_spinner(colorize=False)
        assert s.color_code == ""
        assert s.reset_code == ""

    def test_max_len_matches_message(self):
        s = _make_spinner("Hello")
        assert s._max_len == len("Hello")


class TestSpinnerUpdate:
    def test_update_changes_message(self):
        s = _make_spinner("Old")
        s.update("New message here")
        assert s.message == "New message here"

    def test_update_extends_max_len(self):
        s = _make_spinner("short")
        s.update("much longer message text")
        assert s._max_len >= len("much longer message text")

    def test_update_shorter_keeps_max(self):
        s = _make_spinner("long initial message")
        original_max = s._max_len
        s.update("x")
        assert s._max_len == original_max


# ---------------------------------------------------------------------------
# TOOL_EMOJI
# ---------------------------------------------------------------------------

class TestToolEmoji:
    def test_read_file_has_emoji(self):
        assert "read_file" in TOOL_EMOJI

    def test_search_has_emoji(self):
        assert "search" in TOOL_EMOJI

    def test_all_values_are_strings(self):
        assert all(isinstance(v, str) for v in TOOL_EMOJI.values())

    def test_dict_nonempty(self):
        assert len(TOOL_EMOJI) > 0


# ---------------------------------------------------------------------------
# ToolSpinner
# ---------------------------------------------------------------------------

class TestToolSpinner:
    def test_tool_name_in_message(self):
        ts = ToolSpinner("read_file", "reading...", colorize=False, stderr=io.StringIO())
        assert "read_file" in ts.message

    def test_known_tool_emoji_in_message(self):
        ts = ToolSpinner("read_file", "msg", colorize=False, stderr=io.StringIO())
        assert "📖" in ts.message

    def test_unknown_tool_uses_default_emoji(self):
        ts = ToolSpinner("unknown_tool", "msg", colorize=False, stderr=io.StringIO())
        assert "🛠️" in ts.message

    def test_update_prefixes_with_emoji(self):
        ts = ToolSpinner("search", "searching...", colorize=False, stderr=io.StringIO())
        ts.update("new query")
        assert "search" in ts.message
        assert "new query" in ts.message


# ---------------------------------------------------------------------------
# SKILL_MGR_SCHEMA
# ---------------------------------------------------------------------------

class TestSkillMgrSchema:
    def test_has_description(self):
        assert "description" in SKILL_MGR_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in SKILL_MGR_SCHEMA

    def test_action_required(self):
        assert "action" in SKILL_MGR_SCHEMA["parameters"]["required"]

    def test_action_enum_has_list(self):
        enum = SKILL_MGR_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "list" in enum

    def test_action_enum_has_create(self):
        enum = SKILL_MGR_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "create" in enum

    def test_action_enum_has_delete(self):
        enum = SKILL_MGR_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "delete" in enum

    def test_action_enum_has_edit(self):
        enum = SKILL_MGR_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "edit" in enum


# ---------------------------------------------------------------------------
# skill_mgr_handler
# ---------------------------------------------------------------------------

class TestSkillMgrHandler:
    def _run(self, **kwargs):
        return json.loads(skill_mgr_handler(**kwargs))

    def test_list_returns_skills_key(self):
        result = self._run(action="list")
        assert "skills" in result

    def test_list_success_true(self):
        result = self._run(action="list")
        assert result["success"] is True

    def test_create_missing_fields_fails(self):
        result = self._run(action="create")
        assert result["success"] is False

    def test_create_missing_body_fails(self):
        result = self._run(action="create", name="my_skill", description="d")
        assert result["success"] is False

    def test_delete_missing_name_fails(self):
        result = self._run(action="delete")
        assert result["success"] is False

    def test_edit_missing_name_fails(self):
        result = self._run(action="edit")
        assert result["success"] is False

    def test_invalid_action_returns_failure(self):
        result = self._run(action="invalid_action")
        assert result["success"] is False

    def test_returns_json_string(self):
        raw = skill_mgr_handler(action="list")
        assert isinstance(raw, str)
        json.loads(raw)  # must not raise
