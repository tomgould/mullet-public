"""Tests for diff_display pure functions and prompt_builder pure helpers."""

import pytest

from mullet.cli.diff_display import (
    ToolDiff,
    format_unified_diff,
    format_tool_diff,
    DiffRenderer,
)
from mullet.core.delegation.prompt_builder import (
    Role,
    SevenSectionPrompt,
    build_seven_section_prompt,
)


# ---------------------------------------------------------------------------
# ToolDiff dataclass
# ---------------------------------------------------------------------------

class TestToolDiff:
    def test_fields_stored(self):
        td = ToolDiff(
            tool_name="Edit",
            file_path="src/main.py",
            old_content="old",
            new_content="new",
        )
        assert td.tool_name == "Edit"
        assert td.file_path == "src/main.py"
        assert td.old_content == "old"
        assert td.new_content == "new"

    def test_none_old_content(self):
        td = ToolDiff(tool_name="Write", file_path="new.py", old_content=None, new_content="new")
        assert td.old_content is None

    def test_none_new_content(self):
        td = ToolDiff(tool_name="Delete", file_path="old.py", old_content="old", new_content=None)
        assert td.new_content is None


# ---------------------------------------------------------------------------
# format_unified_diff
# ---------------------------------------------------------------------------

class TestFormatUnifiedDiff:
    def test_identical_content_returns_empty(self):
        result = format_unified_diff("same\n", "same\n", "file.py")
        assert result == ""

    def test_changed_content_returns_diff(self):
        result = format_unified_diff("old line\n", "new line\n", "file.py")
        assert len(result) > 0

    def test_filename_in_output(self):
        result = format_unified_diff("a\n", "b\n", "myfile.py")
        assert "myfile.py" in result

    def test_added_line_present(self):
        result = format_unified_diff("a\n", "a\nb\n", "f.py", colorize=False)
        assert "b" in result

    def test_removed_line_present(self):
        result = format_unified_diff("a\nb\n", "a\n", "f.py", colorize=False)
        assert "-" in result or "b" in result

    def test_long_diff_truncated(self):
        old = "\n".join(f"line {i}" for i in range(200))
        new = "\n".join(f"changed {i}" for i in range(200))
        result = format_unified_diff(old, new, "big.py", colorize=False, max_lines=10)
        assert "truncated" in result.lower()

    def test_no_colorize_mode(self):
        result = format_unified_diff("a\n", "b\n", "f.py", colorize=False)
        assert "\033[" not in result

    def test_empty_old_and_new_different_returns_diff(self):
        result = format_unified_diff("", "content\n", "f.py", colorize=False)
        assert len(result) > 0

    def test_returns_string(self):
        assert isinstance(format_unified_diff("a", "b", "f.py"), str)


# ---------------------------------------------------------------------------
# format_tool_diff
# ---------------------------------------------------------------------------

class TestFormatToolDiff:
    def test_new_file_shows_badge(self):
        result = format_tool_diff("Write", "new.py", None, "content\n", colorize=False)
        assert "NEW FILE" in result

    def test_deleted_file_shows_badge(self):
        result = format_tool_diff("Delete", "old.py", "content\n", None, colorize=False)
        assert "DELETED" in result

    def test_modified_file_shows_badge(self):
        result = format_tool_diff("Edit", "f.py", "old\n", "new\n", colorize=False)
        assert "MODIFIED" in result

    def test_identical_content_returns_empty(self):
        result = format_tool_diff("Edit", "f.py", "same\n", "same\n", colorize=False)
        assert result == ""

    def test_tool_name_in_header(self):
        result = format_tool_diff("MyTool", "f.py", None, "content\n", colorize=False)
        assert "MyTool" in result

    def test_file_path_in_header(self):
        result = format_tool_diff("Edit", "src/utils.py", "a\n", "b\n", colorize=False)
        assert "src/utils.py" in result

    def test_returns_string(self):
        assert isinstance(format_tool_diff("t", "f.py", None, "x"), str)


# ---------------------------------------------------------------------------
# DiffRenderer
# ---------------------------------------------------------------------------

class TestDiffRenderer:
    def test_empty_changes_returns_no_changes(self):
        renderer = DiffRenderer()
        result = renderer.render([])
        assert "No file changes" in result

    def test_renders_new_file(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("Write", "new.py", None, "content\n")]
        result = renderer.render(changes)
        assert "NEW FILE" in result

    def test_renders_modified_file(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("Edit", "f.py", "old\n", "new\n")]
        result = renderer.render(changes)
        assert "MODIFIED" in result

    def test_all_unchanged_returns_no_changes(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("Edit", "f.py", "same\n", "same\n")]
        result = renderer.render(changes)
        assert "No file changes" in result

    def test_returns_string(self):
        renderer = DiffRenderer()
        assert isinstance(renderer.render([]), str)


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------

class TestRole:
    def test_values_are_strings(self):
        for role in Role:
            assert isinstance(role.value, str)

    def test_generic_value(self):
        assert Role.GENERIC.value == "generic"

    def test_code_reviewer_value(self):
        assert Role.CODE_REVIEWER.value == "code_reviewer"

    def test_all_expected_roles_present(self):
        values = {r.value for r in Role}
        assert "architect" in values
        assert "generic" in values
        assert "code_reviewer" in values
        assert "security_analyst" in values


# ---------------------------------------------------------------------------
# SevenSectionPrompt.render
# ---------------------------------------------------------------------------

class TestSevenSectionPromptRender:
    def test_task_in_output(self):
        p = SevenSectionPrompt(task="Do the thing")
        result = p.render()
        assert "Do the thing" in result

    def test_role_in_output(self):
        p = SevenSectionPrompt(task="task", role=Role.CODE_REVIEWER)
        result = p.render()
        assert "code_reviewer" in result

    def test_expected_outcome_included(self):
        p = SevenSectionPrompt(task="t", expected_outcome="Tests pass")
        result = p.render()
        assert "Tests pass" in result

    def test_must_do_items_included(self):
        p = SevenSectionPrompt(task="t", must_do=["step A", "step B"])
        result = p.render()
        assert "step A" in result
        assert "step B" in result

    def test_must_not_do_items_included(self):
        p = SevenSectionPrompt(task="t", must_not_do=["no globals"])
        result = p.render()
        assert "no globals" in result

    def test_empty_optional_sections_omitted(self):
        p = SevenSectionPrompt(task="t")
        result = p.render()
        assert "EXPECTED OUTCOME" not in result
        assert "CONTEXT" not in result

    def test_returns_string(self):
        p = SevenSectionPrompt(task="t")
        assert isinstance(p.render(), str)


# ---------------------------------------------------------------------------
# build_seven_section_prompt
# ---------------------------------------------------------------------------

class TestBuildSevenSectionPrompt:
    def test_simple_task(self):
        result = build_seven_section_prompt("Fix the bug")
        assert "Fix the bug" in result

    def test_empty_task_raises(self):
        with pytest.raises(ValueError):
            build_seven_section_prompt("")

    def test_whitespace_only_task_raises(self):
        with pytest.raises(ValueError):
            build_seven_section_prompt("   ")

    def test_role_string_accepted(self):
        result = build_seven_section_prompt("task", role="code_reviewer")
        assert "code_reviewer" in result

    def test_invalid_role_string_raises(self):
        with pytest.raises(ValueError):
            build_seven_section_prompt("task", role="unknown_role")

    def test_must_do_list_included(self):
        result = build_seven_section_prompt("task", must_do=["Step 1", "Step 2"])
        assert "Step 1" in result
        assert "Step 2" in result

    def test_output_format_included(self):
        result = build_seven_section_prompt("task", output_format="JSON only")
        assert "JSON only" in result

    def test_context_included(self):
        result = build_seven_section_prompt("task", context="background info")
        assert "background info" in result

    def test_returns_string(self):
        assert isinstance(build_seven_section_prompt("task"), str)

    def test_role_enum_accepted(self):
        result = build_seven_section_prompt("task", role=Role.SECURITY_ANALYST)
        assert "security_analyst" in result
