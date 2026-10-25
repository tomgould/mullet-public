"""Tests for prompt_builder and diff_display pure helpers."""

import pytest

from mullet.core.delegation.prompt_builder import (
    Role,
    SevenSectionPrompt,
    build_seven_section_prompt,
)
from mullet.cli.diff_display import (
    ToolDiff,
    format_unified_diff,
    format_tool_diff,
    DiffRenderer,
)


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------

class TestRole:
    def test_architecture_value(self):
        assert Role.ARCHITECTURE.value == "architect"

    def test_code_reviewer_value(self):
        assert Role.CODE_REVIEWER.value == "code_reviewer"

    def test_security_analyst_value(self):
        assert Role.SECURITY_ANALYST.value == "security_analyst"

    def test_generic_value(self):
        assert Role.GENERIC.value == "generic"

    def test_plan_reviewer_value(self):
        assert Role.PLAN_REVIEWER.value == "plan_reviewer"

    def test_scope_analyst_value(self):
        assert Role.SCOPE_ANALYST.value == "scope_analyst"

    def test_role_is_str_enum(self):
        assert isinstance(Role.GENERIC, str)


# ---------------------------------------------------------------------------
# SevenSectionPrompt
# ---------------------------------------------------------------------------

class TestSevenSectionPrompt:
    def test_task_stored(self):
        p = SevenSectionPrompt(task="Fix the bug")
        assert p.task == "Fix the bug"

    def test_render_returns_string(self):
        p = SevenSectionPrompt(task="Do this")
        assert isinstance(p.render(), str)

    def test_render_contains_task(self):
        p = SevenSectionPrompt(task="My important task")
        result = p.render()
        assert "My important task" in result

    def test_render_contains_role(self):
        p = SevenSectionPrompt(task="t", role=Role.CODE_REVIEWER)
        result = p.render()
        assert "code_reviewer" in result

    def test_render_contains_expected_outcome(self):
        p = SevenSectionPrompt(task="t", expected_outcome="Tests pass")
        result = p.render()
        assert "Tests pass" in result

    def test_render_contains_context(self):
        p = SevenSectionPrompt(task="t", context="Look at file.py")
        result = p.render()
        assert "Look at file.py" in result

    def test_render_contains_must_do(self):
        p = SevenSectionPrompt(task="t", must_do=["Keep existing tests"])
        result = p.render()
        assert "Keep existing tests" in result

    def test_render_contains_must_not_do(self):
        p = SevenSectionPrompt(task="t", must_not_do=["Rename functions"])
        result = p.render()
        assert "Rename functions" in result

    def test_render_omits_empty_sections(self):
        p = SevenSectionPrompt(task="t")
        result = p.render()
        assert "3. CONTEXT" not in result
        assert "5. MUST DO" not in result
        assert "6. MUST NOT DO" not in result

    def test_default_role_is_generic(self):
        p = SevenSectionPrompt(task="t")
        assert p.role == Role.GENERIC


# ---------------------------------------------------------------------------
# build_seven_section_prompt
# ---------------------------------------------------------------------------

class TestBuildSevenSectionPrompt:
    def test_returns_string(self):
        result = build_seven_section_prompt("Do the task")
        assert isinstance(result, str)

    def test_empty_task_raises(self):
        with pytest.raises(ValueError):
            build_seven_section_prompt("")

    def test_whitespace_only_task_raises(self):
        with pytest.raises(ValueError):
            build_seven_section_prompt("   ")

    def test_task_in_output(self):
        result = build_seven_section_prompt("Fix the auth bug")
        assert "Fix the auth bug" in result

    def test_expected_outcome_in_output(self):
        result = build_seven_section_prompt("task", expected_outcome="All tests green")
        assert "All tests green" in result

    def test_context_in_output(self):
        result = build_seven_section_prompt("task", context="File: src/auth.py")
        assert "src/auth.py" in result

    def test_constraints_in_output(self):
        result = build_seven_section_prompt("task", constraints="Python 3.10+ only")
        assert "Python 3.10+ only" in result

    def test_must_do_list_in_output(self):
        result = build_seven_section_prompt("task", must_do=["Keep tests"])
        assert "Keep tests" in result

    def test_must_not_do_list_in_output(self):
        result = build_seven_section_prompt("task", must_not_do=["Break API"])
        assert "Break API" in result

    def test_output_format_in_output(self):
        result = build_seven_section_prompt("task", output_format="EDIT blocks only")
        assert "EDIT blocks only" in result

    def test_string_role_valid_converted(self):
        result = build_seven_section_prompt("task", role="code_reviewer")
        assert "code_reviewer" in result

    def test_string_role_invalid_raises(self):
        with pytest.raises(ValueError):
            build_seven_section_prompt("task", role="nonexistent_role")

    def test_role_enum_accepted(self):
        result = build_seven_section_prompt("task", role=Role.SECURITY_ANALYST)
        assert "security_analyst" in result

    def test_empty_must_do_skipped(self):
        result = build_seven_section_prompt("task", must_do=["", "  "])
        assert "5. MUST DO" not in result

    def test_minimal_prompt_structure(self):
        result = build_seven_section_prompt("Minimal task")
        assert "TASK:" in result
        assert "ROLE:" in result

    def test_all_sections_present_when_all_provided(self):
        result = build_seven_section_prompt(
            "task",
            expected_outcome="outcome",
            context="ctx",
            constraints="no globals",
            must_do=["do this"],
            must_not_do=["not this"],
            output_format="json",
        )
        assert "EXPECTED OUTCOME" in result
        assert "CONTEXT" in result
        assert "CONSTRAINTS" in result
        assert "MUST DO" in result
        assert "MUST NOT DO" in result
        assert "OUTPUT FORMAT" in result


# ---------------------------------------------------------------------------
# ToolDiff
# ---------------------------------------------------------------------------

class TestToolDiff:
    def test_fields_stored(self):
        d = ToolDiff(tool_name="Edit", file_path="src/foo.py", old_content="old", new_content="new")
        assert d.tool_name == "Edit"
        assert d.file_path == "src/foo.py"
        assert d.old_content == "old"
        assert d.new_content == "new"

    def test_old_content_none_allowed(self):
        d = ToolDiff(tool_name="Write", file_path="new.py", old_content=None, new_content="content")
        assert d.old_content is None

    def test_new_content_none_allowed(self):
        d = ToolDiff(tool_name="Delete", file_path="gone.py", old_content="was there", new_content=None)
        assert d.new_content is None


# ---------------------------------------------------------------------------
# format_unified_diff
# ---------------------------------------------------------------------------

class TestFormatUnifiedDiff:
    def test_identical_content_returns_empty(self):
        result = format_unified_diff("same\n", "same\n", "file.txt")
        assert result == ""

    def test_changed_content_returns_diff(self):
        result = format_unified_diff("old line\n", "new line\n", "file.txt")
        assert len(result) > 0

    def test_contains_filename(self):
        result = format_unified_diff("a\n", "b\n", "myfile.py")
        assert "myfile.py" in result

    def test_added_line_present(self):
        result = format_unified_diff("", "new line\n", "f.py", colorize=False)
        assert "new line" in result

    def test_removed_line_present(self):
        result = format_unified_diff("old line\n", "", "f.py", colorize=False)
        assert "old line" in result

    def test_colorize_false_no_ansi(self):
        result = format_unified_diff("a\n", "b\n", "f.py", colorize=False)
        assert "\033[" not in result

    def test_truncation_message_when_long(self):
        old = "\n".join(f"line {i}" for i in range(100))
        new = "\n".join(f"LINE {i}" for i in range(100))
        result = format_unified_diff(old, new, "large.py", max_lines=10)
        assert "truncated" in result

    def test_returns_string(self):
        result = format_unified_diff("a", "b", "f.py")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# format_tool_diff
# ---------------------------------------------------------------------------

class TestFormatToolDiff:
    def test_new_file_badge(self):
        result = format_tool_diff("Write", "new.py", None, "content\n", colorize=False)
        assert "NEW FILE" in result

    def test_deleted_file_badge(self):
        result = format_tool_diff("Delete", "gone.py", "content\n", None, colorize=False)
        assert "DELETED" in result

    def test_modified_file_badge(self):
        result = format_tool_diff("Edit", "file.py", "old\n", "new\n", colorize=False)
        assert "MODIFIED" in result

    def test_no_change_returns_empty(self):
        result = format_tool_diff("Edit", "file.py", "same\n", "same\n")
        assert result == ""

    def test_contains_tool_name(self):
        result = format_tool_diff("MyTool", "f.py", None, "x\n", colorize=False)
        assert "MyTool" in result

    def test_contains_file_path(self):
        result = format_tool_diff("Edit", "src/utils.py", "a\n", "b\n", colorize=False)
        assert "src/utils.py" in result

    def test_returns_string(self):
        result = format_tool_diff("Edit", "f.py", "a", "b")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# DiffRenderer
# ---------------------------------------------------------------------------

class TestDiffRenderer:
    def test_empty_list_returns_message(self):
        renderer = DiffRenderer(colorize=False)
        result = renderer.render([])
        assert "No file changes" in result

    def test_renders_changes(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("Edit", "f.py", "old\n", "new\n")]
        result = renderer.render(changes)
        assert len(result) > 0

    def test_renders_multiple_changes(self):
        renderer = DiffRenderer(colorize=False)
        changes = [
            ToolDiff("Write", "a.py", None, "new content\n"),
            ToolDiff("Edit", "b.py", "old\n", "new\n"),
        ]
        result = renderer.render(changes)
        assert "a.py" in result
        assert "b.py" in result

    def test_no_change_diffs_skipped(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("Edit", "f.py", "same\n", "same\n")]
        result = renderer.render(changes)
        assert "No file changes" in result

    def test_colorize_false_no_ansi(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("Edit", "f.py", "a\n", "b\n")]
        result = renderer.render(changes)
        assert "\033[" not in result
