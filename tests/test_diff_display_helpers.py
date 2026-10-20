"""Tests for mullet.cli.diff_display — pure diff formatting helpers."""

from mullet.cli.diff_display import (
    ToolDiff,
    format_unified_diff,
    format_tool_diff,
    DiffRenderer,
)


# ---------------------------------------------------------------------------
# ToolDiff dataclass
# ---------------------------------------------------------------------------

class TestToolDiff:
    def test_fields(self):
        d = ToolDiff(tool_name="edit", file_path="auth.py", old_content="old", new_content="new")
        assert d.tool_name == "edit"
        assert d.file_path == "auth.py"
        assert d.old_content == "old"
        assert d.new_content == "new"

    def test_none_contents(self):
        d = ToolDiff(tool_name="create", file_path="new.py", old_content=None, new_content="content")
        assert d.old_content is None


# ---------------------------------------------------------------------------
# format_unified_diff
# ---------------------------------------------------------------------------

class TestFormatUnifiedDiff:
    def test_identical_content_returns_empty(self):
        result = format_unified_diff("same\n", "same\n", "file.py", colorize=False)
        assert result == ""

    def test_added_line_present(self):
        result = format_unified_diff("line1\n", "line1\nline2\n", "file.py", colorize=False)
        assert "line2" in result
        assert "+" in result

    def test_removed_line_present(self):
        result = format_unified_diff("line1\nline2\n", "line1\n", "file.py", colorize=False)
        assert "line2" in result
        assert "-" in result

    def test_filename_in_header(self):
        result = format_unified_diff("old\n", "new\n", "myfile.py", colorize=False)
        assert "myfile.py" in result

    def test_truncation_at_max_lines(self):
        old_content = "\n".join(f"line{i}" for i in range(100))
        new_content = "\n".join(f"changed{i}" for i in range(100))
        result = format_unified_diff(old_content, new_content, "big.py",
                                     colorize=False, max_lines=10)
        assert "truncated" in result.lower()

    def test_no_truncation_when_short(self):
        result = format_unified_diff("a\n", "b\n", "f.py", colorize=False, max_lines=100)
        assert "truncated" not in result

    def test_old_content_none_treated_as_empty(self):
        result = format_unified_diff(None, "new line\n", "file.py", colorize=False)
        assert "new line" in result

    def test_new_content_none_treated_as_empty(self):
        result = format_unified_diff("old line\n", None, "file.py", colorize=False)
        assert "old line" in result

    def test_colorize_false_no_ansi(self):
        result = format_unified_diff("a\n", "b\n", "f.py", colorize=False)
        assert "\033[" not in result

    def test_colorize_true_has_ansi(self):
        result = format_unified_diff("a\n", "b\n", "f.py", colorize=True)
        assert "\033[" in result

    def test_multiline_diff(self):
        old = "line1\nline2\nline3\n"
        new = "line1\nchanged\nline3\n"
        result = format_unified_diff(old, new, "f.py", colorize=False)
        assert "changed" in result
        assert "line2" in result


# ---------------------------------------------------------------------------
# format_tool_diff
# ---------------------------------------------------------------------------

class TestFormatToolDiff:
    def test_new_file_badge(self):
        result = format_tool_diff("create", "new.py", None, "content\n", colorize=False)
        assert "[NEW FILE]" in result

    def test_deleted_file_badge(self):
        result = format_tool_diff("delete", "old.py", "content\n", None, colorize=False)
        assert "[DELETED]" in result

    def test_modified_file_badge(self):
        result = format_tool_diff("edit", "auth.py", "old\n", "new\n", colorize=False)
        assert "[MODIFIED]" in result

    def test_no_change_returns_empty(self):
        result = format_tool_diff("edit", "unchanged.py", "same\n", "same\n", colorize=False)
        assert result == ""

    def test_tool_name_in_header(self):
        result = format_tool_diff("my_tool", "file.py", None, "text\n", colorize=False)
        assert "my_tool" in result

    def test_file_path_in_header(self):
        result = format_tool_diff("edit", "src/auth.py", "a\n", "b\n", colorize=False)
        assert "src/auth.py" in result

    def test_diff_content_present(self):
        result = format_tool_diff("edit", "f.py", "old\n", "new\n", colorize=False)
        assert "new" in result

    def test_colorize_false_no_ansi(self):
        result = format_tool_diff("edit", "f.py", "a\n", "b\n", colorize=False)
        assert "\033[" not in result


# ---------------------------------------------------------------------------
# DiffRenderer
# ---------------------------------------------------------------------------

class TestDiffRenderer:
    def test_empty_list_returns_no_changes_message(self):
        renderer = DiffRenderer(colorize=False)
        result = renderer.render([])
        assert "No file changes detected" in result

    def test_all_unchanged_returns_no_changes_message(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("edit", "f.py", "same\n", "same\n")]
        result = renderer.render(changes)
        assert "No file changes detected" in result

    def test_renders_multiple_changes(self):
        renderer = DiffRenderer(colorize=False)
        changes = [
            ToolDiff("create", "new.py", None, "content\n"),
            ToolDiff("edit", "old.py", "a\n", "b\n"),
        ]
        result = renderer.render(changes)
        assert "new.py" in result
        assert "old.py" in result

    def test_colorize_false_propagated(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("edit", "f.py", "a\n", "b\n")]
        result = renderer.render(changes)
        assert "\033[" not in result

    def test_colorize_true_propagated(self):
        renderer = DiffRenderer(colorize=True)
        changes = [ToolDiff("edit", "f.py", "a\n", "b\n")]
        result = renderer.render(changes)
        assert "\033[" in result

    def test_max_lines_respected(self):
        renderer = DiffRenderer(colorize=False, max_lines=3)
        old = "\n".join(f"line{i}" for i in range(50))
        new = "\n".join(f"changed{i}" for i in range(50))
        result = renderer.render([ToolDiff("edit", "f.py", old, new)])
        assert "truncated" in result.lower()
