"""Tests for the diff display module."""

from mullet.cli.diff_display import (
    format_unified_diff, format_tool_diff,
    DiffRenderer, ToolDiff,
    RED, GREEN, CYAN, BOLD, RESET,
)


# ── format_unified_diff ─────────────────────────────────────────────────────

def test_no_change():
    assert format_unified_diff("hello\n", "hello\n", "test.txt") == ""


def test_added_line():
    result = format_unified_diff("line1\n", "line1\nline2\n", "test.txt")
    assert f"{BOLD}--- a/test.txt{RESET}" in result
    assert f"{BOLD}+++ b/test.txt{RESET}" in result
    assert CYAN in result
    assert "+line2" in result


def test_removed_line():
    result = format_unified_diff("line1\nline2\n", "line1\n", "test.txt")
    assert "-line2" in result


def test_modified_line():
    result = format_unified_diff("old\n", "new\n", "test.txt")
    assert "-old" in result
    assert "+new" in result


def test_header_coloring():
    result = format_unified_diff("a\n", "b\n", "file.py")
    assert BOLD in result
    assert CYAN in result


def test_no_coloring():
    result = format_unified_diff("a\n", "b\n", "f.py", colorize=False)
    assert BOLD not in result
    assert RED not in result
    assert GREEN not in result


def test_truncation():
    old = "\n".join(f"line{i}" for i in range(100)) + "\n"
    new = "\n".join(f"changed{i}" for i in range(100)) + "\n"
    result = format_unified_diff(old, new, "big.txt", max_lines=10)
    assert "truncated" in result.lower()


def test_line_numbers():
    result = format_unified_diff("a\nb\n", "a\nc\n", "test.txt")
    assert any(c.isdigit() for c in result)


# ── format_tool_diff ────────────────────────────────────────────────────────

def test_new_file():
    result = format_tool_diff("write_file", "new.py", None, "content\n")
    assert "[NEW FILE]" in result
    assert "new.py" in result


def test_deleted_file():
    result = format_tool_diff("delete", "old.py", "content\n", None)
    assert "[DELETED]" in result


def test_modified_file():
    result = format_tool_diff("edit", "mod.py", "old\n", "new\n")
    assert "[MODIFIED]" in result


def test_no_change_tool_diff():
    assert format_tool_diff("edit", "same.py", "same\n", "same\n") == ""


def test_tool_diff_no_coloring():
    result = format_tool_diff("edit", "f.py", "a\n", "b\n", colorize=False)
    assert "[MODIFIED]" in result
    assert BOLD not in result


# ── DiffRenderer ─────────────────────────────────────────────────────────────

def test_renderer_single():
    changes = [ToolDiff("edit", "a.py", "old\n", "new\n")]
    result = DiffRenderer().render(changes)
    assert "[MODIFIED]" in result


def test_renderer_multiple():
    changes = [
        ToolDiff("write", "new.py", None, "hello\n"),
        ToolDiff("edit", "mod.py", "a\n", "b\n"),
    ]
    result = DiffRenderer().render(changes)
    assert "[NEW FILE]" in result
    assert "[MODIFIED]" in result


def test_renderer_empty():
    assert "No file changes" in DiffRenderer().render([])


def test_renderer_custom_max_lines():
    assert DiffRenderer(max_lines=5).max_lines == 5


def test_renderer_no_colorize():
    changes = [ToolDiff("edit", "a.py", "x\n", "y\n")]
    result = DiffRenderer(colorize=False).render(changes)
    assert BOLD not in result


def test_renderer_all_diffs_empty_returns_no_changes():
    """Line 154: non-empty changes list but all produce empty diffs → no changes message."""
    changes = [ToolDiff("edit", "a.py", "same\n", "same\n")]
    result = DiffRenderer().render(changes)
    assert "No file changes" in result
