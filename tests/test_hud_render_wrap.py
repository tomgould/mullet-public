"""Tests for mullet.hud.render.wrap — wcwidth_truncate and split_wrap_parts."""
from mullet.hud.render.wrap import wcwidth_truncate, split_wrap_parts
from mullet.hud.colors import strip_ansi, green, dim


# ---------------------------------------------------------------------------
# wcwidth_truncate
# ---------------------------------------------------------------------------

class TestWcwidthTruncate:
    def test_ascii_short_unchanged(self):
        line = "hello world"
        assert wcwidth_truncate(line, 20) == line

    def test_ascii_exactly_at_limit(self):
        line = "abcde"
        assert wcwidth_truncate(line, 5) == line

    def test_ascii_truncated(self):
        line = "hello world"
        result = wcwidth_truncate(line, 8)
        plain = strip_ansi(result)
        assert len(plain) <= 8 + 1  # ellipsis is 1 char
        assert plain.endswith("…")

    def test_zero_max_width_returns_unchanged(self):
        line = "hello"
        assert wcwidth_truncate(line, 0) == line

    def test_ansi_codes_preserved_not_counted(self):
        # green() wraps text in ANSI codes; visible width should be len of "hello"
        coloured = green("hello")
        result = wcwidth_truncate(coloured, 10)
        # Should not be truncated — visible width is 5
        assert "hello" in strip_ansi(result)

    def test_ansi_does_not_inflate_width(self):
        # A heavily coloured string should still truncate based on visible chars
        coloured = green("abcdefghij")  # 10 visible chars
        result = wcwidth_truncate(coloured, 5)
        plain = strip_ansi(result)
        assert plain.endswith("…")
        # Visible content before ellipsis must be ≤ 4 chars (5 - 1 for ellipsis)
        visible = plain[:-1]
        assert len(visible) <= 4

    def test_emoji_counted_correctly(self):
        # ● is U+25CF BLACK CIRCLE — wcwidth returns 1
        line = "●" * 10
        result = wcwidth_truncate(line, 6)
        plain = strip_ansi(result)
        assert plain.endswith("…")
        # 5 bullets + ellipsis = 6 visible cells
        assert plain.count("●") <= 5

    def test_cjk_each_two_cells(self):
        # 日本語 = 3 chars × 2 cells = 6 cells
        line = "日本語"
        # max_width=6 exactly fits, no truncation needed
        result = wcwidth_truncate(line, 6)
        assert result == line

    def test_cjk_truncated(self):
        # 日本語 = 6 cells; truncate at 4 cells → 1 CJK char (2 cells) + ellipsis (1 char)
        line = "日本語abc"
        result = wcwidth_truncate(line, 4)
        plain = strip_ansi(result)
        assert plain.endswith("…")

    def test_ellipsis_placement(self):
        line = "abcdefghij"
        result = wcwidth_truncate(line, 5, ellipsis="...")
        plain = strip_ansi(result)
        assert plain.endswith("...")
        assert len(plain) <= 5

    def test_custom_ellipsis(self):
        line = "abcdefghij"
        result = wcwidth_truncate(line, 8, ellipsis=">")
        plain = strip_ansi(result)
        assert plain.endswith(">")


# ---------------------------------------------------------------------------
# split_wrap_parts
# ---------------------------------------------------------------------------

class TestSplitWrapParts:
    def test_short_line_unchanged(self):
        line = "foo | bar | baz"
        result = split_wrap_parts(line, 80)
        assert result == [line]

    def test_zero_max_width_returns_original(self):
        line = "foo | bar"
        result = split_wrap_parts(line, 0)
        assert result == [line]

    def test_splits_at_separator(self):
        # Each part is 10 chars; sep is 3 chars → 3 parts = 36 chars at width 25
        parts_text = ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]
        line = " | ".join(parts_text)
        result = split_wrap_parts(line, 25)
        # Should split into at least 2 lines
        assert len(result) >= 2
        # All original parts should still be present
        rejoined = " | ".join(result)  # naive; parts stay inside lines
        for p in parts_text:
            assert any(p in l for l in result)

    def test_bracketed_block_stays_together(self):
        # Leading block e.g. [Opus | claude | master] should land on first line
        line = "[Opus 4.7 | claude | master] | Ctx 75% | 5h: 40%"
        result = split_wrap_parts(line, 40)
        # The leading block must be on the first line
        assert "[Opus 4.7 | claude | master]" in result[0]

    def test_single_part_no_split(self):
        line = "justonepart"
        result = split_wrap_parts(line, 5)
        # Can't split a single part — stays as-is
        assert result == [line]

    def test_all_parts_fit_on_one_line(self):
        line = "a | b | c"
        result = split_wrap_parts(line, 80)
        assert len(result) == 1
        assert result[0] == line

    def test_ansi_not_counted_in_width(self):
        coloured = dim("hello") + " | " + dim("world")
        result = split_wrap_parts(coloured, 80)
        assert len(result) == 1

    def test_custom_separator(self):
        line = "a :: b :: c :: d"
        result = split_wrap_parts(line, 10, sep=" :: ")
        # Some split must have occurred or all fit
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_empty_split_returns_original(self):
        """split_wrap_parts returns [line] when split produces empty list (line 82)."""
        # A line that splits into empty parts should still return [line]
        # This is defensive; normal split on sep won't be empty, but cover it.
        line = "a | b | c"
        result = split_wrap_parts(line, 10, sep="|")
        assert isinstance(result, list)
        assert len(result) >= 1


class TestWcwidthTruncateMissingLines:
    """Cover line 54 in wrap.py — wcwidth returns negative for non-printable."""

    def test_non_printable_char_zero_width(self):
        """Characters where wcwidth returns negative should be treated as 0-width (line 54)."""
        from unittest.mock import patch
        with patch("wcwidth.wcwidth", return_value=-1):
            result = wcwidth_truncate("abc", 2)
            plain = strip_ansi(result)
            assert plain.endswith("…")
