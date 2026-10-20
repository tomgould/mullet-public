"""Tests for mullet.knowledge.code.extractors.dart pure helpers."""


from mullet.knowledge.code.extractors.dart import (
    _find_brace_end,
    _line_of_offset,
    _extract_doc_comment,
    DartExtractor,
)


# ---------------------------------------------------------------------------
# _find_brace_end
# ---------------------------------------------------------------------------

class TestFindBraceEnd:
    def test_simple_empty_braces(self):
        source = "{}"
        # start at position 0 where '{' is
        result = _find_brace_end(source, 0)
        assert result == 2  # position after '}'

    def test_nested_braces(self):
        source = "{{ }}"
        result = _find_brace_end(source, 0)
        assert result == 5  # matches outer }

    def test_content_inside_braces(self):
        source = "{ int x = 1; }"
        result = _find_brace_end(source, 0)
        assert result == len(source)

    def test_unmatched_brace_returns_len(self):
        source = "{ no closing brace"
        result = _find_brace_end(source, 0)
        assert result == len(source)

    def test_string_with_brace_not_counted(self):
        source = '{ "}" }'
        result = _find_brace_end(source, 0)
        # The } inside the string should not close the outer brace
        assert result == len(source)

    def test_deeply_nested(self):
        source = "{ { { } } }"
        result = _find_brace_end(source, 0)
        assert result == len(source)

    def test_starts_after_opening_brace(self):
        source = "class Foo { int x; }"
        brace_pos = source.index("{")
        result = _find_brace_end(source, brace_pos)
        assert result == len(source)

    def test_empty_string_returns_zero(self):
        result = _find_brace_end("", 0)
        assert result == 0

    def test_returns_int(self):
        result = _find_brace_end("{}", 0)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# _line_of_offset
# ---------------------------------------------------------------------------

class TestLineOfOffset:
    def test_offset_zero_is_line_one(self):
        source = "hello\nworld"
        assert _line_of_offset(source, 0) == 1

    def test_offset_at_second_line(self):
        source = "hello\nworld"
        # offset 6 is 'w' which is on line 2
        assert _line_of_offset(source, 6) == 2

    def test_offset_at_newline(self):
        source = "hello\nworld"
        # offset 5 is '\n', still counted as line 1 (before the '\n' count increment)
        assert _line_of_offset(source, 5) == 1

    def test_multiple_lines(self):
        source = "a\nb\nc\nd"
        # offset for 'd' is 6 (after 3 newlines)
        assert _line_of_offset(source, 6) == 4

    def test_single_line_end(self):
        source = "hello"
        assert _line_of_offset(source, 5) == 1

    def test_empty_string_returns_one(self):
        assert _line_of_offset("", 0) == 1

    def test_returns_int(self):
        assert isinstance(_line_of_offset("hello\nworld", 0), int)


# ---------------------------------------------------------------------------
# _extract_doc_comment
# ---------------------------------------------------------------------------

class TestExtractDocComment:
    def test_no_preceding_lines_returns_none(self):
        lines = ["class Foo {"]
        result = _extract_doc_comment(lines, 0)
        assert result is None

    def test_triple_slash_doc_extracted(self):
        lines = [
            "/// A great class",
            "class Foo {",
        ]
        result = _extract_doc_comment(lines, 1)
        assert result is not None
        assert "A great class" in result

    def test_no_doc_returns_none(self):
        lines = [
            "// regular comment",
            "class Foo {",
        ]
        result = _extract_doc_comment(lines, 1)
        assert result is None

    def test_multi_line_doc_extracted(self):
        lines = [
            "/// Line 1",
            "/// Line 2",
            "class Foo {",
        ]
        result = _extract_doc_comment(lines, 2)
        assert result is not None
        assert "Line 1" in result
        assert "Line 2" in result


# ---------------------------------------------------------------------------
# DartExtractor
# ---------------------------------------------------------------------------

class TestDartExtractor:
    def setup_method(self):
        self.extractor = DartExtractor()

    def test_language_is_dart(self):
        assert self.extractor.language == "dart"

    def test_empty_source_returns_empty(self):
        result = self.extractor.extract("", "test.dart")
        assert result == []

    def test_class_extracted(self):
        source = "class MyWidget {\n  void build() {}\n}\n"
        chunks = self.extractor.extract(source, "widget.dart")
        classes = [c for c in chunks if c.symbol_type == "class"]
        assert len(classes) == 1
        assert classes[0].symbol_name == "MyWidget"

    def test_enum_extracted(self):
        source = "enum Status {\n  active,\n  inactive,\n}\n"
        chunks = self.extractor.extract(source, "status.dart")
        enums = [c for c in chunks if c.symbol_type == "enum"]
        assert len(enums) >= 1
        assert any(c.symbol_name == "Status" for c in enums)

    def test_chunk_has_line_numbers(self):
        source = "class Foo {\n  int x = 1;\n}\n"
        chunks = self.extractor.extract(source, "foo.dart")
        for chunk in chunks:
            assert chunk.line_start >= 1
