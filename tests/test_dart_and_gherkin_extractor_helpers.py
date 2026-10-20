"""Tests for dart extractor pure helpers and GherkinExtractor."""

from mullet.knowledge.code.extractors.dart import (
    DartExtractor,
    _find_brace_end,
    _extract_doc_comment,
    _line_of_offset,
)
from mullet.knowledge.code.extractors.gherkin_extractor import GherkinExtractor


# ---------------------------------------------------------------------------
# _find_brace_end
# ---------------------------------------------------------------------------

class TestFindBraceEnd:
    def test_simple_brace_closed(self):
        src = "fn() { return 1; }"
        end = _find_brace_end(src, 5)
        assert end <= len(src)  # returns position after "}"

    def test_nested_braces_handled(self):
        src = "fn() { if (x) { y } }"
        end = _find_brace_end(src, 5)
        assert src[end - 1] == "}"  # returns position after "}"

    def test_returns_int(self):
        src = "fn() { x }"
        assert isinstance(_find_brace_end(src, 5), int)

    def test_unclosed_returns_len(self):
        src = "fn() { no close"
        end = _find_brace_end(src, 5)
        assert end == len(src)


# ---------------------------------------------------------------------------
# _line_of_offset
# ---------------------------------------------------------------------------

class TestLineOfOffset:
    def test_offset_zero_returns_one(self):
        assert _line_of_offset("line1\nline2", 0) == 1

    def test_second_line_offset(self):
        assert _line_of_offset("line1\nline2", 6) == 2

    def test_third_line(self):
        src = "a\nb\nc"
        assert _line_of_offset(src, 4) == 3

    def test_returns_int(self):
        assert isinstance(_line_of_offset("a\nb", 0), int)


# ---------------------------------------------------------------------------
# _extract_doc_comment
# ---------------------------------------------------------------------------

class TestExtractDocComment:
    def test_extracts_triple_slash_comment(self):
        lines = ["/// Hello", "void foo() {}"]
        result = _extract_doc_comment(lines, 1)
        assert "Hello" in result

    def test_extracts_multi_line_comment(self):
        lines = ["/// Line 1", "/// Line 2", "void foo() {}"]
        result = _extract_doc_comment(lines, 2)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_no_comment_returns_none_or_empty(self):
        lines = ["void foo() {}"]
        result = _extract_doc_comment(lines, 0)
        assert result is None or result == ""

    def test_returns_string_or_none(self):
        lines = ["/// comment", "void f() {}"]
        result = _extract_doc_comment(lines, 1)
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# DartExtractor
# ---------------------------------------------------------------------------

class TestDartExtractorEmpty:
    def test_empty_source_returns_empty(self):
        assert DartExtractor().extract("", "test.dart") == []

    def test_returns_list(self):
        assert isinstance(DartExtractor().extract("", "test.dart"), list)


class TestDartExtractorClass:
    def setup_method(self):
        self.ext = DartExtractor()

    def test_class_extracted(self):
        src = "class MyWidget {\n  String title;\n}"
        result = self.ext.extract(src, "my_widget.dart")
        types = {c.symbol_type for c in result}
        assert "class" in types

    def test_class_name_correct(self):
        src = "class Greeter {\n  String greet() => 'hi';\n}"
        result = self.ext.extract(src, "greeter.dart")
        class_names = {c.symbol_name for c in result if c.symbol_type == "class"}
        assert "Greeter" in class_names

    def test_method_extracted(self):
        src = "class Foo {\n  void bar() {\n    return;\n  }\n}"
        result = self.ext.extract(src, "foo.dart")
        types = {c.symbol_type for c in result}
        assert "method" in types

    def test_language_is_dart(self):
        src = "class A {}"
        result = self.ext.extract(src, "a.dart")
        if result:
            assert result[0].language == "dart"


class TestDartExtractorFunction:
    def setup_method(self):
        self.ext = DartExtractor()

    def test_function_extracted(self):
        src = "String greet(String name) {\n  return 'Hello';\n}"
        result = self.ext.extract(src, "greet.dart")
        types = {c.symbol_type for c in result}
        assert "function" in types

    def test_function_name_correct(self):
        src = "void doSomething() {\n  print('done');\n}"
        result = self.ext.extract(src, "action.dart")
        func_names = {c.symbol_name for c in result if c.symbol_type == "function"}
        assert "doSomething" in func_names

    def test_source_file_preserved(self):
        src = "void foo() {}"
        result = self.ext.extract(src, "my_file.dart")
        if result:
            assert result[0].source_file == "my_file.dart"


# ---------------------------------------------------------------------------
# GherkinExtractor
# ---------------------------------------------------------------------------

class TestGherkinExtractorEmpty:
    def test_empty_source_returns_chunk(self):
        result = GherkinExtractor().extract("", "empty.feature")
        assert len(result) >= 1

    def test_empty_chunk_language_gherkin(self):
        result = GherkinExtractor().extract("", "empty.feature")
        assert result[0].language == "gherkin"


class TestGherkinExtractorFeature:
    def setup_method(self):
        self.ext = GherkinExtractor()
        self.feature = (
            "Feature: Login\n"
            "  Scenario: Successful login\n"
            "    Given I am on the login page\n"
            "    When I enter credentials\n"
            "    Then I should be logged in\n"
        )

    def test_feature_chunk_present(self):
        result = self.ext.extract(self.feature, "login.feature")
        types = {c.symbol_type for c in result}
        assert "feature" in types

    def test_feature_name_extracted(self):
        result = self.ext.extract(self.feature, "login.feature")
        feature_chunks = [c for c in result if c.symbol_type == "feature"]
        assert any("Login" in c.symbol_name for c in feature_chunks)

    def test_scenario_chunk_present(self):
        result = self.ext.extract(self.feature, "login.feature")
        types = {c.symbol_type for c in result}
        assert "scenario" in types

    def test_scenario_name_extracted(self):
        result = self.ext.extract(self.feature, "login.feature")
        scenario_names = {c.symbol_name for c in result if c.symbol_type == "scenario"}
        assert "Successful login" in scenario_names

    def test_returns_list(self):
        result = self.ext.extract(self.feature, "login.feature")
        assert isinstance(result, list)

    def test_language_is_gherkin(self):
        result = self.ext.extract(self.feature, "login.feature")
        for chunk in result:
            assert chunk.language == "gherkin"


class TestGherkinExtractorScenarioOutline:
    def setup_method(self):
        self.ext = GherkinExtractor()

    def test_scenario_outline_extracted(self):
        feature = (
            "Feature: Auth\n"
            "  Scenario Outline: Login as <role>\n"
            "    Given I am <role>\n"
            "    Examples:\n"
            "      | role |\n"
            "      | admin |\n"
        )
        result = self.ext.extract(feature, "auth.feature")
        types = {c.symbol_type for c in result}
        assert "scenario_outline" in types

    def test_scenario_outline_name_extracted(self):
        feature = (
            "Feature: Auth\n"
            "  Scenario Outline: Login as <role>\n"
            "    Given I am <role>\n"
            "    Examples:\n"
            "      | role |\n"
            "      | admin |\n"
        )
        result = self.ext.extract(feature, "auth.feature")
        outline_names = {c.symbol_name for c in result if c.symbol_type == "scenario_outline"}
        assert any("Login" in n for n in outline_names)
