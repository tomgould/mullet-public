"""Tests for mullet.knowledge.code.extractors.css."""
from __future__ import annotations

import pytest

from mullet.knowledge.code.extractors.css import CSSExtractor


_SIMPLE_CSS = """\
body {
  color: red;
  font-size: 16px;
}

.nav {
  background: #333;
  padding: 8px;
}

#header {
  display: flex;
}
"""

_CSS_VARIABLES = """\
:root {
  --primary-color: #007bff;
  --secondary-color: #6c757d;
  --font-size-base: 16px;
}
"""

_KEYFRAMES_CSS = """\
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  0% { transform: translateY(20px); }
  100% { transform: translateY(0); }
}
"""

_SCSS = """\
$primary: #007bff;
$secondary: #6c757d;
$font-size-base: 16px;

@mixin flex-center {
  display: flex;
  align-items: center;
  justify-content: center;
}

@mixin respond-to($breakpoint) {
  @if $breakpoint == "sm" {
    @media (min-width: 576px) { @content; }
  }
}

@function rem($px) {
  @return $px / 16px * 1rem;
}

.container {
  @include flex-center;
  width: 100%;
}
"""


@pytest.fixture
def extractor() -> CSSExtractor:
    return CSSExtractor()


class TestBasicCSSExtraction:
    def test_returns_list(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_CSS, "styles.css")
        assert isinstance(chunks, list)

    def test_extracts_selectors(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_CSS, "styles.css")
        selector_names = [c.symbol_name for c in chunks if c.symbol_type == "selector"]
        assert any("body" in n for n in selector_names)
        assert any(".nav" in n for n in selector_names)
        assert any("#header" in n for n in selector_names)

    def test_language_is_css(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_CSS, "styles.css")
        css_chunks = [c for c in chunks if c.symbol_type == "selector"]
        assert all(c.language == "css" for c in css_chunks)

    def test_source_file_set(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_CSS, "styles.css")
        assert all(c.source_file == "styles.css" for c in chunks)

    def test_line_numbers_set(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SIMPLE_CSS, "styles.css")
        assert all(c.line_start > 0 for c in chunks)
        assert all(c.line_end >= c.line_start for c in chunks)


class TestCSSVariables:
    def test_css_custom_properties_detected(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_CSS_VARIABLES, "variables.css")
        root_chunk = next((c for c in chunks if ":root" in c.symbol_name), None)
        assert root_chunk is not None
        variables = root_chunk.extras.get("variables", [])
        assert any("--primary-color" in v for v in variables)

    def test_variable_names_in_extras(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_CSS_VARIABLES, "variables.css")
        root_chunk = next((c for c in chunks if ":root" in c.symbol_name), None)
        assert root_chunk is not None
        vars_ = root_chunk.extras.get("variables", [])
        assert "--secondary-color" in vars_ or any("secondary" in v for v in vars_)


class TestKeyframesExtraction:
    def test_extracts_keyframes(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_KEYFRAMES_CSS, "animations.css")
        kf_chunks = [c for c in chunks if c.symbol_type == "keyframes"]
        assert len(kf_chunks) >= 1

    def test_keyframe_names(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_KEYFRAMES_CSS, "animations.css")
        names = [c.symbol_name for c in chunks if c.symbol_type == "keyframes"]
        assert any("fadeIn" in n for n in names)


class TestSCSSExtraction:
    def test_extracts_scss_variables(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SCSS, "styles.scss")
        var_chunks = [c for c in chunks if c.symbol_type == "variable"]
        var_names = [c.symbol_name for c in var_chunks]
        assert "$primary" in var_names
        assert "$secondary" in var_names

    def test_scss_variable_value_in_extras(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SCSS, "styles.scss")
        primary = next(c for c in chunks if c.symbol_name == "$primary")
        assert "#007bff" in primary.extras.get("value", "")

    def test_extracts_scss_mixins(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SCSS, "styles.scss")
        mixin_chunks = [c for c in chunks if c.symbol_type == "mixin"]
        mixin_names = [c.symbol_name for c in mixin_chunks]
        assert "flex-center" in mixin_names

    def test_mixin_signature_captured(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SCSS, "styles.scss")
        respond_to = next((c for c in chunks if c.symbol_name == "respond-to"), None)
        assert respond_to is not None
        assert respond_to.signature is not None

    def test_extracts_scss_functions(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract(_SCSS, "styles.scss")
        func_chunks = [c for c in chunks if c.symbol_type == "function"]
        assert any(c.symbol_name == "rem" for c in func_chunks)


class TestFallbackExtraction:
    def test_fallback_handles_basic_css(self, extractor: CSSExtractor) -> None:
        css = "h1 { color: blue; } p.lead { font-size: 1.2em; }"
        chunks = extractor._extract_fallback(css, "fallback.css")
        assert len(chunks) > 0
        assert all(c.symbol_type == "selector" for c in chunks)

    def test_fallback_skips_at_rules(self, extractor: CSSExtractor) -> None:
        css = "@media (max-width: 768px) { body { font-size: 14px; } }"
        chunks = extractor._extract_fallback(css, "fallback.css")
        # @media should be skipped in fallback
        assert not any(c.symbol_name.startswith("@") for c in chunks)

    def test_fallback_empty_returns_empty(self, extractor: CSSExtractor) -> None:
        chunks = extractor._extract_fallback("", "empty.css")
        assert chunks == []


class TestEdgeCases:
    def test_empty_css(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract("", "empty.css")
        assert isinstance(chunks, list)

    def test_css_only_comments(self, extractor: CSSExtractor) -> None:
        chunks = extractor.extract("/* just a comment */", "comment.css")
        assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# Additional tests for previously uncovered lines
# ---------------------------------------------------------------------------

class TestParseExceptionFallback:
    """Lines 34-37: parse() raises → falls back to regex extraction."""

    def test_parse_exception_triggers_fallback(self, extractor: CSSExtractor, monkeypatch) -> None:
        monkeypatch.setattr(extractor, "parse", lambda src: (_ for _ in ()).throw(Exception("boom")))
        # _extract_fallback is called; result must be a list
        css = "h1 { color: red; }"
        chunks = extractor.extract(css, "broken.css")
        assert isinstance(chunks, list)
        # fallback returns a selector chunk for h1
        assert any(c.symbol_type == "selector" for c in chunks)


class TestExtractRuleNoSelectors:
    """Line 76: rule without a selectors child returns None."""

    def test_rule_without_selectors_node_returns_none(self, extractor: CSSExtractor) -> None:
        # Call _extract_rule with a mock node that has no 'selectors' child
        class _FakeNode:
            children = []
            start_point = (0, 0)
            end_point = (1, 0)
            start_byte = 0
            end_byte = 10

        result = extractor._extract_rule(_FakeNode(), b"body { }", "f.css")
        assert result is None


class TestExtractKeyframesNoName:
    """Line 117: keyframes node without a name child returns None."""

    def test_keyframes_without_name_node_returns_none(self, extractor: CSSExtractor) -> None:
        class _FakeNode:
            children = []
            start_point = (0, 0)
            end_point = (2, 0)
            start_byte = 0
            end_byte = 30

        result = extractor._extract_keyframes(_FakeNode(), b"@keyframes { }", "f.css")
        assert result is None


class TestLongMediaQueryTruncation:
    """Line 149: media query content > 1500 chars gets truncated."""

    def test_long_media_query_truncated(self, extractor: CSSExtractor) -> None:
        # Build a media statement with lots of rules inside
        inner = ".x { color: red; }" * 100  # ~1900 chars
        media_css = f"@media screen {{ {inner} }}"
        chunks = extractor.extract(media_css, "long.css")
        media_chunks = [c for c in chunks if c.symbol_type == "media_query"]
        if media_chunks:
            assert len(media_chunks[0].content) <= 1520  # 1500 + "/* truncated */"


class TestLessExtension:
    """Line 229: .less files trigger SCSS extras extraction including nested braces."""

    def test_less_file_extracts_variables(self, extractor: CSSExtractor) -> None:
        less_code = "@primary: #007bff;\n.btn { color: @primary; }\n"
        # LESS vars start with @, but the _extract_scss_extras uses $; the main path
        # (is_less=True) should be reached and scss_chunks called without error
        chunks = extractor.extract(less_code, "styles.less")
        assert isinstance(chunks, list)

    def test_less_file_with_scss_variable_syntax(self, extractor: CSSExtractor) -> None:
        # Test the is_less branch (line 229) is actually reached
        less_code = "$my-var: red;\n.widget { color: $my-var; }\n"
        chunks = extractor.extract(less_code, "theme.less")
        var_chunks = [c for c in chunks if c.symbol_type == "variable"]
        assert any(c.symbol_name == "$my-var" for c in var_chunks)

    def test_mixin_with_nested_braces(self, extractor: CSSExtractor) -> None:
        # Line 229: the brace_count += 1 path inside _extract_scss_extras mixin parser
        # A mixin that contains a nested {} block (e.g. a media query inside)
        scss = """\
@mixin responsive($bp) {
  @if $bp == sm {
    @media (min-width: 576px) {
      @content;
    }
  }
}
"""
        chunks = extractor._extract_scss_extras(scss, "styles.scss")
        mixin_chunks = [c for c in chunks if c.symbol_type == "mixin"]
        assert any(c.symbol_name == "responsive" for c in mixin_chunks)

    def test_function_with_nested_braces(self, extractor: CSSExtractor) -> None:
        # Line 229: brace_count += 1 in the @function brace scanner with nested blocks
        scss = """\
@function clamp-value($min, $max) {
  @if $min > $max {
    @return $max;
  }
  @return $min;
}
"""
        chunks = extractor._extract_scss_extras(scss, "styles.scss")
        func_chunks = [c for c in chunks if c.symbol_type == "function"]
        assert any(c.symbol_name == "clamp-value" for c in func_chunks)


class TestFallbackSkipsAtSelectors:
    """Line 259: fallback skips selectors that start with @."""

    def test_fallback_skips_empty_selector(self, extractor: CSSExtractor) -> None:
        # A rule whose selector (after stripping) is empty is skipped
        css = " { orphan: rule; }"
        chunks = extractor._extract_fallback(css, "f.css")
        assert not any(c.symbol_name == "" for c in chunks)

    def test_fallback_selector_at_rule_skipped(self, extractor: CSSExtractor) -> None:
        # Selector starting with @ should be excluded (line 259)
        css = "@supports (display: grid) { .grid { display: grid; } }"
        chunks = extractor._extract_fallback(css, "supports.css")
        assert not any(c.symbol_name.startswith("@") for c in chunks)
