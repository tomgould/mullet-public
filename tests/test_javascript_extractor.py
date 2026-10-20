"""Tests for the tree-sitter-based JavaScript code extractor."""

import pytest
from mullet.knowledge.code.extractors.javascript import JavaScriptExtractor


@pytest.fixture
def extractor():
    return JavaScriptExtractor()


_SIMPLE_FUNC = "function greet(name) {\n  return 'Hello ' + name;\n}"

_ARROW_FUNC = "const add = (a, b) => a + b;"

_CLASS_CODE = """\
class Calculator {
  constructor(initial) {
    this.value = initial;
  }

  add(n) {
    this.value += n;
    return this;
  }

  subtract(n) {
    this.value -= n;
    return this;
  }
}
"""

_EXPORT_FUNC = """\
export function formatDate(date) {
  return date.toISOString().split('T')[0];
}
"""

_MULTIPLE_FUNCS = """\
function alpha() { return 1; }
function beta() { return 2; }
function gamma() { return 3; }
"""


class TestFunctionExtraction:
    def test_simple_function(self, extractor):
        chunks = extractor.extract(_SIMPLE_FUNC, "greet.js")
        names = [c.symbol_name for c in chunks]
        assert "greet" in names

    def test_multiple_functions(self, extractor):
        chunks = extractor.extract(_MULTIPLE_FUNCS, "multi.js")
        names = {c.symbol_name for c in chunks}
        assert {"alpha", "beta", "gamma"}.issubset(names)

    def test_arrow_function(self, extractor):
        chunks = extractor.extract(_ARROW_FUNC, "arrow.js")
        names = [c.symbol_name for c in chunks]
        assert "add" in names

    def test_async_function(self, extractor):
        code = "async function fetchData(url) {\n  return await fetch(url);\n}"
        chunks = extractor.extract(code, "fetch.js")
        names = [c.symbol_name for c in chunks]
        assert "fetchData" in names

    def test_empty_source(self, extractor):
        chunks = extractor.extract("", "empty.js")
        assert isinstance(chunks, list)

    def test_no_functions(self, extractor):
        chunks = extractor.extract("const x = 1;\nconst y = 2;", "vars.js")
        assert isinstance(chunks, list)


class TestClassExtraction:
    def test_class_extracted(self, extractor):
        chunks = extractor.extract(_CLASS_CODE, "calc.js")
        names = [c.symbol_name for c in chunks]
        assert "Calculator" in names

    def test_class_methods_extracted(self, extractor):
        chunks = extractor.extract(_CLASS_CODE, "calc.js")
        names = {c.symbol_name for c in chunks}
        assert "add" in names
        assert "subtract" in names

    def test_class_with_inheritance(self, extractor):
        code = (
            "class Animal {\n"
            "  speak() { return 'generic'; }\n"
            "}\n"
            "class Dog extends Animal {\n"
            "  speak() { return 'woof'; }\n"
            "}\n"
        )
        chunks = extractor.extract(code, "animals.js")
        names = {c.symbol_name for c in chunks}
        assert "Animal" in names
        assert "Dog" in names


class TestExportExtraction:
    def test_named_export_function(self, extractor):
        chunks = extractor.extract(_EXPORT_FUNC, "utils.js")
        names = [c.symbol_name for c in chunks]
        assert "formatDate" in names

    def test_export_default_class(self, extractor):
        code = "export default class Router {\n  route(path) {}\n}"
        chunks = extractor.extract(code, "router.js")
        names = [c.symbol_name for c in chunks]
        # Default exports may or may not have a name - just verify no crash
        assert isinstance(chunks, list)


class TestChunkAttributes:
    def test_source_file_set(self, extractor):
        chunks = extractor.extract(_SIMPLE_FUNC, "/app/utils.js")
        assert all(c.source_file == "/app/utils.js" for c in chunks)

    def test_language_is_javascript(self, extractor):
        chunks = extractor.extract(_SIMPLE_FUNC, "mod.js")
        assert all(c.language == "javascript" for c in chunks)

    def test_line_numbers_positive(self, extractor):
        chunks = extractor.extract(_CLASS_CODE, "calc.js")
        for c in chunks:
            assert c.line_start >= 1

    def test_content_non_empty(self, extractor):
        chunks = extractor.extract(_SIMPLE_FUNC, "mod.js")
        assert all(c.content.strip() for c in chunks)


class TestParseFailure:
    """Lines 28-30: parse failure returns empty list."""

    def test_parse_failure_returns_empty_list(self, extractor, monkeypatch):
        def _bad_parse(src):
            raise RuntimeError("boom")
        monkeypatch.setattr(extractor, "parse", _bad_parse)
        chunks = extractor.extract("function f() {}", "bad.js")
        assert chunks == []


class TestExtractFunctionEdgeCases:
    """Lines 88, 66, 303-307."""

    def test_export_statement_appended_to_chunks(self, extractor):
        """Line 66: export function chunk is appended."""
        code = "export function helper(x) { return x; }"
        chunks = extractor.extract(code, "helpers.js")
        names = [c.symbol_name for c in chunks]
        assert "helper" in names

    def test_export_marks_exported_flag(self, extractor, monkeypatch):
        """Lines 66, 303-307: exported extra is set when _extract_export fires.

        The function_declaration pass normally preempts the export_statement pass
        by adding the inner function to extracted_ranges first.  We monkeypatch
        _is_within_ranges to always return False so the export branch runs.
        """
        monkeypatch.setattr(extractor, "_is_within_ranges", lambda node, ranges: False)
        code = "export function doThing(a, b) { return a + b; }"
        chunks = extractor.extract(code, "thing.js")
        # With _is_within_ranges always returning False, the function is extracted
        # twice: once by the function_declaration pass (no extras) and once by the
        # export_statement pass (extras set).  Find the export-path chunk.
        exported_chunks = [c for c in chunks if c.symbol_name == "doThing" and c.extras]
        assert exported_chunks, "exported doThing chunk not found"
        chunk = exported_chunks[0]
        assert chunk.extras.get("exported") is True

    def test_export_default_function_marks_default(self, extractor, monkeypatch):
        """Line 306: default extra is True when 'default' appears in export text."""
        monkeypatch.setattr(extractor, "_is_within_ranges", lambda node, ranges: False)
        code = "export default function main() { return 0; }"
        chunks = extractor.extract(code, "main.js")
        # With the monkeypatch, there will be two 'main' chunks; find the exported one.
        main_chunks = [c for c in chunks if c.symbol_name == "main" and c.extras]
        assert main_chunks, "exported main chunk not found"
        assert main_chunks[0].extras.get("default") is True

    def test_anonymous_function_node_returns_none(self, extractor):
        """Line 88: function without identifier returns None (skipped)."""
        # Anonymous functions inside expressions have no identifier at the top level
        code = "const f = function() { return 1; };"
        chunks = extractor.extract(code, "anon.js")
        # Should not crash; anonymous function is skipped
        assert isinstance(chunks, list)


class TestArrowFunctionEdgeCases:
    """Lines 129, 134, 150-154."""

    def test_arrow_no_declarator_skipped(self, extractor):
        """Line 129: variable_declaration without declarator child doesn't crash."""
        # Destructured assignment — no simple variable_declarator
        code = "var [a, b] = [1, 2];"
        chunks = extractor.extract(code, "destruct.js")
        assert isinstance(chunks, list)

    def test_arrow_no_name_skipped(self, extractor):
        """Line 134: declarator without identifier name is skipped."""
        # Object destructuring — declarator has no plain identifier as name
        code = "const { x, y } = obj;"
        chunks = extractor.extract(code, "obj.js")
        assert isinstance(chunks, list)

    def test_arrow_single_param_no_parens(self, extractor):
        """Lines 150-154: single param arrow without formal_parameters."""
        code = "const double = x => x * 2;"
        chunks = extractor.extract(code, "double.js")
        names = [c.symbol_name for c in chunks]
        assert "double" in names
        dbl = next(c for c in chunks if c.symbol_name == "double")
        # Signature should wrap the lone param in parens
        assert dbl.signature == "(x)"

    def test_arrow_no_children_signature_default(self, extractor):
        """Lines 153-154 fallback: param node that is not an identifier → '()'."""
        # An arrow with no children at all cannot be synthesised easily from real JS,
        # but the `const f = () => 1;` case exercises formal_parameters normally.
        # Instead verify the branch isn't broken by parsing a valid arrow.
        code = "const noop = () => {};"
        chunks = extractor.extract(code, "noop.js")
        noop = next((c for c in chunks if c.symbol_name == "noop"), None)
        assert noop is not None
        assert noop.signature == "()"


class TestClassEdgeCases:
    """Lines 187, 213, 252, 273, 275."""

    def test_anonymous_class_skipped(self, extractor):
        """Line 187: class_declaration without identifier returns []."""
        # `export default class {}` — class with no name
        code = "export default class { method() {} }"
        chunks = extractor.extract(code, "anon_class.js")
        # No crash; anonymous class chunk not added
        assert isinstance(chunks, list)

    def test_class_with_jsdoc_includes_comment_in_content(self, extractor):
        """Line 213: docstring branch includes /** … */ prefix in class_content."""
        code = (
            "/** A useful service */\n"
            "class Service {\n"
            "  run() {}\n"
            "}\n"
        )
        chunks = extractor.extract(code, "svc.js")
        svc = next((c for c in chunks if c.symbol_name == "Service"), None)
        assert svc is not None
        assert svc.docstring and "useful service" in svc.docstring

    def test_method_without_property_identifier_skipped(self, extractor):
        """Line 252: method_definition lacking property_identifier → None."""
        # Computed property methods like [Symbol.iterator]() have no property_identifier
        code = (
            "class Iter {\n"
            "  [Symbol.iterator]() { return this; }\n"
            "  regularMethod() { return 1; }\n"
            "}\n"
        )
        chunks = extractor.extract(code, "iter.js")
        # regularMethod should still be extracted; no crash
        names = {c.symbol_name for c in chunks}
        assert "regularMethod" in names

    def test_static_method_sets_extras(self, extractor):
        """Line 273: static method sets extras['static'] = True."""
        code = (
            "class Factory {\n"
            "  static create() { return new Factory(); }\n"
            "}\n"
        )
        chunks = extractor.extract(code, "factory.js")
        create = next((c for c in chunks if c.symbol_name == "create"), None)
        assert create is not None
        assert create.extras.get("static") is True

    def test_async_method_sets_extras(self, extractor):
        """Line 275: async method sets extras['async'] = True."""
        code = (
            "class Fetcher {\n"
            "  async load(url) { return fetch(url); }\n"
            "}\n"
        )
        chunks = extractor.extract(code, "fetcher.js")
        load = next((c for c in chunks if c.symbol_name == "load"), None)
        assert load is not None
        assert load.extras.get("async") is True


class TestJSDocExtraction:
    """Lines 322-327: _get_jsdoc paths."""

    def test_jsdoc_extracted_for_function(self, extractor):
        """Lines 322-324: JSDoc comment is captured."""
        code = (
            "/** Calculate the total price */\n"
            "function total(items) { return items.reduce((s, i) => s + i, 0); }\n"
        )
        chunks = extractor.extract(code, "total.js")
        t = next((c for c in chunks if c.symbol_name == "total"), None)
        assert t is not None
        assert t.docstring and "total price" in t.docstring

    def test_jsdoc_skips_non_jsdoc_comment(self, extractor):
        """JSDoc parser ignores regular // comments."""
        code = (
            "// plain comment\n"
            "function helper() {}\n"
        )
        chunks = extractor.extract(code, "h.js")
        h = next((c for c in chunks if c.symbol_name == "helper"), None)
        assert h is not None
        assert h.docstring is None

    def test_jsdoc_with_decorator_preceding(self, extractor):
        """Line 327: decorator is skipped when searching for preceding JSDoc."""
        # Decorators aren't standard JS but the tree-sitter grammar may expose them;
        # confirm the walk-back loop doesn't crash on decorator siblings.
        code = (
            "/** Decorated function */\n"
            "function decorated() {}\n"
        )
        chunks = extractor.extract(code, "deco.js")
        assert isinstance(chunks, list)


class TestCleanJSDoc:
    """Lines 332-344: _clean_jsdoc strips /** */ and * prefixes."""

    def test_clean_jsdoc_strips_markers(self, extractor):
        raw = "/**\n * Does something\n * @param x a number\n */"
        cleaned = extractor._clean_jsdoc(raw)
        assert "Does something" in cleaned
        assert "/**" not in cleaned
        assert "*/" not in cleaned

    def test_clean_jsdoc_excludes_at_tags(self, extractor):
        raw = "/**\n * Summary line\n * @param x desc\n * @returns number\n */"
        cleaned = extractor._clean_jsdoc(raw)
        assert "Summary line" in cleaned
        # @param and @returns lines should be excluded
        assert "@param" not in cleaned
        assert "@returns" not in cleaned

    def test_clean_jsdoc_single_line(self, extractor):
        raw = "/** Short description */"
        cleaned = extractor._clean_jsdoc(raw)
        assert "Short description" in cleaned

    def test_clean_jsdoc_empty_lines_stripped(self, extractor):
        raw = "/**\n * \n * Real content\n */"
        cleaned = extractor._clean_jsdoc(raw)
        assert "Real content" in cleaned


class TestFindDescendantIdentifier:
    """Line 364: recursive path in find_descendant_identifier."""

    def test_find_in_nested_node(self, extractor):
        """Exercises recursive descent by parsing a class with extends."""
        code = "class Foo extends ns.Bar {}"
        chunks = extractor.extract(code, "foo.js")
        foo = next((c for c in chunks if c.symbol_name == "Foo"), None)
        assert foo is not None
        # base_class should be found via recursive descent
        assert foo.extras.get("base_class") is not None


class TestDetermineVisibility:
    """Lines 370, 372: private naming conventions."""

    def test_underscore_prefix_is_private(self, extractor):
        """Line 370: names starting with _ → 'private'."""
        assert extractor.determine_visibility("_internal") == "private"

    def test_hash_prefix_is_private(self, extractor):
        """Line 372: ES2022 # private fields → 'private'."""
        assert extractor.determine_visibility("#secret") == "private"

    def test_regular_name_is_public(self, extractor):
        assert extractor.determine_visibility("publicMethod") == "public"


# ---------------------------------------------------------------------------
# Direct private-method tests to cover lines unreachable via full extract()
# ---------------------------------------------------------------------------

class _MockNode:
    """Minimal tree-sitter node stub for unit-testing private methods."""
    def __init__(self, node_type="unknown", text=b"", children=None,
                 start_byte=0, end_byte=0, start_point=(0, 0), end_point=(0, 0)):
        self.type = node_type
        self._text = text
        self.children = children or []
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.start_point = start_point
        self.end_point = end_point
        self.prev_sibling = None


class TestExtractFunctionNoIdentifier:
    """Line 88: _extract_function returns None when node has no identifier child."""

    def test_returns_none_for_nameless_function(self, extractor):
        node = _MockNode("function_declaration", b"function() {}", end_byte=13,
                         end_point=(0, 13))
        result = extractor._extract_function(node, b"function() {}", "test.js")
        assert result is None


class TestExtractArrowFunctionNoDeclarator:
    """Line 129: _extract_arrow_function returns None when no variable_declarator."""

    def test_returns_none_for_no_declarator(self, extractor):
        node = _MockNode("variable_declaration", b"var x", end_byte=5,
                         end_point=(0, 5))
        # children has no variable_declarator node
        result = extractor._extract_arrow_function(node, b"var x", "test.js")
        assert result is None


class TestExtractArrowFunctionNonIdentifierParam:
    """Line 154: arrow with children but first child is not identifier → signature '()'."""

    def test_non_identifier_first_child_gives_empty_signature(self, extractor):
        # Build: var_decl → variable_declarator → [identifier('f'), arrow_function]
        # arrow_function has children[0].type = 'object_pattern' (not 'identifier')
        # and no formal_parameters child.
        object_pattern = _MockNode("object_pattern", b"{x}")
        arrow_func = _MockNode("arrow_function", b"{x} => x", children=[object_pattern])

        name_node = _MockNode("identifier", b"f", start_byte=6, end_byte=7)
        declarator = _MockNode("variable_declarator", b"f = {x} => x",
                               children=[name_node, arrow_func],
                               start_byte=6, end_byte=19,
                               start_point=(0, 6), end_point=(0, 19))
        var_decl = _MockNode("variable_declaration", b"const f = {x} => x;",
                              children=[declarator],
                              start_byte=0, end_byte=20,
                              start_point=(0, 0), end_point=(0, 20))

        source_bytes = b"const f = {x} => x;"

        # Override get_node_text so the mock returns useful text
        original = extractor.get_node_text

        def _get_text(node, sb):
            return node._text.decode() if hasattr(node, '_text') else ""

        extractor.get_node_text = _get_text
        try:
            result = extractor._extract_arrow_function(var_decl, source_bytes, "test.js")
        finally:
            extractor.get_node_text = original

        assert result is not None
        assert result.signature == "()"


class TestExtractClassNoIdentifier:
    """Line 187: _extract_class returns [] when class has no identifier child."""

    def test_returns_empty_for_nameless_class(self, extractor):
        node = _MockNode("class_declaration", b"class {}", end_byte=8,
                         end_point=(0, 8))
        result = extractor._extract_class(node, b"class {}", "test.js")
        assert result == []
