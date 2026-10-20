import pytest
from unittest.mock import MagicMock

from mullet.knowledge.code.extractors.typescript import TypeScriptExtractor


@pytest.fixture
def ts():
    return TypeScriptExtractor()


def test_extract_tsx_switches_language_and_restores(ts):
    captured = []

    def impl(src, fp):
        captured.append(ts.tree_sitter_language)
        return []

    ts._extract_impl = impl
    ts.extract("", "file.tsx")
    assert captured == ["tsx"]
    assert ts.tree_sitter_language == "typescript"
    assert ts._parser is None


def test_extract_ts_keeps_language(ts):
    captured = []

    def impl(src, fp):
        captured.append(ts.tree_sitter_language)
        return []

    ts._extract_impl = impl
    ts.extract("", "file.ts")
    assert captured == ["typescript"]


def test_get_interface_extends(ts):
    clause = MagicMock()
    type_id = MagicMock()
    ts.find_child_by_type = lambda n, t: clause if t == "extends_type_clause" else None
    ts.find_descendants_by_type = lambda n, t: [type_id] if t == "type_identifier" else []
    ts.get_node_text = lambda n, b: "Base"
    result = ts._get_interface_extends(MagicMock(), b"")
    assert result == ["Base"]


def test_get_interface_properties(ts):
    body = MagicMock()
    prop = MagicMock()
    name_node = MagicMock()

    def find_child(n, t):
        if t == "object_type":
            return body
        if t == "property_identifier":
            return name_node
        return None

    def find_descendants(n, t):
        if t == "property_signature":
            return [prop]
        return []

    def get_text(n, b):
        if n is name_node:
            return "name"
        return ""

    ts.find_child_by_type = find_child
    ts.find_descendants_by_type = find_descendants
    ts.get_node_text = get_text
    result = ts._get_interface_properties(MagicMock(), b"")
    assert result == ["name"]


def test_get_enum_members(ts):
    body = MagicMock()
    member = MagicMock()
    name_node = MagicMock()

    def find_child(n, t):
        if t == "enum_body":
            return body
        if t == "property_identifier":
            return name_node
        return None

    def find_children(n, t):
        if t == "enum_assignment":
            return [member]
        if t == "property_identifier":
            return [name_node]
        return []

    def get_text(n, b):
        if n is name_node:
            return "A"
        return ""

    ts.find_child_by_type = find_child
    ts.find_children_by_type = find_children
    ts.get_node_text = get_text
    result = ts._get_enum_members(MagicMock(), b"")
    assert "A" in result


def test_clean_jsdoc(ts):
    raw = "/**\n * Line one\n * Line two\n */"
    result = ts._clean_jsdoc(raw)
    assert "Line one" in result
    assert "Line two" in result


def test_determine_visibility_public(ts):
    assert ts.determine_visibility("publicMethod") == "public"


def test_determine_visibility_private(ts):
    assert ts.determine_visibility("_internal") == "private"
    assert ts.determine_visibility("#field") == "private"
