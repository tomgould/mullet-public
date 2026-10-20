"""Tests for search utilities — contextualization + reference resolution."""
from mullet.knowledge.engine.search.contextualize import query_contextualize
from mullet.knowledge.engine.search.reference_resolver import extract_references


def test_self_contained_query_unchanged():
    assert query_contextualize("What is a self-contained query?", []) == "What is a self-contained query?"


def test_self_contained_with_history_unchanged():
    result = query_contextualize("How does the router work?", ["search for auth"])
    assert result == "How does the router work?"


def test_follow_up_expanded():
    result = query_contextualize("what about the tests?", ["How does the router work?"])
    assert "router" in result
    assert "tests" in result


def test_short_follow_up_expanded():
    result = query_contextualize("tests?", ["How does the router work?"])
    assert "router" in result


def test_extract_refs_finds_see_also():
    refs = extract_references('See also the Authentication Module for details.')
    assert len(refs) > 0


def test_extract_refs_finds_appendix():
    refs = extract_references('Refer to Appendix B for the full specification.')
    assert any("B" in r for r in refs)


def test_extract_refs_finds_defined_in():
    refs = extract_references('The handler is defined in `auth/handler.py`.')
    assert any("auth/handler.py" in r for r in refs)


def test_extract_refs_empty_for_plain_text():
    refs = extract_references("This is plain text with no references at all.")
    assert refs == []
