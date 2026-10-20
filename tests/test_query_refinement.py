import pytest
from mullet.core.knowledge.query_refinement import QueryRefiner

@pytest.fixture
def refiner():
    return QueryRefiner()

def test_expand_query_single_word(refiner):
    expanded = refiner.expand_query("search")
    assert expanded == "discover find lookup retrieve search"

def test_expand_query_multiple_words(refiner):
    expanded = refiner.expand_query("search code")
    assert expanded == "code discover find lookup program retrieve script search software"

def test_expand_query_no_synonyms(refiner):
    expanded = refiner.expand_query("hello world")
    assert expanded == "hello world"

def test_expand_query_case_insensitivity(refiner):
    expanded = refiner.expand_query("Search Code")
    assert expanded == "code discover find lookup program retrieve script search software"

def test_refine_basic_query(refiner):
    refined = refiner.refine("search document")
    assert refined == "article discover document file find lookup record retrieve search text"

def test_refine_query_with_context_ignored(refiner):
    refined = refiner.refine("query tool", context="irrelevant context")
    assert refined == "inquiry instrument mechanism query question request tool utility"

def test_refine_empty_query(refiner):
    refined = refiner.refine("")
    assert refined == ""

def test_refine_query_with_duplicates(refiner):
    refined = refiner.refine("search search")
    assert refined == "discover find lookup retrieve search"
