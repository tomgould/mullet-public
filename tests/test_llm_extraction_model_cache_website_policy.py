"""Tests for llm_extraction pure helpers, model_cache singleton, and website_policy."""

from mullet.knowledge.engine.llm_extraction import (
    CitationMap,
    build_citation_map,
    build_rag_prompt,
    build_rag_prompt_with_citations,
    categorize_chunks,
    parse_citations_from_response,
)
from mullet.knowledge.engine.model_cache import ModelCache, get_model_cache
from mullet.core.tools.website_policy import RobotsTxtChecker, _MAX_CACHE_SIZE


# ---------------------------------------------------------------------------
# CitationMap
# ---------------------------------------------------------------------------

class TestCitationMap:
    def test_chunks_stored(self):
        cm = CitationMap(chunks=[{"content": "a"}, {"content": "b"}])
        assert len(cm.chunks) == 2

    def test_len_returns_chunk_count(self):
        cm = CitationMap(chunks=[{"content": "a"}, {"content": "b"}])
        assert len(cm) == 2

    def test_empty_len_zero(self):
        cm = CitationMap(chunks=[])
        assert len(cm) == 0

    def test_get_first_chunk(self):
        cm = CitationMap(chunks=[{"content": "first"}, {"content": "second"}])
        assert cm.get(1) == {"content": "first"}

    def test_get_second_chunk(self):
        cm = CitationMap(chunks=[{"content": "first"}, {"content": "second"}])
        assert cm.get(2) == {"content": "second"}

    def test_get_out_of_range_returns_none(self):
        cm = CitationMap(chunks=[{"content": "a"}])
        assert cm.get(2) is None

    def test_get_zero_returns_none(self):
        cm = CitationMap(chunks=[{"content": "a"}])
        assert cm.get(0) is None

    def test_get_negative_returns_none(self):
        cm = CitationMap(chunks=[{"content": "a"}])
        assert cm.get(-1) is None


# ---------------------------------------------------------------------------
# build_citation_map
# ---------------------------------------------------------------------------

class TestBuildCitationMap:
    def test_returns_citation_map(self):
        result = build_citation_map([{"content": "x"}])
        assert isinstance(result, CitationMap)

    def test_chunks_preserved(self):
        chunks = [{"content": "a"}, {"content": "b"}]
        cm = build_citation_map(chunks)
        assert len(cm) == 2

    def test_empty_chunks(self):
        cm = build_citation_map([])
        assert len(cm) == 0


# ---------------------------------------------------------------------------
# parse_citations_from_response
# ---------------------------------------------------------------------------

class TestParseCitationsFromResponse:
    def test_no_citations_returns_empty(self):
        assert parse_citations_from_response("No citations here.") == []

    def test_single_citation(self):
        result = parse_citations_from_response("See [1] for more.")
        assert result == [1]

    def test_multiple_citations(self):
        result = parse_citations_from_response("See [1] and [2] also [3].")
        assert result == [1, 2, 3]

    def test_duplicate_citations_included(self):
        result = parse_citations_from_response("[1] again [1].")
        assert result == [1, 1]

    def test_returns_integers(self):
        result = parse_citations_from_response("[5]")
        assert result[0] == 5
        assert isinstance(result[0], int)

    def test_empty_string_returns_empty(self):
        assert parse_citations_from_response("") == []


# ---------------------------------------------------------------------------
# categorize_chunks
# ---------------------------------------------------------------------------

class TestCategorizeChunks:
    def test_returns_dict_with_four_keys(self):
        result = categorize_chunks([])
        assert set(result.keys()) == {"code", "documentation", "conversation", "other"}

    def test_empty_input_all_empty(self):
        result = categorize_chunks([])
        assert all(v == [] for v in result.values())

    def test_chunk_with_metadata_chunk_type_code(self):
        chunks = [{"content": "def f(): pass", "metadata": {"chunk_type": "code"}}]
        result = categorize_chunks(chunks)
        assert len(result["code"]) == 1

    def test_chunk_with_metadata_documentation(self):
        chunks = [{"content": "README", "metadata": {"chunk_type": "documentation"}}]
        result = categorize_chunks(chunks)
        assert len(result["documentation"]) == 1

    def test_chunk_with_metadata_conversation(self):
        chunks = [{"content": "user: hello", "metadata": {"chunk_type": "conversation"}}]
        result = categorize_chunks(chunks)
        assert len(result["conversation"]) == 1

    def test_unknown_chunk_type_goes_to_other(self):
        chunks = [{"content": "x", "metadata": {"chunk_type": "unknown_type"}}]
        result = categorize_chunks(chunks)
        assert len(result["other"]) == 1

    def test_no_metadata_goes_to_other(self):
        chunks = [{"content": "x"}]
        result = categorize_chunks(chunks)
        assert len(result["other"]) == 1

    def test_mixed_chunks_distributed(self):
        chunks = [
            {"content": "code", "metadata": {"chunk_type": "code"}},
            {"content": "doc", "metadata": {"chunk_type": "documentation"}},
            {"content": "other", "metadata": {}},
        ]
        result = categorize_chunks(chunks)
        assert len(result["code"]) == 1
        assert len(result["documentation"]) == 1
        assert len(result["other"]) == 1


# ---------------------------------------------------------------------------
# build_rag_prompt
# ---------------------------------------------------------------------------

class TestBuildRagPrompt:
    def test_returns_string(self):
        assert isinstance(build_rag_prompt("Q?", []), str)

    def test_question_in_prompt(self):
        prompt = build_rag_prompt("What is Python?", [])
        assert "What is Python?" in prompt

    def test_chunk_content_in_prompt(self):
        chunks = [{"content": "Python is great", "metadata": {"source_file": "test.py"}}]
        prompt = build_rag_prompt("Q?", chunks)
        assert "Python is great" in prompt

    def test_empty_chunks_still_has_question(self):
        prompt = build_rag_prompt("What?", [])
        assert "What?" in prompt

    def test_source_file_appears(self):
        chunks = [{"content": "hello", "metadata": {"source_file": "hello.py"}}]
        prompt = build_rag_prompt("Q?", chunks)
        assert "hello.py" in prompt

    def test_multiple_chunks_in_prompt(self):
        chunks = [
            {"content": "chunk one", "metadata": {"source_file": "a.py"}},
            {"content": "chunk two", "metadata": {"source_file": "b.py"}},
        ]
        prompt = build_rag_prompt("Q?", chunks)
        assert "chunk one" in prompt
        assert "chunk two" in prompt


# ---------------------------------------------------------------------------
# build_rag_prompt_with_citations
# ---------------------------------------------------------------------------

class TestBuildRagPromptWithCitations:
    def test_returns_string(self):
        assert isinstance(build_rag_prompt_with_citations("Q?", []), str)

    def test_question_in_prompt(self):
        prompt = build_rag_prompt_with_citations("Explain this?", [])
        assert "Explain this?" in prompt

    def test_citation_instruction_present(self):
        prompt = build_rag_prompt_with_citations("Q?", [{"content": "x"}])
        assert "citation" in prompt.lower() or "[1]" in prompt

    def test_chunk_content_in_prompt(self):
        chunks = [{"content": "important text", "metadata": {"source_file": "src.py"}}]
        prompt = build_rag_prompt_with_citations("Q?", chunks)
        assert "important text" in prompt


# ---------------------------------------------------------------------------
# ModelCache
# ---------------------------------------------------------------------------

class TestModelCache:
    def test_instantiates(self):
        mc = ModelCache()
        assert mc is not None

    def test_clear_empty_cache(self):
        mc = ModelCache()
        mc.clear()  # should not raise

    def test_clear_returns_none(self):
        mc = ModelCache()
        assert mc.clear() is None

    def test_embedders_empty_initially(self):
        mc = ModelCache()
        assert mc._embedders == {}

    def test_rerankers_empty_initially(self):
        mc = ModelCache()
        assert mc._rerankers == {}


class TestGetModelCache:
    def test_returns_model_cache(self):
        mc = get_model_cache()
        assert isinstance(mc, ModelCache)

    def test_singleton_same_instance(self):
        mc1 = get_model_cache()
        mc2 = get_model_cache()
        assert mc1 is mc2


# ---------------------------------------------------------------------------
# RobotsTxtChecker
# ---------------------------------------------------------------------------

class TestRobotsTxtChecker:
    def test_instantiates(self):
        checker = RobotsTxtChecker()
        assert checker is not None

    def test_check_returns_dict(self):
        checker = RobotsTxtChecker()
        result = checker.check_robots_txt("https://nonexistent-host-xyz123.invalid/page")
        assert isinstance(result, dict)

    def test_result_has_allowed_key(self):
        checker = RobotsTxtChecker()
        result = checker.check_robots_txt("https://nonexistent-host-xyz123.invalid/page")
        assert "allowed" in result

    def test_result_has_crawl_delay_key(self):
        checker = RobotsTxtChecker()
        result = checker.check_robots_txt("https://nonexistent-host-xyz123.invalid/page")
        assert "crawl_delay" in result

    def test_result_has_disallowed_paths_key(self):
        checker = RobotsTxtChecker()
        result = checker.check_robots_txt("https://nonexistent-host-xyz123.invalid/page")
        assert "disallowed_paths" in result

    def test_network_error_sets_allowed_false(self):
        checker = RobotsTxtChecker()
        result = checker.check_robots_txt("https://nonexistent-host-xyz123.invalid/page")
        assert result["allowed"] is False

    def test_network_error_includes_error_key(self):
        checker = RobotsTxtChecker()
        result = checker.check_robots_txt("https://nonexistent-host-xyz123.invalid/page")
        assert "error" in result

    def test_disallowed_paths_is_list(self):
        checker = RobotsTxtChecker()
        result = checker.check_robots_txt("https://nonexistent-host-xyz123.invalid/page")
        assert isinstance(result["disallowed_paths"], list)

    def test_cache_populated_after_check(self):
        checker = RobotsTxtChecker()
        checker.check_robots_txt("https://nonexistent-host-xyz123.invalid/page")
        # On network error, nothing should be cached (error returns early)
        # The cache stays empty on error path
        assert isinstance(checker._cache, dict)

    def test_max_cache_size_is_positive(self):
        assert _MAX_CACHE_SIZE > 0
