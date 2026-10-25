"""Tests for llm_extraction.py and pool_metadata.py."""
import json
import pytest

from mullet.knowledge.engine.llm_extraction import (
    CitationMap,
    build_citation_map,
    parse_citations_from_response,
    _build_context_parts,
    build_rag_prompt,
    categorize_chunks,
    build_rag_prompt_with_citations,
)
from mullet.knowledge.engine.pool_metadata import (
    read_metadata,
    ensure_created,
    mark_searched,
    mark_indexed,
    delete_pool,
    METADATA_FILENAME,
)


# ===========================================================================
# CitationMap
# ===========================================================================

class TestCitationMap:
    def test_get_first_chunk(self):
        cm = CitationMap(chunks=[{"id": "a"}, {"id": "b"}])
        assert cm.get(1) == {"id": "a"}

    def test_get_last_chunk(self):
        cm = CitationMap(chunks=[{"id": "a"}, {"id": "b"}])
        assert cm.get(2) == {"id": "b"}

    def test_get_zero_returns_none(self):
        cm = CitationMap(chunks=[{"id": "a"}])
        assert cm.get(0) is None

    def test_get_out_of_range_returns_none(self):
        cm = CitationMap(chunks=[{"id": "a"}])
        assert cm.get(99) is None

    def test_len(self):
        cm = CitationMap(chunks=[{}, {}, {}])
        assert len(cm) == 3

    def test_empty_len(self):
        assert len(CitationMap()) == 0


class TestBuildCitationMap:
    def test_wraps_chunks(self):
        chunks = [{"content": "x"}, {"content": "y"}]
        cm = build_citation_map(chunks)
        assert isinstance(cm, CitationMap)
        assert len(cm) == 2
        assert cm.get(1) == {"content": "x"}


# ===========================================================================
# parse_citations_from_response
# ===========================================================================

class TestParseCitationsFromResponse:
    def test_no_citations_returns_empty(self):
        assert parse_citations_from_response("No citations here.") == []

    def test_single_citation(self):
        assert parse_citations_from_response("See [1] for details.") == [1]

    def test_multiple_citations_ordered_by_appearance(self):
        assert parse_citations_from_response("See [2] and [1] and [3].") == [2, 1, 3]

    def test_duplicate_citations_preserved(self):
        assert parse_citations_from_response("[1] and [1] again.") == [1, 1]


# ===========================================================================
# _build_context_parts
# ===========================================================================

class TestBuildContextParts:
    def _chunk(self, content="text", source="doc.md"):
        return {"content": content, "metadata": {"source_file": source}}

    def test_single_chunk_included(self):
        parts, chars, idx = _build_context_parts([self._chunk()], max_context_chars=1000)
        assert len(parts) == 1

    def test_source_format_without_citations(self):
        parts, _, _ = _build_context_parts([self._chunk("hello", "readme.md")], 1000)
        assert "[Source 1: readme.md]" in parts[0]

    def test_citation_format(self):
        parts, _, _ = _build_context_parts([self._chunk("hello", "readme.md")], 1000, citations=True)
        assert "[1] readme.md" in parts[0]

    def test_budget_exceeded_stops_adding(self):
        chunk = self._chunk("x" * 100)
        parts, _, _ = _build_context_parts([chunk, chunk], max_context_chars=10)
        assert len(parts) == 0

    def test_start_index_respected(self):
        parts, _, next_idx = _build_context_parts([self._chunk()], 1000, start_index=5)
        assert "[Source 5:" in parts[0]
        assert next_idx == 6

    def test_multiple_chunks_numbered_sequentially(self):
        chunks = [self._chunk("a"), self._chunk("b"), self._chunk("c")]
        parts, _, _ = _build_context_parts(chunks, 10000)
        assert "[Source 1:" in parts[0]
        assert "[Source 2:" in parts[1]
        assert "[Source 3:" in parts[2]

    def test_missing_source_defaults_to_unknown(self):
        chunk = {"content": "text", "metadata": {}}
        parts, _, _ = _build_context_parts([chunk], 1000)
        assert "Unknown" in parts[0]


# ===========================================================================
# build_rag_prompt
# ===========================================================================

class TestBuildRagPrompt:
    def _chunk(self, content="answer"):
        return {"content": content, "metadata": {"source_file": "doc.md"}}

    def test_question_in_prompt(self):
        prompt = build_rag_prompt("What is Python?", [self._chunk()])
        assert "What is Python?" in prompt

    def test_source_format_in_prompt(self):
        prompt = build_rag_prompt("Q?", [self._chunk()])
        assert "[Source 1:" in prompt

    def test_empty_chunks(self):
        prompt = build_rag_prompt("Q?", [])
        assert "Q?" in prompt

    def test_returns_string(self):
        assert isinstance(build_rag_prompt("Q?", [self._chunk()]), str)


# ===========================================================================
# build_rag_prompt_with_citations
# ===========================================================================

class TestBuildRagPromptWithCitations:
    def _chunk(self, content="answer"):
        return {"content": content, "metadata": {"source_file": "doc.md"}}

    def test_citation_format_in_prompt(self):
        prompt = build_rag_prompt_with_citations("Q?", [self._chunk()])
        assert "[1]" in prompt

    def test_citation_instruction_in_prompt(self):
        prompt = build_rag_prompt_with_citations("Q?", [self._chunk()])
        assert "citation" in prompt.lower()

    def test_question_in_prompt(self):
        prompt = build_rag_prompt_with_citations("Why?", [self._chunk()])
        assert "Why?" in prompt


# ===========================================================================
# categorize_chunks
# ===========================================================================

class TestCategorizeChunks:
    def _chunk(self, chunk_type=None):
        meta = {}
        if chunk_type:
            meta["chunk_type"] = chunk_type
        return {"content": "x", "metadata": meta}

    def test_code_chunk(self):
        result = categorize_chunks([self._chunk("code")])
        assert len(result["code"]) == 1
        assert len(result["other"]) == 0

    def test_documentation_chunk(self):
        result = categorize_chunks([self._chunk("documentation")])
        assert len(result["documentation"]) == 1

    def test_conversation_chunk(self):
        result = categorize_chunks([self._chunk("conversation")])
        assert len(result["conversation"]) == 1

    def test_unknown_type_goes_to_other(self):
        result = categorize_chunks([self._chunk("random_type")])
        assert len(result["other"]) == 1

    def test_no_chunk_type_goes_to_other(self):
        result = categorize_chunks([self._chunk()])
        assert len(result["other"]) == 1

    def test_empty_returns_empty_categories(self):
        result = categorize_chunks([])
        assert result == {"code": [], "documentation": [], "conversation": [], "other": []}

    def test_mixed_chunks(self):
        chunks = [self._chunk("code"), self._chunk("documentation"), self._chunk("unknown")]
        result = categorize_chunks(chunks)
        assert len(result["code"]) == 1
        assert len(result["documentation"]) == 1
        assert len(result["other"]) == 1


# ===========================================================================
# pool_metadata
# ===========================================================================

class TestReadMetadata:
    def test_returns_empty_dict_when_no_file(self, tmp_path):
        assert read_metadata(tmp_path) == {}

    def test_returns_parsed_json(self, tmp_path):
        path = tmp_path / METADATA_FILENAME
        path.write_text(json.dumps({"created_at": "2024-01-01"}))
        result = read_metadata(tmp_path)
        assert result["created_at"] == "2024-01-01"

    def test_returns_empty_on_invalid_json(self, tmp_path):
        path = tmp_path / METADATA_FILENAME
        path.write_text("not json")
        assert read_metadata(tmp_path) == {}


class TestEnsureCreated:
    def test_creates_metadata_with_created_at(self, tmp_path):
        ensure_created(tmp_path)
        meta = read_metadata(tmp_path)
        assert "created_at" in meta

    def test_idempotent_does_not_overwrite(self, tmp_path):
        ensure_created(tmp_path)
        first_created = read_metadata(tmp_path)["created_at"]
        ensure_created(tmp_path)
        assert read_metadata(tmp_path)["created_at"] == first_created


class TestMarkSearched:
    def test_bumps_search_count(self, tmp_path):
        mark_searched(tmp_path)
        meta = read_metadata(tmp_path)
        assert meta["search_count"] == 1
        mark_searched(tmp_path)
        meta = read_metadata(tmp_path)
        assert meta["search_count"] == 2

    def test_sets_last_searched_at(self, tmp_path):
        mark_searched(tmp_path)
        assert "last_searched_at" in read_metadata(tmp_path)

    def test_noop_when_dir_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        mark_searched(missing)  # should not raise or create the dir
        assert not missing.exists()

    def test_sets_created_at_if_missing(self, tmp_path):
        mark_searched(tmp_path)
        meta = read_metadata(tmp_path)
        assert "created_at" in meta


class TestMarkIndexed:
    def test_sets_last_indexed_at(self, tmp_path):
        mark_indexed(tmp_path)
        assert "last_indexed_at" in read_metadata(tmp_path)

    def test_noop_when_dir_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        mark_indexed(missing)  # should not raise
        assert not missing.exists()

    def test_sets_created_at_if_missing(self, tmp_path):
        mark_indexed(tmp_path)
        assert "created_at" in read_metadata(tmp_path)


class TestDeletePool:
    def test_deletes_pool_directory(self, tmp_path):
        pool = tmp_path / "mypool"
        pool.mkdir()
        (pool / "data.json").write_text("{}")
        delete_pool(pool)
        assert not pool.exists()

    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            delete_pool(tmp_path / "nonexistent")
