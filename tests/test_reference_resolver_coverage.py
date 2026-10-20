"""Coverage tests for mullet.knowledge.engine.reference_resolver gaps."""

import pytest
from unittest.mock import MagicMock, patch

from mullet.knowledge.engine.reference_resolver import (
    clear_reference_cache,
    detect_references,
    detect_references_in_chunks,
    build_reference_queries,
    _extract_reference_context,
    _extract_semantic_queries,
    is_reference_content,
    _fetch_adjacent_chunks,
    resolve_references,
    resolve_references_in_response,
)


@pytest.fixture(autouse=True)
def reset_reference_cache():
    clear_reference_cache()
    yield
    clear_reference_cache()


# ---------------------------------------------------------------------------
# detect_references – config delegation & uncovered type branches
# ---------------------------------------------------------------------------

class TestDetectReferencesWithConfig:
    def test_delegates_to_config(self):
        mock_config = MagicMock()
        mock_config.detect_references.return_value = [
            {"type": "custom", "id": "X", "key": "custom:X", "match": "Custom X"}
        ]
        result = detect_references("some text", config=mock_config)
        mock_config.detect_references.assert_called_once_with("some text")
        assert result == [{"type": "custom", "id": "X", "key": "custom:X", "match": "Custom X"}]


class TestDetectReferencesLegacyBranches:
    def test_detects_figure(self):
        refs = detect_references("See Figure 3.4.")
        assert any(r["type"] == "figure" for r in refs)

    def test_detects_chapter(self):
        refs = detect_references("See Chapter 2.")
        assert any(r["type"] == "chapter" for r in refs)

    def test_detects_part(self):
        refs = detect_references("See Part A.")
        assert any(r["type"] == "part" for r in refs)

    def test_unknown_type_with_monkeypatched_patterns(self):
        with patch(
            "mullet.knowledge.engine.reference_resolver.REFERENCE_PATTERNS",
            [r"\b(?:see\s+)?Custom\s+([A-Z0-9]+)\b"],
        ):
            refs = detect_references("See Custom X.")
            assert len(refs) == 1
            assert refs[0]["type"] == "unknown"


# ---------------------------------------------------------------------------
# detect_references_in_chunks – config path
# ---------------------------------------------------------------------------

class TestDetectReferencesInChunksWithConfig:
    def test_delegates_to_config(self):
        mock_config = MagicMock()
        mock_config.detect_references.return_value = [
            {"type": "chapter", "id": "1", "key": "chapter:1", "match": "Chapter 1"}
        ]
        chunks = [{"content": "text"}]
        result = detect_references_in_chunks(chunks, config=mock_config)
        assert len(result) == 1
        assert result[0]["type"] == "chapter"

    def test_config_skips_prioritize(self):
        mock_config = MagicMock()
        mock_config.detect_references.return_value = [
            {"type": "chapter", "id": "1", "key": "chapter:1", "match": "Chapter 1"}
        ]
        chunks = [{"content": "text"}]
        with patch(
            "mullet.knowledge.engine.reference_resolver._prioritize_references"
        ) as mock_prio:
            detect_references_in_chunks(chunks, config=mock_config)
            mock_prio.assert_not_called()


# ---------------------------------------------------------------------------
# build_reference_queries – config delegation & uncovered type branches
# ---------------------------------------------------------------------------

class TestBuildReferenceQueriesWithConfig:
    def test_delegates_to_config(self):
        mock_config = MagicMock()
        mock_config.build_queries_for_reference.return_value = ["custom query"]
        ref = {
            "type": "appendix",
            "id": "H",
            "key": "appendix:H",
            "match": "Appendix H",
        }
        result = build_reference_queries([ref], config=mock_config)
        mock_config.build_queries_for_reference.assert_called_once_with(ref)
        assert "custom query" in result


class TestBuildReferenceQueriesLegacyBranches:
    def test_table_query(self):
        refs = [
            {"type": "table", "id": "2.1", "key": "table:2.1", "match": "Table 2.1"}
        ]
        queries = build_reference_queries(refs)
        assert any("Table 2.1" in q for q in queries)

    def test_chapter_query(self):
        refs = [
            {"type": "chapter", "id": "2", "key": "chapter:2", "match": "Chapter 2"}
        ]
        queries = build_reference_queries(refs)
        assert any("Chapter 2" in q for q in queries)

    def test_part_query(self):
        refs = [
            {"type": "part", "id": "A", "key": "part:A", "match": "Part A"}
        ]
        queries = build_reference_queries(refs)
        assert any("Part A" in q for q in queries)

    def test_unknown_type_falls_through(self):
        refs = [
            {"type": "figure", "id": "1", "key": "figure:1", "match": "Figure 1"}
        ]
        queries = build_reference_queries(refs)
        assert any("Figure 1" in q for q in queries)

    def test_context_triggers_semantic_extraction(self):
        with patch(
            "mullet.knowledge.engine.reference_resolver._extract_semantic_queries"
        ) as mock_extract:
            mock_extract.return_value = ["semantic query"]
            refs = [
                {
                    "type": "appendix",
                    "id": "H",
                    "key": "appendix:H",
                    "match": "Appendix H",
                }
            ]
            queries = build_reference_queries(refs, context="some context", config=None)
            mock_extract.assert_called_once_with("some context", None)
            assert "semantic query" in queries

    def test_deduplicates_queries(self):
        refs = [
            {
                "type": "appendix",
                "id": "H",
                "key": "appendix:H",
                "match": "Appendix H",
            },
            {
                "type": "appendix",
                "id": "H",
                "key": "appendix:H",
                "match": "Annex H",
            },
        ]
        queries = build_reference_queries(refs)
        assert len(queries) == len(set(queries))


# ---------------------------------------------------------------------------
# _extract_reference_context
# ---------------------------------------------------------------------------

class TestExtractReferenceContext:
    def test_empty_match_returns_empty(self):
        ref = {"match": ""}
        assert _extract_reference_context([{"content": "foo"}], ref) == ""

    def test_match_found_returns_context(self):
        ref = {"match": "Appendix H"}
        chunks = [{"content": "For more info see Appendix H in the document."}]
        result = _extract_reference_context(chunks, ref)
        assert "Appendix H" in result

    def test_match_not_found_returns_empty(self):
        ref = {"match": "Appendix Z"}
        chunks = [{"content": "No mention here."}]
        assert _extract_reference_context(chunks, ref) == ""

    def test_extracts_context_around_match(self):
        ref = {"match": "Appendix H"}
        content = "A" * 100 + "See Appendix H in the document." + "B" * 100
        chunks = [{"content": content}]
        result = _extract_reference_context(chunks, ref)
        assert "Appendix H" in result
        assert len(result) <= 200 + len("Appendix H")


# ---------------------------------------------------------------------------
# _extract_semantic_queries
# ---------------------------------------------------------------------------

class TestExtractSemanticQueries:
    def test_short_context_returns_empty(self):
        assert _extract_semantic_queries("") == []
        assert _extract_semantic_queries("short") == []

    def test_adjective_noun_extraction(self):
        context = "This is about high risk countries and mandatory requirements."
        result = _extract_semantic_queries(context)
        assert any("high risk" in q for q in result)

    def test_compound_phrases_from_defaults(self):
        context = "high risk countries list due diligence"
        result = _extract_semantic_queries(context)
        assert "high risk countries" in result

    def test_config_provides_compound_phrases(self):
        mock_config = MagicMock()
        mock_config.get_semantic_phrases.return_value = ["custom phrase", "another phrase"]
        context = "custom phrase appears here"
        result = _extract_semantic_queries(context, config=mock_config)
        assert "custom phrase" in result
        mock_config.get_semantic_phrases.assert_called_once()

    def test_config_phrases_no_match(self):
        mock_config = MagicMock()
        mock_config.get_semantic_phrases.return_value = ["nonexistent phrase"]
        context = "totally unrelated text"
        result = _extract_semantic_queries(context, config=mock_config)
        assert result == []

    def test_deduplication_and_limit(self):
        context = "high risk countries high risk countries"
        result = _extract_semantic_queries(context)
        assert len(result) == len(set(result))
        assert len(result) <= 3

    def test_compound_phrase_without_adjective_noun(self):
        context = "due diligence requirements"
        result = _extract_semantic_queries(context)
        assert "due diligence" in result

    def test_adjective_noun_combination(self):
        context = "Enhanced requirements are needed."
        result = _extract_semantic_queries(context)
        assert "enhanced requirements" in result


# ---------------------------------------------------------------------------
# is_reference_content – content pattern match (lines 437-438)
# ---------------------------------------------------------------------------

class TestIsReferenceContentPatterns:
    def test_content_pattern_colon(self):
        chunk = {"content": "Appendix H:\nSome data"}
        ref = {"type": "appendix", "id": "H", "key": "appendix:H"}
        assert is_reference_content(chunk, ref) is True

    def test_content_pattern_dash(self):
        chunk = {"content": "Appendix H - Some data"}
        ref = {"type": "appendix", "id": "H", "key": "appendix:H"}
        assert is_reference_content(chunk, ref) is True

    def test_content_pattern_newline(self):
        chunk = {"content": "Appendix H\nSome data"}
        ref = {"type": "appendix", "id": "H", "key": "appendix:H"}
        assert is_reference_content(chunk, ref) is True

    def test_content_pattern_mid_text(self):
        chunk = {"content": "For details see appendix h - more data"}
        ref = {"type": "appendix", "id": "H", "key": "appendix:H"}
        assert is_reference_content(chunk, ref) is True

    def test_content_pattern_for_section(self):
        chunk = {"content": "For details see section 1: more data"}
        ref = {"type": "section", "id": "1", "key": "section:1"}
        assert is_reference_content(chunk, ref) is True

    def test_metadata_parent_section_match(self):
        chunk = {
            "content": "content here",
            "metadata": {"parent_section": "appendix h"},
        }
        ref = {"type": "appendix", "id": "H", "key": "appendix:H"}
        assert is_reference_content(chunk, ref) is True

    def test_metadata_source_file_underscore_match(self):
        chunk = {
            "content": "content here",
            "metadata": {"source_file": "appendix_h.pdf"},
        }
        ref = {"type": "appendix", "id": "H", "key": "appendix:H"}
        assert is_reference_content(chunk, ref) is True


# ---------------------------------------------------------------------------
# _fetch_adjacent_chunks
# ---------------------------------------------------------------------------

class TestFetchAdjacentChunks:
    def test_no_found_chunks_returns_empty(self):
        searcher = MagicMock()
        assert _fetch_adjacent_chunks([], searcher, set()) == []

    def test_missing_bm25_chunks_returns_empty(self):
        searcher = MagicMock()
        assert _fetch_adjacent_chunks([{"content": "c"}], searcher, set()) == []

    def test_empty_bm25_chunks_returns_empty(self):
        searcher = MagicMock()
        searcher.bm25_chunks = []
        assert _fetch_adjacent_chunks([{"content": "c"}], searcher, set()) == []

    def test_skips_source_not_in_all_chunks(self):
        searcher = MagicMock()
        found = [{"content": "a", "metadata": {"source_file": "doc1", "chunk_index": 0}}]
        searcher.bm25_chunks = []
        assert _fetch_adjacent_chunks(found, searcher, set()) == []

    def test_finds_adjacent_chunks(self):
        searcher = MagicMock()
        chunks = [
            {"content": "a", "metadata": {"source_file": "doc1", "chunk_index": 0}},
            {"content": "b", "metadata": {"source_file": "doc1", "chunk_index": 1}},
            {"content": "c", "metadata": {"source_file": "doc1", "chunk_index": 2}},
            {"content": "d", "metadata": {"source_file": "doc1", "chunk_index": 3}},
        ]
        searcher.bm25_chunks = chunks
        found = [chunks[1]]  # index 1
        existing = set()
        result = _fetch_adjacent_chunks(found, searcher, existing, max_adjacent=5)
        result_indices = [c["metadata"]["chunk_index"] for c in result]
        assert 0 in result_indices
        assert 2 in result_indices
        assert 3 in result_indices

    def test_includes_chunks_within_two_positions(self):
        searcher = MagicMock()
        chunks = [
            {"content": str(i), "metadata": {"source_file": "doc1", "chunk_index": i}}
            for i in range(6)
        ]
        searcher.bm25_chunks = chunks
        found = [chunks[2]]  # index 2
        result = _fetch_adjacent_chunks(found, searcher, set())
        result_indices = [c["metadata"]["chunk_index"] for c in result]
        assert 0 in result_indices
        assert 1 in result_indices
        assert 3 in result_indices
        assert 4 in result_indices
        assert 5 not in result_indices

    def test_respects_existing_ids(self):
        searcher = MagicMock()
        chunks = [
            {"content": "a", "metadata": {"source_file": "doc1", "chunk_index": 0}},
            {"content": "b", "metadata": {"source_file": "doc1", "chunk_index": 1}},
        ]
        searcher.bm25_chunks = chunks
        found = [chunks[1]]
        existing = {hash("a"[:200])}
        result = _fetch_adjacent_chunks(found, searcher, existing, max_adjacent=5)
        assert all(hash(c.get("content", "")[:200]) not in existing for c in result)

    def test_respects_max_adjacent(self):
        searcher = MagicMock()
        chunks = [
            {"content": str(i), "metadata": {"source_file": "doc1", "chunk_index": i}}
            for i in range(10)
        ]
        searcher.bm25_chunks = chunks
        found = [chunks[5]]
        existing = set()
        result = _fetch_adjacent_chunks(found, searcher, existing, max_adjacent=2)
        assert len(result) <= 2

    def test_skips_negative_chunk_index(self):
        searcher = MagicMock()
        chunks = [
            {"content": "a", "metadata": {"source_file": "doc1", "chunk_index": -1}},
            {"content": "b", "metadata": {"source_file": "doc1", "chunk_index": 1}},
        ]
        searcher.bm25_chunks = chunks
        found = [chunks[1]]
        result = _fetch_adjacent_chunks(found, searcher, set())
        assert len(result) == 0

    def test_found_with_no_valid_index_skips(self):
        searcher = MagicMock()
        found = [{"content": "a", "metadata": {"source_file": "doc1"}}]
        searcher.bm25_chunks = [
            {"content": "b", "metadata": {"source_file": "doc1", "chunk_index": 0}}
        ]
        result = _fetch_adjacent_chunks(found, searcher, set())
        assert result == []


# ---------------------------------------------------------------------------
# resolve_references – main multi-hop resolution logic
# ---------------------------------------------------------------------------

class TestResolveReferences:
    def test_config_disabled_returns_original(self):
        mock_config = MagicMock()
        mock_config.enabled = False
        chunks = [{"content": "test"}]
        result, log = resolve_references(chunks, MagicMock(), config=mock_config)
        assert result is chunks
        assert any("disabled" in entry for entry in log)

    def test_no_refs_returns_original(self):
        chunks = [{"content": "plain text with no references"}]
        result, log = resolve_references(chunks, MagicMock())
        assert result == chunks
        assert any("No references detected" in entry for entry in log)

    def test_scan_top_n_chunks_logs_subset(self):
        chunks = [{"content": "See Appendix H."}, {"content": "plain"}]
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = []
        result, log = resolve_references(
            chunks, searcher, scan_top_n_chunks=1, use_cache=False
        )
        assert any("Scanned top 1 chunks" in entry for entry in log)

    def test_scan_all_chunks_when_scan_top_n_none(self):
        chunks = [
            {"content": "See Appendix H."},
            {"content": "See Appendix Z."},
        ]
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = []
        result, log = resolve_references(
            chunks, searcher, scan_top_n_chunks=None, use_cache=False
        )
        assert any("Detected 2 references" in entry for entry in log)

    def test_limits_refs_to_resolve(self):
        text = (
            "See Appendix A. See Appendix B. See Appendix C. "
            "See Appendix D. See Appendix E. See Appendix F."
        )
        chunks = [{"content": text}]
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = []
        result, log = resolve_references(
            chunks, searcher, max_refs_to_resolve=3, use_cache=False
        )
        assert any("Detected 6 references" in entry for entry in log)
        resolve_logs = [
            entry for entry in log if "Could not resolve" in entry or "Resolved" in entry
        ]
        assert len(resolve_logs) <= 3

    def test_cache_hit_adds_cached_chunks(self):
        from mullet.knowledge.engine.reference_resolver import _reference_cache

        ref_key = "appendix:H"
        cached_chunk = {"content": "cached data", "metadata": {"source_file": "s"}}
        _reference_cache[ref_key] = [cached_chunk]
        chunks = [{"content": "See Appendix H."}]
        searcher = MagicMock(spec=["search"])
        result, log = resolve_references(chunks, searcher, use_cache=True)
        assert any("Cache hit" in entry for entry in log)
        contents = [c["content"] for c in result]
        assert "cached data" in contents

    def test_cache_hit_skips_existing_ids(self):
        from mullet.knowledge.engine.reference_resolver import _reference_cache

        content = "Appendix H:\nData"
        ref_key = "appendix:H"
        _reference_cache[ref_key] = [{"content": content, "metadata": {"source_file": "s"}}]
        chunks = [{"content": content}]
        searcher = MagicMock(spec=["search"])
        result, log = resolve_references(chunks, searcher, use_cache=True)
        assert len([c for c in result if c["content"] == content]) == 1

    def test_use_cache_false_ignores_cache(self):
        from mullet.knowledge.engine.reference_resolver import _reference_cache

        ref_key = "appendix:H"
        _reference_cache[ref_key] = [{"content": "cached", "metadata": {"source_file": "s"}}]
        chunks = [{"content": "See Appendix H."}]
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = []
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert not any("Cache hit" in entry for entry in log)

    def test_cache_hit_then_miss(self):
        from mullet.knowledge.engine.reference_resolver import _reference_cache

        _reference_cache["appendix:A"] = [
            {"content": "cached A", "metadata": {"source_file": "s"}}
        ]
        chunks = [{"content": "See Appendix A and Appendix B."}]
        ref_chunk = {
            "content": "Appendix B:\nData",
            "metadata": {"source_file": "doc.pdf"},
        }
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = [ref_chunk]
        result, log = resolve_references(chunks, searcher, use_cache=True)
        assert any("Cache hit" in entry for entry in log)
        assert any("Resolved Appendix B" in entry for entry in log)

    def test_search_finds_reference_content(self):
        chunks = [{"content": "See Appendix H for details."}]
        ref_chunk = {
            "content": "Appendix H:\nData here",
            "metadata": {"source_file": "doc.pdf"},
        }
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = [ref_chunk]
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any("Resolved See Appendix H" in entry for entry in log)
        assert any(c.get("_resolved_reference") == "appendix:H" for c in result)

    def test_logs_no_results_for_query(self):
        chunks = [{"content": "See Appendix H."}]
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = []
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any("No results for query" in entry for entry in log)

    def test_semantic_query_includes_high_scoring_chunk(self):
        chunks = [{"content": "See Appendix H for high risk countries."}]
        semantic_chunk = {
            "content": "List of high risk countries",
            "metadata": {"source_file": "doc.pdf"},
            "_combined_score": 0.6,
        }

        def _fake_search(query, top_k):
            if "Appendix H" in query or "Annex H" in query:
                return []
            return [semantic_chunk]

        searcher = MagicMock(spec=["search"])
        searcher.search.side_effect = _fake_search
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any(c.get("_semantic_match") for c in result)

    def test_semantic_query_includes_high_combined_score(self):
        chunks = [{"content": "See Appendix H for high risk countries."}]
        semantic_chunk = {
            "content": "country data",
            "metadata": {"source_file": "doc.pdf"},
            "_combined_score": 0.8,
        }
        filler = {"content": "filler", "metadata": {"source_file": "doc.pdf"}}

        def _fake_search(query, top_k):
            if "Appendix H" in query or "Annex H" in query:
                return []
            return [filler] * 5 + [semantic_chunk]

        searcher = MagicMock(spec=["search"])
        searcher.search.side_effect = _fake_search
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any(c.get("_semantic_match") for c in result)

    def test_content_mention_includes_chunk(self):
        chunks = [{"content": "See Appendix H."}]
        mention_chunk = {
            "content": "Appendix H\nThis is the content.",
            "metadata": {"source_file": "doc.pdf"},
        }
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = [mention_chunk]
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any(c.get("_resolved_reference") == "appendix:H" for c in result)

    def test_content_mention_fallback_for_table(self):
        chunks = [{"content": "See Table 1."}]
        mention_chunk = {
            "content": "Table 1 is important",
            "metadata": {"source_file": "doc.pdf"},
        }
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = [mention_chunk]
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any(c.get("_resolved_reference") == "table:1" for c in result)

    def test_search_exception_logged(self):
        chunks = [{"content": "See Appendix H."}]
        searcher = MagicMock(spec=["search"])
        searcher.search.side_effect = RuntimeError("search failed")
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any("Error resolving" in entry for entry in log)

    def test_uses_fast_search_when_available(self):
        chunks = [{"content": "See Appendix H."}]
        ref_chunk = {
            "content": "Appendix H:\nData",
            "metadata": {"source_file": "doc.pdf"},
        }
        searcher = MagicMock()
        searcher.fast_search.return_value = [ref_chunk]
        result, log = resolve_references(chunks, searcher, use_cache=False)
        searcher.fast_search.assert_called()
        assert any(c.get("_resolved_reference") == "appendix:H" for c in result)

    def test_appendix_fetches_adjacent_chunks(self):
        chunks = [{"content": "See Appendix H."}]
        ref_chunk = {
            "content": "Appendix H:\nData",
            "metadata": {"source_file": "doc1", "chunk_index": 1},
        }
        adjacent_chunk = {
            "content": "More data",
            "metadata": {"source_file": "doc1", "chunk_index": 2},
        }
        searcher = MagicMock(spec=["search", "bm25_chunks"])
        searcher.search.return_value = [ref_chunk]
        searcher.bm25_chunks = [
            {
                "content": "Other",
                "metadata": {"source_file": "doc1", "chunk_index": 0},
            },
            ref_chunk,
            adjacent_chunk,
        ]
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any(c.get("_adjacent_to_reference") for c in result)

    def test_no_new_chunks_returns_original(self):
        chunks = [{"content": "See Appendix H."}]
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = []
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert result == chunks
        assert any(
            "Could not find content for any references" in entry for entry in log
        )

    def test_deduplicates_original_and_resolved(self):
        content = "Appendix H:\nData"
        original = [{"content": content, "metadata": {"source_file": "doc.pdf"}}]
        ref_chunk = {"content": content, "metadata": {"source_file": "doc.pdf"}}
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = [ref_chunk]
        result, log = resolve_references(original, searcher, use_cache=False)
        assert len([c for c in result if c["content"] == content]) == 1

    def test_prioritizes_spreadsheet_reference_data(self):
        original = [
            {
                "content": "See Appendix A for high risk countries.",
                "metadata": {"source_file": "orig.pdf"},
            }
        ]
        chunk_priority = {
            "content": "Appendix A:\nData",
            "metadata": {"source_file": "ref_list.xlsx"},
        }
        chunk_semantic_0 = {
            "content": "country data",
            "metadata": {"source_file": "countries_data.xlsx"},
            "_combined_score": 0.6,
        }
        chunk_semantic_3 = {
            "content": "form sheet",
            "metadata": {"source_file": "assessment_form.xlsx"},
            "_combined_score": 0.6,
        }
        chunk_semantic_5 = {
            "content": "pdf doc",
            "metadata": {"source_file": "manual.pdf"},
            "_combined_score": 0.6,
        }
        chunk_semantic_6 = {
            "content": "plain text",
            "metadata": {"source_file": "notes.txt"},
            "_combined_score": 0.6,
        }
        semantic_results = [
            chunk_priority,
            chunk_semantic_0,
            chunk_semantic_3,
            chunk_semantic_5,
            chunk_semantic_6,
        ]

        def _fake_search(query, top_k):
            if "Appendix A" in query or "Annex A" in query:
                return []
            return semantic_results

        searcher = MagicMock(spec=["search"])
        searcher.search.side_effect = _fake_search
        result, log = resolve_references(original, searcher, use_cache=False)
        sources = [c["metadata"]["source_file"] for c in result]
        # Tier 0 spreadsheets first (sorted by descending score)
        assert sources[0] == "countries_data.xlsx"
        assert sources[1] == "ref_list.xlsx"
        # Tier 3 next
        assert sources[2] == "assessment_form.xlsx"
        # Then pdf, then txt
        pdf_idx = sources.index("manual.pdf")
        txt_idx = sources.index("notes.txt")
        assert pdf_idx < txt_idx
        assert any("Data sources prioritized" in entry for entry in log)

    def test_tier_1_other_spreadsheet(self):
        chunks = [{"content": "See Appendix H."}]
        ref_chunk = {
            "content": "Appendix H:\nData",
            "metadata": {"source_file": "report.xlsx"},
        }
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = [ref_chunk]
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any(c["metadata"]["source_file"] == "report.xlsx" for c in result)

    def test_tier_2_non_spreadsheet_reference(self):
        chunks = [{"content": "See Appendix H."}]
        ref_chunk = {
            "content": "Appendix H:\nData",
            "metadata": {"source_file": "countries_reference.txt"},
        }
        searcher = MagicMock(spec=["search"])
        searcher.search.return_value = [ref_chunk]
        result, log = resolve_references(chunks, searcher, use_cache=False)
        assert any(
            c["metadata"]["source_file"] == "countries_reference.txt" for c in result
        )

    def test_config_passed_through(self):
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.detect_references.return_value = []
        chunks = [{"content": "See Appendix H."}]
        searcher = MagicMock(spec=["search"])
        result, log = resolve_references(
            chunks, searcher, config=mock_config
        )
        mock_config.detect_references.assert_called()


# ---------------------------------------------------------------------------
# resolve_references_in_response
# ---------------------------------------------------------------------------

class TestResolveReferencesInResponse:
    def test_no_refs_returns_empty(self):
        result, log = resolve_references_in_response("no refs", MagicMock())
        assert result == []
        assert log == []

    def test_finds_reference_content(self):
        response = "See Appendix H for details."
        ref_chunk = {
            "content": "Appendix H:\nData",
            "metadata": {"source_file": "doc.pdf"},
        }
        searcher = MagicMock()
        searcher.search.return_value = [ref_chunk]
        result, log = resolve_references_in_response(response, searcher)
        assert len(result) >= 1
        assert any(
            "Found See Appendix H content" in entry for entry in log
        )

    def test_search_exception_silenced(self):
        response = "See Appendix H."
        searcher = MagicMock()
        searcher.search.side_effect = RuntimeError("fail")
        result, log = resolve_references_in_response(response, searcher)
        assert result == []
        assert any("Error:" in entry for entry in log)

    def test_limits_to_three_refs(self):
        response = "See Appendix A. See Appendix B. See Appendix C. See Appendix D."
        searcher = MagicMock()
        searcher.search.return_value = []
        result, log = resolve_references_in_response(response, searcher)
        assert searcher.search.call_count <= 3
