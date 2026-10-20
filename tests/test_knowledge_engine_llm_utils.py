"""Tests for pure utility functions in mullet/knowledge/engine/llm.py."""
import time
import requests
import pytest
from unittest.mock import MagicMock, patch, call

from mullet.knowledge.engine.llm import (
    is_llm_error,
    _retry_with_backoff,
    CitationMap,
    build_citation_map,
    parse_citations_from_response,
    build_rag_prompt_with_citations,
    categorize_chunks,
    allocate_budgets,
    _build_context_parts,
    build_rag_prompt,
)


# ---------------------------------------------------------------------------
# is_llm_error
# ---------------------------------------------------------------------------

class TestIsLlmError:
    def test_returns_true_for_error_prefix(self):
        assert is_llm_error("Error: something went wrong") is True

    def test_returns_false_for_normal_response(self):
        assert is_llm_error("The answer is 42") is False

    def test_returns_false_for_empty(self):
        assert is_llm_error("") is False

    def test_case_sensitive(self):
        assert is_llm_error("error: lowercase") is False


# ---------------------------------------------------------------------------
# _retry_with_backoff
# ---------------------------------------------------------------------------

class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self):
        func = MagicMock(return_value="ok")
        result = _retry_with_backoff(func, max_retries=2)
        assert result == "ok"
        func.assert_called_once()

    def test_retries_on_connection_error(self):
        func = MagicMock(side_effect=[requests.ConnectionError("conn"), "ok"])
        with patch("time.sleep"):
            result = _retry_with_backoff(func, max_retries=1, base_delay=0.01)
        assert result == "ok"
        assert func.call_count == 2

    def test_raises_after_max_retries(self):
        func = MagicMock(side_effect=requests.ConnectionError("always fails"))
        with patch("time.sleep"), pytest.raises(requests.ConnectionError):
            _retry_with_backoff(func, max_retries=2, base_delay=0.01)
        assert func.call_count == 3  # initial + 2 retries

    def test_non_retryable_exception_propagates_immediately(self):
        func = MagicMock(side_effect=ValueError("not retryable"))
        with pytest.raises(ValueError):
            _retry_with_backoff(func, max_retries=2)
        func.assert_called_once()

    def test_exponential_backoff_delays(self):
        func = MagicMock(side_effect=[requests.Timeout("t"), requests.Timeout("t"), "ok"])
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            _retry_with_backoff(func, max_retries=2, base_delay=1.0)
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# CitationMap
# ---------------------------------------------------------------------------

class TestCitationMap:
    def test_get_returns_entry(self):
        cm = CitationMap(entries={1: {"source": "doc.md"}})
        assert cm.get(1) == {"source": "doc.md"}

    def test_get_returns_none_for_missing(self):
        cm = CitationMap()
        assert cm.get(99) is None

    def test_len(self):
        cm = CitationMap(entries={1: {}, 2: {}})
        assert len(cm) == 2

    def test_len_empty(self):
        assert len(CitationMap()) == 0


# ---------------------------------------------------------------------------
# build_citation_map
# ---------------------------------------------------------------------------

class TestBuildCitationMap:
    def _chunk(self, source="doc.md", content="hello", page=None, section=None, score=0.9):
        return {
            "metadata": {"source_file": source, "page_number": page, "section": section},
            "content": content,
            "_combined_score": score,
        }

    def test_empty_chunks_returns_empty_map(self):
        cm = build_citation_map([])
        assert len(cm) == 0

    def test_single_chunk_numbered_from_1(self):
        cm = build_citation_map([self._chunk()])
        assert 1 in cm.entries

    def test_source_file_extracted(self):
        cm = build_citation_map([self._chunk(source="readme.md")])
        assert cm.entries[1]["source"] == "readme.md"

    def test_missing_source_file_defaults_to_unknown(self):
        cm = build_citation_map([{"metadata": {}, "content": "x", "_combined_score": 0}])
        assert cm.entries[1]["source"] == "Unknown"

    def test_rerank_score_preferred_over_combined(self):
        chunk = self._chunk()
        chunk["_rerank_score"] = 0.99
        cm = build_citation_map([chunk])
        assert cm.entries[1]["score"] == 0.99

    def test_combined_score_used_when_no_rerank(self):
        cm = build_citation_map([self._chunk(score=0.75)])
        assert cm.entries[1]["score"] == 0.75

    def test_content_preview_truncated_to_150(self):
        long_content = "x" * 200
        cm = build_citation_map([{"metadata": {}, "content": long_content, "_combined_score": 0}])
        assert len(cm.entries[1]["content_preview"]) == 150

    def test_multiple_chunks_numbered_sequentially(self):
        chunks = [self._chunk("a.md"), self._chunk("b.md"), self._chunk("c.md")]
        cm = build_citation_map(chunks)
        assert {1, 2, 3} == set(cm.entries.keys())

    def test_page_and_section_stored(self):
        chunk = self._chunk(page=5, section="Intro")
        cm = build_citation_map([chunk])
        assert cm.entries[1]["page"] == 5
        assert cm.entries[1]["section"] == "Intro"


# ---------------------------------------------------------------------------
# parse_citations_from_response
# ---------------------------------------------------------------------------

class TestParseCitationsFromResponse:
    def _cm(self, *sources):
        cm = CitationMap()
        for i, s in enumerate(sources, 1):
            cm.entries[i] = {"source": s, "citation_number": None}
        return cm

    def test_no_citations_returns_empty(self):
        cm = self._cm("a.md")
        assert parse_citations_from_response("No citations here.", cm) == []

    def test_single_citation_extracted(self):
        cm = self._cm("a.md")
        result = parse_citations_from_response("See [1] for details.", cm)
        assert len(result) == 1
        assert result[0]["source"] == "a.md"
        assert result[0]["citation_number"] == 1

    def test_duplicate_citations_deduplicated(self):
        cm = self._cm("a.md")
        result = parse_citations_from_response("[1] and [1] again.", cm)
        assert len(result) == 1

    def test_out_of_range_citation_ignored(self):
        cm = self._cm("a.md")  # only entry 1 exists
        result = parse_citations_from_response("See [99].", cm)
        assert result == []

    def test_multiple_citations_ordered_by_appearance(self):
        cm = self._cm("a.md", "b.md", "c.md")
        result = parse_citations_from_response("First [2], then [1], then [3].", cm)
        assert [r["citation_number"] for r in result] == [2, 1, 3]

    def test_citation_entry_is_copy(self):
        cm = self._cm("a.md")
        result = parse_citations_from_response("[1]", cm)
        result[0]["source"] = "modified"
        assert cm.entries[1]["source"] == "a.md"  # original unchanged


# ---------------------------------------------------------------------------
# categorize_chunks
# ---------------------------------------------------------------------------

class TestCategorizeChunks:
    def _chunk(self, chunk_type="document", resolved=False, cross_ref=False, content="x"):
        c = {"metadata": {"chunk_type": chunk_type}, "content": content}
        if resolved:
            c["_resolved_reference"] = True
        if cross_ref:
            c["_cross_referenced"] = True
        return c

    def test_doc_chunk_goes_to_doc(self):
        result = categorize_chunks([self._chunk("document")])
        assert len(result["doc"]) == 1

    def test_code_chunk_goes_to_code(self):
        result = categorize_chunks([self._chunk("code")])
        assert len(result["code"]) == 1

    def test_resolved_reference_goes_to_reference(self):
        result = categorize_chunks([self._chunk(resolved=True)])
        assert len(result["reference"]) == 1

    def test_cross_referenced_goes_to_reference(self):
        result = categorize_chunks([self._chunk(cross_ref=True)])
        assert len(result["reference"]) == 1

    def test_empty_returns_empty_categories(self):
        result = categorize_chunks([])
        assert result == {"doc": [], "code": [], "reference": []}

    def test_mixed_chunks_categorized_correctly(self):
        chunks = [self._chunk("document"), self._chunk("code"), self._chunk(resolved=True)]
        result = categorize_chunks(chunks)
        assert len(result["doc"]) == 1
        assert len(result["code"]) == 1
        assert len(result["reference"]) == 1


# ---------------------------------------------------------------------------
# allocate_budgets
# ---------------------------------------------------------------------------

class TestAllocateBudgets:
    def test_all_categories_present(self):
        b = allocate_budgets(1000, counts={"doc": 2, "code": 1, "reference": 1})
        assert b["doc"] + b["code"] + b["reference"] <= 1000

    def test_empty_category_gets_zero_budget(self):
        b = allocate_budgets(1000, counts={"doc": 1, "code": 0, "reference": 0})
        assert b["code"] == 0
        assert b["reference"] == 0

    def test_no_active_categories_returns_total_to_doc(self):
        b = allocate_budgets(1000, counts={"doc": 0, "code": 0, "reference": 0})
        assert b["doc"] == 1000
        assert b["code"] == 0
        assert b["reference"] == 0

    def test_default_counts_distributes_proportionally(self):
        b = allocate_budgets(1000)  # default counts all 1
        assert b["doc"] > 0
        assert b["code"] > 0
        assert b["reference"] > 0

    def test_budget_proportional_to_ratios(self):
        b = allocate_budgets(1000, doc_ratio=0.8, code_ratio=0.1, ref_ratio=0.1,
                             counts={"doc": 1, "code": 1, "reference": 1})
        assert b["doc"] > b["code"]
        assert b["doc"] > b["reference"]


# ---------------------------------------------------------------------------
# _build_context_parts
# ---------------------------------------------------------------------------

class TestBuildContextParts:
    def _chunk(self, content, source="src.md"):
        return {"content": content, "metadata": {"source_file": source}}

    def test_single_chunk_included(self):
        parts, chars, idx = _build_context_parts([self._chunk("hello")], max_chars=1000)
        assert len(parts) == 1

    def test_part_format_without_citations(self):
        parts, _, _ = _build_context_parts([self._chunk("content", "doc.md")], max_chars=1000)
        assert "[Source 1: doc.md]" in parts[0]
        assert "content" in parts[0]

    def test_part_format_with_citations(self):
        parts, _, _ = _build_context_parts([self._chunk("content", "doc.md")], max_chars=1000, citations=True)
        assert "[1] doc.md" in parts[0]

    def test_budget_exceeded_stops_adding(self):
        chunks = [self._chunk("x" * 100), self._chunk("y" * 100)]
        parts, _, _ = _build_context_parts(chunks, max_chars=10)
        assert len(parts) == 0  # even first chunk exceeds budget

    def test_start_index_respected(self):
        parts, _, next_idx = _build_context_parts([self._chunk("c")], max_chars=1000, start_index=5)
        assert "[Source 5:" in parts[0]
        assert next_idx == 6

    def test_returns_correct_next_index(self):
        chunks = [self._chunk("a"), self._chunk("b")]
        _, _, next_idx = _build_context_parts(chunks, max_chars=10000, start_index=1)
        assert next_idx == 3

    def test_total_chars_returned(self):
        chunk = self._chunk("hello")
        _, chars, _ = _build_context_parts([chunk], max_chars=1000)
        assert chars > 0


# ---------------------------------------------------------------------------
# build_rag_prompt
# ---------------------------------------------------------------------------

class TestBuildRagPrompt:
    def _chunk(self, content="answer", score=0.9, source="doc.md", chunk_type="document"):
        return {
            "content": content,
            "metadata": {"source_file": source, "chunk_type": chunk_type},
            "_combined_score": score,
        }

    def test_returns_list_with_user_message(self):
        result = build_rag_prompt("What?", [self._chunk()])
        assert isinstance(result, list)
        assert result[0]["role"] == "user"

    def test_question_in_message(self):
        result = build_rag_prompt("What is Python?", [self._chunk()])
        assert "What is Python?" in result[0]["content"]

    def test_empty_chunks(self):
        result = build_rag_prompt("Q?", [])
        assert len(result) == 1

    def test_best_at_end_reverses_order(self):
        chunks = [self._chunk("low", score=0.1), self._chunk("high", score=0.9)]
        result = build_rag_prompt("Q?", chunks, best_at_end=True)
        content = result[0]["content"]
        # lower score comes before higher score (sorted ascending → high is last)
        assert content.index("low") < content.index("high")

    def test_citation_format(self):
        result = build_rag_prompt("Q?", [self._chunk()], citations=True)
        assert "[1]" in result[0]["content"]

    def test_source_format_without_citations(self):
        result = build_rag_prompt("Q?", [self._chunk()], citations=False)
        assert "[Source 1:" in result[0]["content"]

    def test_truncation_logged(self, caplog):
        import logging
        # Many chunks but tiny budget — should trigger warning
        chunks = [self._chunk(content="x" * 50) for _ in range(10)]
        with caplog.at_level(logging.WARNING):
            build_rag_prompt("Q?", chunks, max_context_chars=10)
        assert "truncated" in caplog.text.lower()

    def test_budget_ratios_categorize_chunks(self):
        chunks = [
            self._chunk(content="doc content", chunk_type="document"),
            self._chunk(content="code content", chunk_type="code"),
        ]
        result = build_rag_prompt("Q?", chunks,
                                  doc_budget_ratio=0.6, code_budget_ratio=0.3, reference_budget_ratio=0.1)
        content = result[0]["content"]
        assert "doc content" in content or "code content" in content

    def test_partial_budget_ratios_ignored(self):
        # Only doc_budget_ratio provided — should fall back to non-budgeted path
        chunks = [self._chunk()]
        result = build_rag_prompt("Q?", chunks, doc_budget_ratio=0.6)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# build_rag_prompt_with_citations
# ---------------------------------------------------------------------------

class TestBuildRagPromptWithCitations:
    def test_returns_tuple_of_messages_and_citation_map(self):
        chunks = [{"content": "hello", "metadata": {"source_file": "a.md"}, "_combined_score": 0.9}]
        messages, cmap = build_rag_prompt_with_citations("Q?", chunks)
        assert isinstance(messages, list)
        assert isinstance(cmap, CitationMap)

    def test_citation_map_has_entries(self):
        chunks = [{"content": "c", "metadata": {}, "_combined_score": 0}]
        _, cmap = build_rag_prompt_with_citations("Q?", chunks)
        assert len(cmap) == 1

    def test_messages_use_citation_format(self):
        chunks = [{"content": "c", "metadata": {"source_file": "x.md"}, "_combined_score": 0}]
        messages, _ = build_rag_prompt_with_citations("Q?", chunks)
        assert "[1]" in messages[0]["content"]
