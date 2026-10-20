"""Tests for llm_extraction (CitationMap, RAG prompt, categorize) and verifier pure helpers."""


from mullet.knowledge.engine.llm_extraction import (
    CitationMap,
    build_citation_map,
    parse_citations_from_response,
    _build_context_parts,
    build_rag_prompt,
    categorize_chunks,
    build_rag_prompt_with_citations,
)
from mullet.core.orchestration.verifier import (
    CriterionResult,
    VerificationResult,
    extract_criteria,
    _parse_criterion_result,
)


# ---------------------------------------------------------------------------
# CitationMap
# ---------------------------------------------------------------------------

class TestCitationMap:
    def test_len(self):
        cm = CitationMap(chunks=[{"content": "a"}, {"content": "b"}])
        assert len(cm) == 2

    def test_len_empty(self):
        assert len(CitationMap()) == 0

    def test_get_first_chunk(self):
        cm = CitationMap(chunks=[{"content": "first"}, {"content": "second"}])
        assert cm.get(1)["content"] == "first"

    def test_get_last_chunk(self):
        cm = CitationMap(chunks=[{"content": "a"}, {"content": "b"}])
        assert cm.get(2)["content"] == "b"

    def test_get_zero_returns_none(self):
        cm = CitationMap(chunks=[{"content": "a"}])
        assert cm.get(0) is None

    def test_get_beyond_end_returns_none(self):
        cm = CitationMap(chunks=[{"content": "a"}])
        assert cm.get(2) is None

    def test_get_negative_returns_none(self):
        cm = CitationMap(chunks=[{"content": "a"}])
        assert cm.get(-1) is None


# ---------------------------------------------------------------------------
# build_citation_map
# ---------------------------------------------------------------------------

class TestBuildCitationMap:
    def test_returns_citation_map(self):
        cm = build_citation_map([{"content": "text"}])
        assert isinstance(cm, CitationMap)

    def test_chunks_stored(self):
        chunks = [{"content": "a"}, {"content": "b"}]
        cm = build_citation_map(chunks)
        assert len(cm) == 2

    def test_empty_chunks(self):
        cm = build_citation_map([])
        assert len(cm) == 0

    def test_get_via_citation_map(self):
        chunks = [{"content": "chunk text", "metadata": {}}]
        cm = build_citation_map(chunks)
        assert cm.get(1) is not None


# ---------------------------------------------------------------------------
# parse_citations_from_response
# ---------------------------------------------------------------------------

class TestParseCitationsFromResponse:
    def test_returns_list(self):
        assert isinstance(parse_citations_from_response("hello"), list)

    def test_empty_when_no_citations(self):
        assert parse_citations_from_response("no brackets here") == []

    def test_single_citation(self):
        assert parse_citations_from_response("see [1] for details") == [1]

    def test_multiple_citations(self):
        result = parse_citations_from_response("see [1] and [2] and [3]")
        assert result == [1, 2, 3]

    def test_citation_integers(self):
        result = parse_citations_from_response("[5]")
        assert all(isinstance(x, int) for x in result)

    def test_duplicate_citations(self):
        result = parse_citations_from_response("[1] and [1] again")
        assert result == [1, 1]

    def test_empty_string_returns_empty(self):
        assert parse_citations_from_response("") == []


# ---------------------------------------------------------------------------
# _build_context_parts
# ---------------------------------------------------------------------------

def _make_chunks():
    return [
        {"content": "Content A here", "metadata": {"source_file": "a.py"}},
        {"content": "Content B here", "metadata": {"source_file": "b.py"}},
    ]


class TestBuildContextParts:
    def test_returns_tuple_of_three(self):
        result = _build_context_parts(_make_chunks(), 10000)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_parts_is_list(self):
        parts, _, _ = _build_context_parts(_make_chunks(), 10000)
        assert isinstance(parts, list)

    def test_all_chunks_included_when_budget_ample(self):
        parts, _, _ = _build_context_parts(_make_chunks(), 10000)
        assert len(parts) == 2

    def test_truncated_when_budget_small(self):
        parts, _, _ = _build_context_parts(_make_chunks(), 5)
        assert len(parts) == 0

    def test_total_chars_positive(self):
        _, total, _ = _build_context_parts(_make_chunks(), 10000)
        assert total > 0

    def test_idx_increments(self):
        _, _, idx = _build_context_parts(_make_chunks(), 10000, start_index=1)
        assert idx == 3  # started at 1, moved past 2 chunks

    def test_citation_format(self):
        parts, _, _ = _build_context_parts(_make_chunks(), 10000, citations=True)
        assert parts[0].startswith("[1]")

    def test_source_format(self):
        parts, _, _ = _build_context_parts(_make_chunks(), 10000, citations=False)
        assert "[Source 1:" in parts[0]

    def test_source_filename_in_part(self):
        parts, _, _ = _build_context_parts(_make_chunks(), 10000)
        assert "a.py" in parts[0]


# ---------------------------------------------------------------------------
# build_rag_prompt
# ---------------------------------------------------------------------------

class TestBuildRagPrompt:
    def test_returns_string(self):
        assert isinstance(build_rag_prompt("q?", []), str)

    def test_question_in_prompt(self):
        prompt = build_rag_prompt("What is Python?", [])
        assert "What is Python?" in prompt

    def test_chunk_content_in_prompt(self):
        chunks = [{"content": "Python is a language", "metadata": {"source_file": "wiki.md"}}]
        prompt = build_rag_prompt("q?", chunks)
        assert "Python is a language" in prompt

    def test_context_header_present(self):
        prompt = build_rag_prompt("q?", [])
        assert "Context" in prompt

    def test_empty_chunks_still_returns_prompt(self):
        prompt = build_rag_prompt("q?", [])
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# build_rag_prompt_with_citations
# ---------------------------------------------------------------------------

class TestBuildRagPromptWithCitations:
    def test_returns_string(self):
        assert isinstance(build_rag_prompt_with_citations("q?", []), str)

    def test_question_in_prompt(self):
        prompt = build_rag_prompt_with_citations("Define Python", [])
        assert "Define Python" in prompt

    def test_citation_instruction_present(self):
        prompt = build_rag_prompt_with_citations("q?", [])
        assert "[1]" in prompt or "citations" in prompt.lower()

    def test_chunk_numbered_with_bracket(self):
        chunks = [{"content": "text", "metadata": {"source_file": "f.py"}}]
        prompt = build_rag_prompt_with_citations("q?", chunks)
        assert "[1]" in prompt


# ---------------------------------------------------------------------------
# categorize_chunks
# ---------------------------------------------------------------------------

class TestCategorizeChunks:
    def test_returns_dict(self):
        assert isinstance(categorize_chunks([]), dict)

    def test_has_all_category_keys(self):
        result = categorize_chunks([])
        for key in ("code", "documentation", "conversation", "other"):
            assert key in result

    def test_code_chunk_categorized(self):
        chunks = [{"content": "c", "metadata": {"chunk_type": "code"}}]
        result = categorize_chunks(chunks)
        assert len(result["code"]) == 1

    def test_documentation_chunk_categorized(self):
        chunks = [{"content": "d", "metadata": {"chunk_type": "documentation"}}]
        result = categorize_chunks(chunks)
        assert len(result["documentation"]) == 1

    def test_conversation_chunk_categorized(self):
        chunks = [{"content": "c", "metadata": {"chunk_type": "conversation"}}]
        result = categorize_chunks(chunks)
        assert len(result["conversation"]) == 1

    def test_unknown_type_goes_to_other(self):
        chunks = [{"content": "x", "metadata": {"chunk_type": "mystery"}}]
        result = categorize_chunks(chunks)
        assert len(result["other"]) == 1

    def test_no_chunk_type_goes_to_other(self):
        chunks = [{"content": "x", "metadata": {}}]
        result = categorize_chunks(chunks)
        assert len(result["other"]) == 1

    def test_multiple_chunks_categorized(self):
        chunks = [
            {"content": "c", "metadata": {"chunk_type": "code"}},
            {"content": "d", "metadata": {"chunk_type": "documentation"}},
            {"content": "x", "metadata": {"chunk_type": "other_type"}},
        ]
        result = categorize_chunks(chunks)
        assert len(result["code"]) == 1
        assert len(result["documentation"]) == 1
        assert len(result["other"]) == 1


# ---------------------------------------------------------------------------
# CriterionResult
# ---------------------------------------------------------------------------

class TestCriterionResult:
    def test_criterion_stored(self):
        cr = CriterionResult(criterion="user can log in")
        assert cr.criterion == "user can log in"

    def test_passed_defaults_false(self):
        cr = CriterionResult(criterion="test")
        assert cr.passed is False

    def test_evidence_defaults_empty(self):
        cr = CriterionResult(criterion="test")
        assert cr.evidence == ""

    def test_attempt_defaults_one(self):
        cr = CriterionResult(criterion="test")
        assert cr.attempt == 1


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResultSummary:
    def test_all_pass_summary(self):
        vr = VerificationResult(passed=True, criteria_results={"a": True, "b": True}, attempts=1)
        assert "PASS" in vr.summary
        assert "2/2" in vr.summary

    def test_fail_summary(self):
        vr = VerificationResult(passed=False, criteria_results={"a": True, "b": False}, attempts=1)
        assert "FAIL" in vr.summary
        assert "1/2" in vr.summary

    def test_attempts_in_summary(self):
        vr = VerificationResult(passed=True, criteria_results={}, attempts=3)
        assert "3" in vr.summary

    def test_empty_criteria_summary(self):
        vr = VerificationResult(passed=True, criteria_results={}, attempts=1)
        assert "0/0" in vr.summary


class TestVerificationResultFailedCriteria:
    def test_no_failures_empty_list(self):
        vr = VerificationResult(passed=True, criteria_results={"a": True, "b": True})
        assert vr.failed_criteria == []

    def test_failed_criteria_listed(self):
        vr = VerificationResult(passed=False, criteria_results={"a": True, "b": False, "c": False})
        assert set(vr.failed_criteria) == {"b", "c"}

    def test_all_failed(self):
        vr = VerificationResult(passed=False, criteria_results={"a": False})
        assert vr.failed_criteria == ["a"]


# ---------------------------------------------------------------------------
# extract_criteria
# ---------------------------------------------------------------------------

class TestExtractCriteria:
    def test_returns_list(self):
        assert isinstance(extract_criteria(""), list)

    def test_empty_returns_empty(self):
        assert extract_criteria("") == []

    def test_checkbox_unchecked(self):
        result = extract_criteria("- [ ] user can log in")
        assert "user can log in" in result

    def test_checkbox_checked(self):
        result = extract_criteria("- [x] tests pass")
        assert "tests pass" in result

    def test_numbered_item(self):
        result = extract_criteria("1. API works correctly")
        assert "API works correctly" in result

    def test_numbered_item_with_paren(self):
        result = extract_criteria("2) feature is complete")
        assert "feature is complete" in result

    def test_ac_prefix(self):
        result = extract_criteria("AC: authentication required")
        assert "authentication required" in result

    def test_multiple_types_extracted(self):
        text = "- [ ] login works\n1. API responds\nAC: no errors"
        result = extract_criteria(text)
        assert len(result) == 3

    def test_blank_lines_ignored(self):
        result = extract_criteria("- [ ] login\n\n1. API works")
        assert len(result) == 2

    def test_short_bullet_ignored(self):
        result = extract_criteria("- ok")  # only 1 word, below threshold
        assert result == []


# ---------------------------------------------------------------------------
# _parse_criterion_result
# ---------------------------------------------------------------------------

class TestParseCriterionResult:
    def test_returns_tuple_of_two(self):
        result = _parse_criterion_result("CRITERION 1: PASS - done", 1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pass_detected(self):
        passed, _ = _parse_criterion_result("CRITERION 1: PASS - all good", 1)
        assert passed is True

    def test_fail_detected(self):
        passed, _ = _parse_criterion_result("CRITERION 1: FAIL - missing", 1)
        assert passed is False

    def test_evidence_returned(self):
        _, evidence = _parse_criterion_result("CRITERION 1: PASS - tests green", 1)
        assert isinstance(evidence, str)
        assert len(evidence) > 0

    def test_criterion_not_in_response_returns_false(self):
        passed, evidence = _parse_criterion_result("no criterion info here", 3)
        assert passed is False

    def test_all_pass_fallback(self):
        passed, evidence = _parse_criterion_result("All tests pass successfully", 99)
        assert passed is True

    def test_multiple_criteria_parsed_independently(self):
        response = "CRITERION 1: PASS - ok\nCRITERION 2: FAIL - broken"
        p1, _ = _parse_criterion_result(response, 1)
        p2, _ = _parse_criterion_result(response, 2)
        assert p1 is True
        assert p2 is False
