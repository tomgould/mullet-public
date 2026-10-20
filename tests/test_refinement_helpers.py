"""Tests for refinement pure helpers: needs_contextualization, combine_questions_for_search, get_quality_metrics."""

import pytest

from mullet.knowledge.engine.refinement import (
    needs_contextualization,
    combine_questions_for_search,
    get_quality_metrics,
    REFINEMENT_SYSTEM_PROMPT,
    FOLLOWUP_INDICATORS,
    STANDALONE_INDICATORS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestRefinementConstants:
    def test_system_prompt_is_string(self):
        assert isinstance(REFINEMENT_SYSTEM_PROMPT, str)

    def test_system_prompt_nonempty(self):
        assert len(REFINEMENT_SYSTEM_PROMPT) > 100

    def test_followup_indicators_nonempty(self):
        assert len(FOLLOWUP_INDICATORS) > 0

    def test_standalone_indicators_nonempty(self):
        assert len(STANDALONE_INDICATORS) > 0

    def test_followup_indicators_list(self):
        assert isinstance(FOLLOWUP_INDICATORS, list)

    def test_standalone_indicators_list(self):
        assert isinstance(STANDALONE_INDICATORS, list)


# ---------------------------------------------------------------------------
# needs_contextualization
# ---------------------------------------------------------------------------

class TestNeedsContextualization:
    def test_returns_bool(self):
        assert isinstance(needs_contextualization("hello"), bool)

    def test_very_short_question_needs_context(self):
        assert needs_contextualization("ok") is True

    def test_four_words_needs_context(self):
        assert needs_contextualization("what about that thing") is True

    def test_standalone_what_is_no_context(self):
        assert needs_contextualization("What is the authentication process?") is False

    def test_standalone_how_does_no_context(self):
        assert needs_contextualization("How does the authentication system work?") is False

    def test_followup_indicator_needs_context(self):
        assert needs_contextualization("Can you explain that in more detail?") is True

    def test_explain_further_needs_context(self):
        assert needs_contextualization("Please explain further about the topic") is True

    def test_tell_me_more_needs_context(self):
        assert needs_contextualization("Tell me more about it") is True

    def test_short_3_words_needs_context(self):
        assert needs_contextualization("what is it") is True

    def test_longer_question_no_context(self):
        assert needs_contextualization("The feature needs to be implemented correctly everywhere") is False


# ---------------------------------------------------------------------------
# combine_questions_for_search
# ---------------------------------------------------------------------------

class TestCombineQuestionsForSearch:
    def test_returns_string(self):
        assert isinstance(combine_questions_for_search([]), str)

    def test_empty_everything_returns_empty(self):
        assert combine_questions_for_search([], topic="", original_question="") == ""

    def test_topic_in_output(self):
        result = combine_questions_for_search([], topic="Authentication", original_question="")
        assert "Authentication" in result

    def test_topic_as_heading(self):
        result = combine_questions_for_search([], topic="Auth", original_question="")
        assert "## Auth" in result

    def test_original_question_in_output(self):
        result = combine_questions_for_search([], topic="", original_question="What is auth?")
        assert "What is auth?" in result

    def test_questions_in_output(self):
        result = combine_questions_for_search(["q1", "q2"], topic="", original_question="")
        assert "q1" in result
        assert "q2" in result

    def test_original_not_duplicated_in_questions(self):
        result = combine_questions_for_search(["q0", "q1"], topic="T", original_question="q0")
        count = result.count("q0")
        assert count == 1

    def test_full_output_structure(self):
        result = combine_questions_for_search(["alt1", "alt2"], topic="Auth", original_question="original")
        assert "## Auth" in result
        assert "original" in result
        assert "alt1" in result
        assert "alt2" in result

    def test_no_topic_no_heading(self):
        result = combine_questions_for_search(["q1"], topic="", original_question="orig")
        assert "##" not in result


# ---------------------------------------------------------------------------
# get_quality_metrics
# ---------------------------------------------------------------------------

class TestGetQualityMetrics:
    def test_returns_dict(self):
        assert isinstance(get_quality_metrics([]), dict)

    def test_empty_returns_zero_metrics(self):
        m = get_quality_metrics([])
        assert m["positive_count"] == 0
        assert m["avg_score"] == 0
        assert m["top_score"] == 0
        assert m["score_spread"] == 0
        assert m["high_quality_count"] == 0

    def test_has_all_required_keys(self):
        m = get_quality_metrics([])
        for key in ("positive_count", "avg_score", "top_score", "score_spread", "high_quality_count"):
            assert key in m

    def test_positive_count_excludes_negatives(self):
        chunks = [
            {"_rerank_score": 0.8},
            {"_rerank_score": 0.3},
            {"_rerank_score": -0.1},
        ]
        m = get_quality_metrics(chunks)
        assert m["positive_count"] == 2

    def test_top_score_correct(self):
        chunks = [{"_rerank_score": 0.8}, {"_rerank_score": 0.3}]
        m = get_quality_metrics(chunks)
        assert m["top_score"] == pytest.approx(0.8)

    def test_avg_score_correct(self):
        chunks = [{"_rerank_score": 0.6}, {"_rerank_score": 0.4}]
        m = get_quality_metrics(chunks)
        assert m["avg_score"] == pytest.approx(0.5)

    def test_score_spread_correct(self):
        chunks = [{"_rerank_score": 0.8}, {"_rerank_score": 0.2}]
        m = get_quality_metrics(chunks)
        assert m["score_spread"] == pytest.approx(0.6)

    def test_score_spread_single_chunk_is_zero(self):
        chunks = [{"_rerank_score": 0.7}]
        m = get_quality_metrics(chunks)
        assert m["score_spread"] == 0.0

    def test_high_quality_above_threshold(self):
        chunks = [{"_rerank_score": 0.6}, {"_rerank_score": 0.4}, {"_rerank_score": 0.8}]
        m = get_quality_metrics(chunks)
        assert m["high_quality_count"] == 2  # > 0.5

    def test_uses_combined_score_fallback(self):
        chunks = [{"_combined_score": 0.9}]
        m = get_quality_metrics(chunks)
        assert m["top_score"] == pytest.approx(0.9)

    def test_rerank_score_takes_priority(self):
        chunks = [{"_rerank_score": 0.7, "_combined_score": 0.3}]
        m = get_quality_metrics(chunks)
        assert m["top_score"] == pytest.approx(0.7)
