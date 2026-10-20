"""Tests for pure functions in mullet.knowledge.engine.refinement."""

from datetime import datetime

from mullet.knowledge.engine.refinement import (
    contextualize_query,
    refine_question,
    analyze_search_sufficiency,
    generate_alternative_queries,
    create_refinement_log,
    needs_contextualization,
    get_quality_metrics,
    combine_questions_for_search,
    generate_query_variants,
    _suggest_alternative_terms,
)


class TestNeedsContextualization:
    def test_very_short_question_needs_context(self):
        assert needs_contextualization("Why?") is True

    def test_four_word_question_needs_context(self):
        assert needs_contextualization("What does it do?") is True

    def test_standalone_full_question_no_context(self):
        assert needs_contextualization("How does the authentication system work?") is False

    def test_pronoun_start_needs_context(self):
        assert needs_contextualization("That function works differently.") is True

    def test_this_start_needs_context(self):
        assert needs_contextualization("This needs to be explained in detail.") is True

    def test_how_question_is_standalone(self):
        result = needs_contextualization("How does the router handle incoming requests?")
        assert result is False


class TestGetQualityMetrics:
    def test_empty_chunks_all_zero(self):
        m = get_quality_metrics([])
        assert m["positive_count"] == 0
        assert m["avg_score"] == 0
        assert m["top_score"] == 0
        assert m["score_spread"] == 0
        assert m["high_quality_count"] == 0

    def test_single_chunk_rerank_score(self):
        m = get_quality_metrics([{"_rerank_score": 0.8}])
        assert m["top_score"] == 0.8
        assert m["avg_score"] == 0.8
        assert m["high_quality_count"] == 1

    def test_combined_score_used_as_fallback(self):
        m = get_quality_metrics([{"_combined_score": 0.6}])
        assert m["top_score"] == 0.6

    def test_score_spread_multiple_chunks(self):
        chunks = [{"_rerank_score": 0.9}, {"_rerank_score": 0.3}]
        m = get_quality_metrics(chunks)
        assert abs(m["score_spread"] - 0.6) < 1e-9

    def test_high_quality_count(self):
        chunks = [{"_rerank_score": 0.8}, {"_rerank_score": 0.2}, {"_rerank_score": 0.6}]
        m = get_quality_metrics(chunks)
        assert m["high_quality_count"] == 2

    def test_positive_count_excludes_zeros(self):
        chunks = [{"_rerank_score": 0.5}, {"_rerank_score": 0.0}]
        m = get_quality_metrics(chunks)
        assert m["positive_count"] == 1


class TestCombineQuestionsForSearch:
    def test_empty_questions_and_topic(self):
        result = combine_questions_for_search([], "", "")
        assert result == ""

    def test_topic_becomes_heading(self):
        result = combine_questions_for_search([], topic="Authentication")
        assert "## Authentication" in result

    def test_original_question_included(self):
        result = combine_questions_for_search([], original_question="What is CDD?")
        assert "What is CDD?" in result

    def test_questions_appended(self):
        result = combine_questions_for_search(["Q1", "Q2"], original_question="Q0")
        assert "Q1" in result
        assert "Q2" in result

    def test_original_not_duplicated(self):
        result = combine_questions_for_search(["same"], original_question="same")
        count = result.count("same")
        assert count == 1


class TestSuggestAlternativeTerms:
    def test_kyc_returns_alternatives(self):
        suggestions = _suggest_alternative_terms("What is KYC?")
        assert len(suggestions) > 0
        joined = " ".join(suggestions)
        assert "customer" in joined.lower() or "verification" in joined.lower()

    def test_aml_returns_alternatives(self):
        suggestions = _suggest_alternative_terms("What is AML screening?")
        assert len(suggestions) > 0

    def test_generic_question_uses_word_slices(self):
        suggestions = _suggest_alternative_terms("what are the requirements for registration")
        assert len(suggestions) > 0

    def test_max_four_suggestions(self):
        suggestions = _suggest_alternative_terms("kyc aml cdd pep high risk electronic rely")
        assert len(suggestions) <= 4


class TestGenerateQueryVariants:
    def test_no_llm_returns_original(self):
        result = generate_query_variants("find auth handler", get_llm_response=None)
        assert result == ["find auth handler"]

    def test_llm_returns_valid_json_array(self):
        def mock_llm(msgs, sys_prompt):
            return '["variant one", "variant two"]'

        result = generate_query_variants("original query", count=2, get_llm_response=mock_llm)
        assert result[0] == "original query"
        assert "variant one" in result
        assert "variant two" in result

    def test_llm_json_parse_error_returns_original(self):
        def bad_llm(msgs, sys_prompt):
            return "not json at all"

        result = generate_query_variants("query", get_llm_response=bad_llm)
        assert result == ["query"]

    def test_llm_exception_returns_original(self):
        def exploding_llm(msgs, sys_prompt):
            raise RuntimeError("connection error")

        result = generate_query_variants("query", get_llm_response=exploding_llm)
        assert result == ["query"]


class TestNeedsContextualizationAdditional:
    def test_pronoun_first_word_needs_context(self):
        """Lines 92-93: question starts with pronoun 'it' → True."""
        # "It works fine, but can we improve it?" — "it" as first word
        assert needs_contextualization("It seems different from before") is True

    def test_medium_length_question_no_indicator_short_returns_true(self):
        """Line 96: 5-word question without standalone indicators → True."""
        assert needs_contextualization("Why does this matter much") is True

    def test_seven_word_no_indicator_returns_false(self):
        """Line 96: 7-word question without indicators → False (len > 6)."""
        result = needs_contextualization("what happens in the configuration update path")
        assert result is False


class TestContextualizeQuery:
    def test_standalone_question_not_rewritten(self):
        """Line 116-117: non-followup question → returned as-is, was_rewritten=False."""
        q, rewritten = contextualize_query("What is the authentication module?", [])
        assert q == "What is the authentication module?"
        assert rewritten is False

    def test_no_history_returns_original(self):
        """Lines 120-121: followup but no history (< 2 messages) → original."""
        history = [{"role": "user", "content": "hello"}]
        q, rewritten = contextualize_query("That seems odd", history)
        assert q == "That seems odd"
        assert rewritten is False

    def test_no_llm_returns_original(self):
        """Lines 123-125: followup + history but no LLM → original."""
        history = [
            {"role": "user", "content": "tell me about X"},
            {"role": "assistant", "content": "X is a thing"},
        ]
        q, rewritten = contextualize_query("That", history, get_llm_response=None)
        assert q == "That"
        assert rewritten is False

    def test_successful_rewrite(self):
        """Lines 128-174: LLM returns longer rewritten question → (rewritten, True)."""
        history = [
            {"role": "user", "content": "Tell me about CDD"},
            {"role": "assistant", "content": "CDD means customer due diligence"},
        ]
        mock_llm = lambda msgs, sys: "What are the specific requirements for customer due diligence?"
        q, rewritten = contextualize_query("Tell me more", history, get_llm_response=mock_llm)
        assert rewritten is True
        assert "customer due diligence" in q.lower()

    def test_llm_returns_shorter_returns_original(self):
        """Line 172: rewritten is not longer than original → returns original."""
        history = [
            {"role": "user", "content": "context"},
            {"role": "assistant", "content": "reply"},
        ]
        mock_llm = lambda msgs, sys: "No"  # shorter than "That seems odd"
        q, rewritten = contextualize_query("That seems odd", history, get_llm_response=mock_llm)
        assert rewritten is False

    def test_llm_returns_quoted_string_unquoted(self):
        """Lines 166-169: response wrapped in quotes → quotes stripped."""
        history = [
            {"role": "user", "content": "what is X?"},
            {"role": "assistant", "content": "X is very important"},
        ]
        long_answer = '"What are the detailed requirements for X in compliance contexts?"'
        mock_llm = lambda msgs, sys: long_answer
        q, rewritten = contextualize_query("That", history, get_llm_response=mock_llm)
        assert not q.startswith('"')

    def test_llm_exception_returns_original(self):
        """Lines 176-179: LLM raises → returns original, was_rewritten=False."""
        history = [
            {"role": "user", "content": "context"},
            {"role": "assistant", "content": "answer"},
        ]
        def bad_llm(msgs, sys): raise RuntimeError("API down")
        q, rewritten = contextualize_query("That", history, get_llm_response=bad_llm)
        assert rewritten is False

    def test_long_assistant_message_truncated(self):
        """Line 136-137: assistant content >500 chars → truncated with ..."""
        long_content = "A" * 600
        history = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": long_content},
        ]
        # The prompt should be built without error (truncation occurs in building)
        mock_llm = lambda msgs, sys: "What is a very much longer rewritten question here?"
        q, rewritten = contextualize_query("That", history, get_llm_response=mock_llm)
        # We just verify no exception; result may vary
        assert isinstance(q, str)


class TestRefineQuestion:
    def test_no_llm_returns_fallback(self):
        """Lines 198-204: no LLM → fallback dict with original question."""
        result = refine_question("What is CDD?", get_llm_response=None)
        assert result["suggestions"][0]["text"] == "What is CDD?"
        assert result["topic"] == ""

    def test_with_llm_success(self):
        """Lines 220-232: LLM returns valid JSON → parsed and returned."""
        response_json = '{"topic": "CDD", "understanding": "definition", "suggestions": [{"text": "Q1", "angle": "angle1"}]}'
        mock_llm = lambda msgs, sys: response_json
        result = refine_question("What is CDD?", get_llm_response=mock_llm)
        assert result["topic"] == "CDD"

    def test_with_conversation_history(self):
        """Lines 208-214: conversation_history adds context to prompt."""
        response_json = '{"topic": "T", "understanding": "U", "suggestions": []}'
        history = [{"role": "user", "content": "previous question"}]
        mock_llm = lambda msgs, sys: response_json
        result = refine_question("Tell me more", history, get_llm_response=mock_llm)
        assert isinstance(result, dict)

    def test_json_decode_error_returns_fallback(self):
        """Line 234-235: invalid JSON → fallback returned."""
        mock_llm = lambda msgs, sys: "not json"
        result = refine_question("What is X?", get_llm_response=mock_llm)
        assert "suggestions" in result

    def test_exception_returns_fallback(self):
        """Lines 236-237: generic exception → fallback returned."""
        def bad_llm(msgs, sys): raise RuntimeError("timeout")
        result = refine_question("What is X?", get_llm_response=bad_llm)
        assert "suggestions" in result


class TestAnalyzeSearchSufficiency:
    def _chunk(self, rerank=0.0, content="", source=""):
        return {"_rerank_score": rerank, "content": content, "metadata": {"source_file": source}}

    def test_empty_chunks_returns_insufficient(self):
        result = analyze_search_sufficiency("q", [])
        assert result["sufficient"] is False
        assert result["reason"] == "No chunks found"

    def test_low_top_score_returns_insufficient(self):
        """Line 341: top_score < 0.3 → insufficient."""
        chunks = [self._chunk(rerank=0.1)]
        result = analyze_search_sufficiency("q", chunks)
        assert result["sufficient"] is False
        assert "Low relevance" in result["reason"]

    def test_positive_count_too_low_returns_insufficient(self):
        """Line 350: positive_count < 2 with acceptable top_score → insufficient."""
        chunks = [self._chunk(rerank=0.5)]  # 1 positive, top_score = 0.5
        result = analyze_search_sufficiency("q", chunks)
        assert result["sufficient"] is False
        assert "positive" in result["reason"].lower()

    def test_reference_only_chunks_returns_insufficient(self):
        """Lines 395-402: 3+ reference-only chunks → insufficient."""
        ref_content = "please refer to appendix for details"
        chunks = [self._chunk(rerank=0.4, content=ref_content) for _ in range(3)]
        chunks += [self._chunk(rerank=0.4)]  # make positive_count >= 2
        result = analyze_search_sufficiency("q", chunks)
        assert result["sufficient"] is False
        assert "references" in result["reason"].lower()

    def test_list_query_needs_more_high_quality(self):
        """Lines 412-419: list query with < 3 high-quality chunks → insufficient."""
        chunks = [self._chunk(rerank=0.6), self._chunk(rerank=0.6)]  # 2 high-quality, positive_count >= 2
        result = analyze_search_sufficiency("list of all the things", chunks)
        assert result["sufficient"] is False
        assert "List query" in result["reason"]

    def test_high_quality_chunks_returns_sufficient(self):
        """Lines 422-429: 2+ high-quality chunks → sufficient."""
        chunks = [self._chunk(rerank=0.7), self._chunk(rerank=0.8), self._chunk(rerank=0.4)]
        result = analyze_search_sufficiency("what is X?", chunks)
        assert result["sufficient"] is True

    def test_borderline_no_llm_with_positive_count_returns_sufficient(self):
        """Lines 432-441: borderline case without LLM, >= 3 positive → sufficient."""
        # Need: top_score >= 0.3, positive_count >= 2, not list query, not reference only, high_quality < 2
        chunks = [
            self._chunk(rerank=0.35),  # positive but not high-quality (< 0.5)
            self._chunk(rerank=0.35),
            self._chunk(rerank=0.35),
        ]
        result = analyze_search_sufficiency("something specific about requirements", chunks, get_llm_response=None)
        assert result["sufficient"] is True

    def test_borderline_no_llm_insufficient_result(self):
        """Lines 442-448: borderline, < 3 positive, no LLM → insufficient."""
        # top_score >= 0.3, positive_count == 2, high_quality < 2
        chunks = [
            self._chunk(rerank=0.35),
            self._chunk(rerank=0.35),
        ]
        result = analyze_search_sufficiency("something specific", chunks, get_llm_response=None)
        assert result["sufficient"] is False

    def test_llm_analysis_used_for_borderline(self):
        """Lines 450-502: borderline case with LLM → LLM result used."""
        chunks = [
            self._chunk(rerank=0.35, content="Some content " * 10),
            self._chunk(rerank=0.35, content="More content " * 10),
        ]
        llm_json = '{"sufficient": true, "reason": "LLM says ok", "missing_info": [], "better_search_terms": []}'
        mock_llm = lambda msgs, sys: llm_json
        result = analyze_search_sufficiency("something specific", chunks, get_llm_response=mock_llm)
        assert result["sufficient"] is True
        assert "quality_metrics" in result

    def test_llm_exception_falls_back_to_metrics(self):
        """Lines 492-502: LLM raises → fallback using positive_count."""
        chunks = [
            self._chunk(rerank=0.35),
            self._chunk(rerank=0.35),
            self._chunk(rerank=0.35),
        ]
        def bad_llm(msgs, sys): raise RuntimeError("down")
        result = analyze_search_sufficiency("x", chunks, get_llm_response=bad_llm)
        assert "quality_metrics" in result


class TestGenerateAlternativeQueries:
    def test_no_llm_returns_empty(self):
        result = generate_alternative_queries("q", ["prev"], ["missing"], get_llm_response=None)
        assert result == []

    def test_llm_returns_valid_json(self):
        mock_llm = lambda msgs, sys: '["query1", "query2", "query3"]'
        result = generate_alternative_queries("q", ["prev"], ["missing"], get_llm_response=mock_llm)
        assert "query1" in result
        assert len(result) == 3

    def test_llm_invalid_json_returns_empty(self):
        mock_llm = lambda msgs, sys: "not json"
        result = generate_alternative_queries("q", [], [], get_llm_response=mock_llm)
        assert result == []

    def test_llm_exception_returns_empty(self):
        def bad_llm(msgs, sys): raise RuntimeError("fail")
        result = generate_alternative_queries("q", [], [], get_llm_response=bad_llm)
        assert result == []


class TestCreateRefinementLog:
    def test_basic_log_structure(self):
        """Lines 666-679: basic log with no start_time → duration_seconds=0."""
        result = create_refinement_log(
            original_question="What is X?",
            final_question="What is X really?",
            suggestions=[{"text": "Alt Q", "angle": "Different angle"}],
        )
        assert result["original_question"] == "What is X?"
        assert result["final_question"] == "What is X really?"
        assert result["duration_seconds"] == 0
        assert "timestamp" in result

    def test_log_with_start_time(self):
        """Line 667: start_time provided → duration > 0."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        result = create_refinement_log(
            original_question="Q",
            final_question="Q2",
            suggestions=[],
            start_time=start,
        )
        assert result["duration_seconds"] > 0

    def test_selected_questions_included(self):
        """Line 675: selected_questions=None → empty list in result."""
        result = create_refinement_log("Q", "Q", [], selected_questions=["selected Q"])
        assert result["selected_questions"] == ["selected Q"]


class TestRemainingGapLines:
    def test_contextualize_single_quoted_response_stripped(self):
        """Line 169: response wrapped in single quotes → quotes stripped."""
        history = [
            {"role": "user", "content": "what is X?"},
            {"role": "assistant", "content": "X is very important"},
        ]
        long_answer = "'What are the detailed requirements for X in compliance contexts?'"
        mock_llm = lambda msgs, sys: long_answer
        q, rewritten = contextualize_query("That", history, get_llm_response=mock_llm)
        assert not q.startswith("'")

    def test_refine_question_json_decode_error_on_brace_content(self):
        """Line 235: response has braces but invalid JSON → JSONDecodeError caught."""
        mock_llm = lambda msgs, sys: "{invalid json content}"
        result = refine_question("What is X?", get_llm_response=mock_llm)
        assert "suggestions" in result  # fallback returned

    def test_analyze_sufficiency_negative_score_chunk_skipped(self):
        """Line 377: chunk with score < 0 → continue (skipped in reference check)."""
        chunks = [
            {"_rerank_score": -0.1, "content": "negative score chunk", "metadata": {"source_file": ""}},
            {"_rerank_score": 0.6, "content": "good content " * 20, "metadata": {"source_file": ""}},
            {"_rerank_score": 0.6, "content": "more content " * 20, "metadata": {"source_file": ""}},
        ]
        result = analyze_search_sufficiency("what is X?", chunks)
        assert result["sufficient"] is True  # high-quality chunks found
