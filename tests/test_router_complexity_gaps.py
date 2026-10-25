"""Gap tests for mullet/core/router/complexity.py — targets uncovered branches."""

from unittest.mock import patch

from mullet.core.router.complexity import (
    Tier,
    LexicalSignals,
    StructuralSignals,
    ContextSignals,
    ComplexitySignals,
    PROVIDER_TIER_CAPABILITIES,
    tiers_for_providers,
    extract_signals,
    score_complexity,
    score_to_tier,
    confidence_breakdown,
    strip_escape_prefix,
    is_simple_message,
    is_reasoning_task,
    apply_speed_mode,
    speed_mode_badge,
    recommend_tier,
    classify_task_type,
    _detect_question_depth,
    _count_subtasks,
    _detect_domain,
    _assess_reversibility,
    _assess_impact,
)


# ---------------------------------------------------------------------------
# tiers_for_providers
# ---------------------------------------------------------------------------

class TestTiersForProviders:
    def test_single_provider(self):
        # Phase E: local is capable at all tiers (free fallback at every level)
        assert tiers_for_providers({"local"}) == ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]

    def test_multiple_providers_union(self):
        assert tiers_for_providers({"local", "sonnet"}) == ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]

    def test_unknown_provider_ignored(self):
        assert tiers_for_providers({"unknown"}) == []

    def test_frozenset_input(self):
        assert tiers_for_providers(frozenset(["opus"])) == ["CRITICAL"]

    def test_all_providers(self):
        result = tiers_for_providers(set(PROVIDER_TIER_CAPABILITIES.keys()))
        assert result == ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]


# ---------------------------------------------------------------------------
# extract_signals — branch coverage gaps
# ---------------------------------------------------------------------------

class TestExtractSignalsGaps:
    def test_code_block_count(self):
        prompt = "```python\nprint(1)\n```\n```python\nprint(2)\n```"
        signals = extract_signals(prompt)
        assert signals.lexical.code_block_count == 2

    def test_how_question_depth(self):
        signals = extract_signals("how do I fix this?")
        assert signals.lexical.question_depth == "how"

    def test_implicit_requirements(self):
        signals = extract_signals("just fix it")
        assert signals.lexical.has_implicit_requirements is True
        signals2 = extract_signals("make it work please")
        assert signals2.lexical.has_implicit_requirements is True

    def test_has_test_requirements(self):
        signals = extract_signals("add tests for the login module")
        assert signals.structural.has_test_requirements is True

    def test_requires_external_knowledge_api(self):
        signals = extract_signals("call the api and check docs")
        assert signals.structural.requires_external_knowledge is True

    def test_requires_external_knowledge_spec(self):
        signals = extract_signals("read the spec file")
        assert signals.structural.requires_external_knowledge is True

    def test_infrastructure_domain(self):
        signals = extract_signals("deploy to kubernetes and configure nginx")
        assert signals.structural.domain_specificity == "infrastructure"

    def test_debugging_keywords(self):
        signals = extract_signals("investigate the race condition and root cause")
        assert signals.lexical.has_debugging_keywords is True

    def test_long_prompt_word_count(self):
        prompt = "word " * 300
        signals = extract_signals(prompt)
        assert signals.lexical.word_count == 300

    def test_subtasks_numbered_list(self):
        prompt = "1. First\n2. Second\n3. Third\n4. Fourth"
        signals = extract_signals(prompt)
        assert signals.structural.estimated_subtasks == 4

    def test_subtasks_action_verbs(self):
        prompt = "add auth, create tests, fix bugs, update docs"
        signals = extract_signals(prompt)
        assert signals.structural.estimated_subtasks >= 4


# ---------------------------------------------------------------------------
# score_complexity — branch coverage for missing lines
# ---------------------------------------------------------------------------

class TestScoreComplexityGaps:
    def test_word_count_above_200(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(word_count=250),
        )
        assert score_complexity(signals) == 10

    def test_word_count_above_500(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(word_count=600),
        )
        assert score_complexity(signals) == 20

    def test_file_path_count_2_plus(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(file_path_count=3),
        )
        assert score_complexity(signals) == 15

    def test_code_block_count_positive(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(code_block_count=2),
        )
        assert score_complexity(signals) == 5

    def test_architecture_keywords(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(has_architecture_keywords=True),
        )
        assert score_complexity(signals) == 20

    def test_debugging_keywords(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(has_debugging_keywords=True),
        )
        assert score_complexity(signals) == 15

    def test_simple_keywords_reduces_score(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(has_simple_keywords=True),
        )
        assert score_complexity(signals) == 0  # clamped at 0

    def test_risk_keywords(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(has_risk_keywords=True),
        )
        assert score_complexity(signals) == 20

    def test_question_depth_why(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(question_depth="why"),
        )
        assert score_complexity(signals) == 10

    def test_question_depth_how(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(question_depth="how"),
        )
        assert score_complexity(signals) == 5

    def test_question_depth_what(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(question_depth="what"),
        )
        assert score_complexity(signals) == 0

    def test_implicit_requirements(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(has_implicit_requirements=True),
        )
        assert score_complexity(signals) == 10

    def test_subtasks_greater_than_3(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(estimated_subtasks=5),
        )
        assert score_complexity(signals) == 20

    def test_subtasks_greater_than_1(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(estimated_subtasks=2),
        )
        assert score_complexity(signals) == 10

    def test_cross_file_dependencies(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(cross_file_dependencies=True),
        )
        assert score_complexity(signals) == 15

    def test_has_test_requirements(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(has_test_requirements=True),
        )
        assert score_complexity(signals) == 5

    def test_domain_security(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(domain_specificity="security"),
        )
        assert score_complexity(signals) == 20

    def test_domain_infrastructure(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(domain_specificity="infrastructure"),
        )
        assert score_complexity(signals) == 15

    def test_domain_general(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(domain_specificity="general"),
        )
        assert score_complexity(signals) == 0

    def test_requires_external_knowledge(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(requires_external_knowledge=True),
        )
        assert score_complexity(signals) == 10

    def test_reversibility_difficult(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(reversibility="difficult"),
        )
        assert score_complexity(signals) == 20

    def test_reversibility_moderate(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(reversibility="moderate"),
        )
        assert score_complexity(signals) == 10

    def test_reversibility_easy(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(reversibility="easy"),
        )
        assert score_complexity(signals) == 0

    def test_impact_system_wide(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(impact_scope="system-wide"),
        )
        assert score_complexity(signals) == 20

    def test_impact_module(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(impact_scope="module"),
        )
        assert score_complexity(signals) == 10

    def test_impact_function(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(impact_scope="function"),
        )
        assert score_complexity(signals) == 0

    def test_context_previous_failures_2_plus(self):
        signals = ComplexitySignals(
            context=ContextSignals(previous_failures=3),
        )
        assert score_complexity(signals) == 20

    def test_context_previous_failures_1(self):
        signals = ComplexitySignals(
            context=ContextSignals(previous_failures=1),
        )
        assert score_complexity(signals) == 10

    def test_context_agent_chain_depth_2_plus(self):
        signals = ComplexitySignals(
            context=ContextSignals(agent_chain_depth=2),
        )
        assert score_complexity(signals) == 10

    def test_context_plan_complexity_5_plus(self):
        signals = ComplexitySignals(
            context=ContextSignals(plan_complexity=5),
        )
        assert score_complexity(signals) == 15

    def test_context_plan_complexity_3_plus(self):
        signals = ComplexitySignals(
            context=ContextSignals(plan_complexity=3),
        )
        assert score_complexity(signals) == 5

    def test_combined_max_score(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(
                word_count=600, file_path_count=3, code_block_count=1,
                has_architecture_keywords=True, has_debugging_keywords=True,
                has_risk_keywords=True, question_depth="why",
                has_implicit_requirements=True,
            ),
            structural=StructuralSignals(
                estimated_subtasks=5, cross_file_dependencies=True,
                has_test_requirements=True, domain_specificity="security",
                requires_external_knowledge=True, reversibility="difficult",
                impact_scope="system-wide",
            ),
            context=ContextSignals(
                previous_failures=3, agent_chain_depth=2, plan_complexity=5,
            ),
        )
        score = score_complexity(signals)
        assert score > 200  # Well above any threshold


# ---------------------------------------------------------------------------
# confidence_breakdown — branch coverage for missing lines
# ---------------------------------------------------------------------------

class TestConfidenceBreakdownGaps:
    def test_lexical_word_count_200(self):
        signals = ComplexitySignals(lexical=LexicalSignals(word_count=250))
        result = confidence_breakdown(signals)
        assert result["lexical_signals"]["word_count_200+"] == 10

    def test_lexical_word_count_500(self):
        signals = ComplexitySignals(lexical=LexicalSignals(word_count=600))
        result = confidence_breakdown(signals)
        assert result["lexical_signals"]["word_count_500+"] == 10

    def test_lexical_file_path_count(self):
        signals = ComplexitySignals(lexical=LexicalSignals(file_path_count=3))
        result = confidence_breakdown(signals)
        assert result["lexical_signals"]["file_path_count_2+"] == 15

    def test_lexical_code_block_count(self):
        signals = ComplexitySignals(lexical=LexicalSignals(code_block_count=2))
        result = confidence_breakdown(signals)
        assert result["lexical_signals"]["code_block_count_1+"] == 5

    def test_lexical_debugging_keywords(self):
        signals = ComplexitySignals(lexical=LexicalSignals(has_debugging_keywords=True))
        result = confidence_breakdown(signals)
        assert result["lexical_signals"]["debugging_keywords"] == 15

    def test_lexical_question_depth_how(self):
        signals = ComplexitySignals(lexical=LexicalSignals(question_depth="how"))
        result = confidence_breakdown(signals)
        assert result["lexical_signals"]["question_depth_how"] == 5

    def test_lexical_question_depth_why(self):
        signals = ComplexitySignals(lexical=LexicalSignals(question_depth="why"))
        result = confidence_breakdown(signals)
        assert result["lexical_signals"]["question_depth_why"] == 10

    def test_lexical_implicit_requirements(self):
        signals = ComplexitySignals(lexical=LexicalSignals(has_implicit_requirements=True))
        result = confidence_breakdown(signals)
        assert result["lexical_signals"]["implicit_requirements"] == 10

    def test_structural_subtasks_3_plus(self):
        signals = ComplexitySignals(structural=StructuralSignals(estimated_subtasks=5))
        result = confidence_breakdown(signals)
        assert result["structural_signals"]["estimated_subtasks_3+"] == 20

    def test_structural_subtasks_1_plus(self):
        signals = ComplexitySignals(structural=StructuralSignals(estimated_subtasks=2))
        result = confidence_breakdown(signals)
        assert result["structural_signals"]["estimated_subtasks_1+"] == 10

    def test_structural_test_requirements(self):
        signals = ComplexitySignals(structural=StructuralSignals(has_test_requirements=True))
        result = confidence_breakdown(signals)
        assert result["structural_signals"]["test_requirements"] == 5

    def test_structural_domain_infrastructure(self):
        signals = ComplexitySignals(structural=StructuralSignals(domain_specificity="infrastructure"))
        result = confidence_breakdown(signals)
        assert result["structural_signals"]["domain_infrastructure"] == 15

    def test_structural_external_knowledge(self):
        signals = ComplexitySignals(structural=StructuralSignals(requires_external_knowledge=True))
        result = confidence_breakdown(signals)
        assert result["structural_signals"]["external_knowledge"] == 10

    def test_structural_reversibility_moderate(self):
        signals = ComplexitySignals(structural=StructuralSignals(reversibility="moderate"))
        result = confidence_breakdown(signals)
        assert result["structural_signals"]["reversibility_moderate"] == 10

    def test_structural_impact_module(self):
        signals = ComplexitySignals(structural=StructuralSignals(impact_scope="module"))
        result = confidence_breakdown(signals)
        assert result["structural_signals"]["impact_module"] == 10

    def test_structural_impact_system_wide(self):
        signals = ComplexitySignals(structural=StructuralSignals(impact_scope="system-wide"))
        result = confidence_breakdown(signals)
        assert result["structural_signals"]["impact_system_wide"] == 20

    def test_context_previous_failures_1(self):
        signals = ComplexitySignals(context=ContextSignals(previous_failures=1))
        result = confidence_breakdown(signals)
        assert result["context_signals"]["previous_failures_1+"] == 10

    def test_context_previous_failures_2(self):
        signals = ComplexitySignals(context=ContextSignals(previous_failures=2))
        result = confidence_breakdown(signals)
        assert result["context_signals"]["previous_failures_2+"] == 20

    def test_context_plan_complexity_3(self):
        signals = ComplexitySignals(context=ContextSignals(plan_complexity=3))
        result = confidence_breakdown(signals)
        assert result["context_signals"]["plan_complexity_3+"] == 5

    def test_context_plan_complexity_5(self):
        signals = ComplexitySignals(context=ContextSignals(plan_complexity=5))
        result = confidence_breakdown(signals)
        assert result["context_signals"]["plan_complexity_5+"] == 15

    def test_total_score_clamping_negative(self):
        signals = ComplexitySignals(lexical=LexicalSignals(has_simple_keywords=True))
        result = confidence_breakdown(signals)
        assert result["total_score"] == 0.0

    def test_full_breakdown_combined(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(
                word_count=600, file_path_count=3, code_block_count=1,
                has_architecture_keywords=True, has_debugging_keywords=True,
                has_risk_keywords=True, question_depth="why",
                has_implicit_requirements=True,
            ),
            structural=StructuralSignals(
                estimated_subtasks=5, cross_file_dependencies=True,
                has_test_requirements=True, domain_specificity="security",
                requires_external_knowledge=True, reversibility="difficult",
                impact_scope="system-wide",
            ),
            context=ContextSignals(
                previous_failures=3, agent_chain_depth=2, plan_complexity=5,
            ),
        )
        result = confidence_breakdown(signals)
        assert result["total_score"] > 200
        assert len(result["lexical_signals"]) > 0
        assert len(result["structural_signals"]) > 0
        assert len(result["context_signals"]) > 0


# ---------------------------------------------------------------------------
# score_to_tier — edge cases
# ---------------------------------------------------------------------------

class TestScoreToTierGaps:
    def test_custom_threshold_local_max(self):
        tier = score_to_tier(25, thresholds={"local_max": 30})
        assert tier == Tier.LOW

    def test_custom_threshold_standard_max(self):
        tier = score_to_tier(50, thresholds={"standard_max": 76})
        assert tier == Tier.STANDARD

    def test_custom_threshold_advanced_max(self):
        tier = score_to_tier(85, thresholds={"advanced_max": 90})
        assert tier == Tier.ADVANCED

    def test_threshold_aliases_low_max(self):
        tier = score_to_tier(25, thresholds={"low_max": 30, "medium_max": 76})
        assert tier == Tier.LOW

    def test_allowed_tiers_empty(self):
        tier = score_to_tier(50, allowed_tiers=[])
        assert tier == Tier.STANDARD

    def test_allowed_tiers_capped_up(self):
        tier = score_to_tier(5, allowed_tiers=["ADVANCED", "CRITICAL"])
        assert tier == Tier.ADVANCED

    def test_allowed_tiers_capped_down(self):
        tier = score_to_tier(95, allowed_tiers=["LOW", "STANDARD"])
        assert tier == Tier.STANDARD

    def test_allowed_tiers_recommended_in_list(self):
        tier = score_to_tier(50, allowed_tiers=["LOW", "STANDARD", "ADVANCED"])
        assert tier == Tier.STANDARD

    def test_allowed_tiers_single_item_below(self):
        tier = score_to_tier(95, allowed_tiers=["STANDARD"])
        assert tier == Tier.STANDARD

    def test_medium_alias(self):
        assert Tier.MEDIUM == Tier.STANDARD
        assert Tier.HIGH == Tier.ADVANCED


# ---------------------------------------------------------------------------
# strip_escape_prefix — remaining prefixes
# ---------------------------------------------------------------------------

class TestStripEscapePrefixGaps:
    def test_just_prefix(self):
        text, had = strip_escape_prefix("just: list files")
        assert text == "list files"
        assert had is True

    def test_minor_prefix(self):
        text, had = strip_escape_prefix("minor: fix typo")
        assert text == "fix typo"
        assert had is True

    def test_small_prefix(self):
        text, had = strip_escape_prefix("small: update comment")
        assert text == "update comment"
        assert had is True

    def test_no_prefix(self):
        text, had = strip_escape_prefix("regular task")
        assert text == "regular task"
        assert had is False


# ---------------------------------------------------------------------------
# is_simple_message — branch coverage
# ---------------------------------------------------------------------------

class TestIsSimpleMessageGaps:
    def test_escape_prefix_forces_simple(self):
        assert is_simple_message("quick: anything here") is True

    def test_too_long_text(self):
        assert is_simple_message("x" * 161) is False

    def test_too_many_words(self):
        assert is_simple_message("word " * 30) is False

    def test_too_many_newlines(self):
        assert is_simple_message("a\nb\nc") is False

    def test_contains_backtick(self):
        assert is_simple_message("run `ls` here") is False

    def test_contains_url(self):
        assert is_simple_message("see https://example.com") is False
        assert is_simple_message("see http://example.com") is False

    def test_blocklisted_keyword(self):
        assert is_simple_message("implement this feature") is False

    def test_truly_simple(self):
        assert is_simple_message("list files") is True
        assert is_simple_message("status") is True


# ---------------------------------------------------------------------------
# is_reasoning_task
# ---------------------------------------------------------------------------

class TestIsReasoningTaskGaps:
    def test_reasoning_keywords(self):
        assert is_reasoning_task("analyse the code") is True
        assert is_reasoning_task("review the logic") is True
        assert is_reasoning_task("trace the execution") is True
        assert is_reasoning_task("evaluate the options") is True
        assert is_reasoning_task("assess the risk") is True
        assert is_reasoning_task("critique the design") is True
        assert is_reasoning_task("audit the security") is True
        assert is_reasoning_task("compare A and B") is True

    def test_why_does_reasoning(self):
        assert is_reasoning_task("why does this fail?") is True

    def test_tradeoff_reasoning(self):
        assert is_reasoning_task("trade-off between A and B") is True
        assert is_reasoning_task("tradeoff between A and B") is True

    def test_simple_not_reasoning(self):
        assert is_reasoning_task("list files") is False
        assert is_reasoning_task("fix typo") is False

    def test_simple_message_short_circuits(self):
        # Even if it has a reasoning keyword, simple messages return False
        assert is_reasoning_task("quick: analyse this") is False


# ---------------------------------------------------------------------------
# apply_speed_mode — remaining branches
# ---------------------------------------------------------------------------

class TestApplySpeedModeGaps:
    def test_economy_returns_same(self):
        assert apply_speed_mode(Tier.LOW, "economy") == Tier.LOW
        assert apply_speed_mode(Tier.CRITICAL, "economy") == Tier.CRITICAL

    def test_balanced_low_to_standard(self):
        assert apply_speed_mode(Tier.LOW, "balanced") == Tier.STANDARD

    def test_balanced_standard_unchanged(self):
        assert apply_speed_mode(Tier.STANDARD, "balanced") == Tier.STANDARD

    def test_fast_low_to_standard(self):
        assert apply_speed_mode(Tier.LOW, "fast") == Tier.STANDARD

    def test_fast_standard_unchanged(self):
        assert apply_speed_mode(Tier.STANDARD, "fast") == Tier.STANDARD

    def test_max_low_to_advanced(self):
        assert apply_speed_mode(Tier.LOW, "max") == Tier.ADVANCED

    def test_max_standard_to_advanced(self):
        assert apply_speed_mode(Tier.STANDARD, "max") == Tier.ADVANCED

    def test_max_advanced_unchanged(self):
        assert apply_speed_mode(Tier.ADVANCED, "max") == Tier.ADVANCED

    def test_max_critical_unchanged(self):
        assert apply_speed_mode(Tier.CRITICAL, "max") == Tier.CRITICAL

    def test_unknown_mode(self):
        assert apply_speed_mode(Tier.LOW, "unknown") == Tier.LOW
        assert apply_speed_mode(Tier.CRITICAL, "turbo") == Tier.CRITICAL


# ---------------------------------------------------------------------------
# speed_mode_badge
# ---------------------------------------------------------------------------

class TestSpeedModeBadgeGaps:
    def test_all_known_badges(self):
        assert speed_mode_badge("economy") == "[eco]"
        assert speed_mode_badge("balanced") == "[bal]"
        assert speed_mode_badge("fast") == "[fast]"
        assert speed_mode_badge("max") == "[max]"

    def test_unknown_badge(self):
        assert speed_mode_badge("turbo") == "[turbo]"
        assert speed_mode_badge("") == "[]"


# ---------------------------------------------------------------------------
# recommend_tier
# ---------------------------------------------------------------------------

class TestRecommendTierGaps:
    def test_simple_task_low(self):
        tier = recommend_tier("what is the version?")
        assert tier == Tier.LOW

    def test_file_count_overrides(self):
        tier = recommend_tier("update config", file_count=10)
        assert tier in (Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL)

    def test_complex_security_task(self):
        tier = recommend_tier(
            "audit the jwt authentication and rbac permissions for security vulnerabilities"
        )
        assert tier != Tier.LOW

    def test_context_escalation(self):
        ctx = ContextSignals(previous_failures=2, agent_chain_depth=2, plan_complexity=5)
        tier = recommend_tier("update config", context=ctx)
        assert tier in (Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL)

    def test_file_count_zero_ignored(self):
        tier_with_zero = recommend_tier("list files", file_count=0)
        tier_without = recommend_tier("list files")
        assert tier_with_zero == tier_without


# ---------------------------------------------------------------------------
# classify_task_type — completely uncovered
# ---------------------------------------------------------------------------

class TestClassifyTaskType:
    def test_code_task(self):
        assert classify_task_type("write a function to sort arrays") == "code"
        assert classify_task_type("implement oauth2") == "code"
        assert classify_task_type("fix the bug in auth") == "code"
        assert classify_task_type("refactor the router") == "code"
        assert classify_task_type("create a new module") == "code"
        assert classify_task_type("add input validation") == "code"
        assert classify_task_type("build the docker image") == "code"
        assert classify_task_type("port this to rust") == "code"
        assert classify_task_type("migrate the database") == "code"
        assert classify_task_type("debug the crash") == "code"

    def test_search_task(self):
        assert classify_task_type("find all TODOs") == "search"
        assert classify_task_type("search for the bug") == "search"
        assert classify_task_type("read the config file") == "search"
        assert classify_task_type("look for errors") == "search"
        assert classify_task_type("check the logs") == "search"
        assert classify_task_type("where is the router defined") == "search"
        assert classify_task_type("grep for auth") == "search"
        assert classify_task_type("locate the missing file") == "search"
        assert classify_task_type("show me the config") == "search"

    def test_generate_task(self):
        assert classify_task_type("explain the code") == "generate"
        assert classify_task_type("summarize the PR") == "generate"
        assert classify_task_type("document the API") == "generate"
        assert classify_task_type("describe the architecture") == "generate"
        assert classify_task_type("summarize the changelog") == "generate"
        assert classify_task_type("list all dependencies") == "generate"
        assert classify_task_type("compare A and B") == "generate"

    def test_lookup_task(self):
        assert classify_task_type("what is the version?") == "lookup"
        assert classify_task_type("how many tests fail?") == "lookup"
        assert classify_task_type("which branch am I on?") == "lookup"
        assert classify_task_type("does it work?") == "lookup"
        assert classify_task_type("is there a config file?") == "lookup"

    def test_default_to_code(self):
        # When no keywords match, default is "code"
        assert classify_task_type("qwertyuiop asdfghjkl") == "code"

    def test_tie_breaker(self):
        # If multiple categories tie, max returns the first one encountered with max value
        # Since dict iteration is insertion-ordered, "code" wins if tied first
        result = classify_task_type("write and find")  # both "write" and "find" match
        # Both score 1, so max picks the first in insertion order: "code"
        assert result == "code"


# ---------------------------------------------------------------------------
# Internal helpers — edge cases
# ---------------------------------------------------------------------------

class TestInternalHelpersGaps:
    def test_detect_question_depth_ordering(self):
        # "why" takes precedence over "how" and "what"
        assert _detect_question_depth("how and why") == "why"
        assert _detect_question_depth("what and how") == "how"

    def test_count_subtasks_numbered_with_parenthesis(self):
        prompt = "1) First\n2) Second"
        assert _count_subtasks(prompt) == 2

    def test_count_subtasks_single_action_verb(self):
        assert _count_subtasks("list files") == 1

    def test_count_subtasks_no_numbered_list_few_verbs(self):
        prompt = "add auth"
        assert _count_subtasks(prompt) == 1

    def test_detect_domain_security_over_infra(self):
        # Security terms checked first
        assert _detect_domain("auth in kubernetes") == "security"

    def test_detect_domain_infra(self):
        assert _detect_domain("deploy with terraform") == "infrastructure"
        assert _detect_domain("setup nginx on aws") == "infrastructure"

    def test_assess_reversibility_hard_variants(self):
        assert _assess_reversibility("database migration") == "difficult"
        assert _assess_reversibility("drop the table") == "difficult"
        assert _assess_reversibility("remove table users") == "difficult"
        assert _assess_reversibility("schema change") == "difficult"
        assert _assess_reversibility("breaking change") == "difficult"

    def test_assess_reversibility_moderate_variants(self):
        assert _assess_reversibility("refactor the code") == "moderate"
        assert _assess_reversibility("rename the function") == "moderate"
        assert _assess_reversibility("move the file") == "moderate"
        assert _assess_reversibility("restructure the project") == "moderate"

    def test_assess_impact_system_wide_keywords(self):
        assert _assess_impact("entire codebase", 0) == "system-wide"
        assert _assess_impact("all modules", 0) == "system-wide"
        assert _assess_impact("everything", 0) == "system-wide"

    def test_assess_impact_module_keywords(self):
        assert _assess_impact("the module", 0) == "module"
        assert _assess_impact("the package", 0) == "module"
        assert _assess_impact("the service", 0) == "module"

    def test_assess_impact_file_count_5(self):
        assert _assess_impact("update files", 5) == "module"

    def test_assess_impact_function_keywords(self):
        assert _assess_impact("fix the function", 0) == "function"
        assert _assess_impact("update method", 0) == "function"
        assert _assess_impact("refactor class", 0) == "function"

    def test_assess_impact_file_count_2(self):
        assert _assess_impact("update files", 2) == "function"

    def test_assess_impact_default_line(self):
        assert _assess_impact("add comment", 0) == "line"
        assert _assess_impact("fix typo", 1) == "line"


# ---------------------------------------------------------------------------
# Integration-style end-to-end
# ---------------------------------------------------------------------------

class TestIntegrationGaps:
    def test_full_pipeline_simple(self):
        signals = extract_signals("list files")
        score = score_complexity(signals)
        tier = score_to_tier(score)
        assert tier == Tier.LOW

    def test_full_pipeline_complex(self):
        prompt = (
            "refactor the authentication system to support OAuth2 with JWT tokens, "
            "handle all edge cases, migrate the database schema, "
            "and ensure production deployment is safe"
        )
        signals = extract_signals(prompt)
        score = score_complexity(signals)
        tier = score_to_tier(score)
        assert tier in (Tier.ADVANCED, Tier.CRITICAL)

    def test_recommend_tier_with_file_count(self):
        tier = recommend_tier("fix the auth module", file_count=3)
        assert tier in (Tier.STANDARD, Tier.ADVANCED)

    def test_speed_mode_with_recommend_tier(self):
        tier = recommend_tier("list files")
        assert apply_speed_mode(tier, "fast") == Tier.STANDARD
        assert apply_speed_mode(tier, "max") == Tier.ADVANCED

    def test_classify_then_map_to_tier(self):
        task = "find all TODO comments in the codebase"
        task_type = classify_task_type(task)
        from mullet.core.router.complexity import TASK_TYPE_PROVIDERS
        assert task_type in TASK_TYPE_PROVIDERS

    def test_confidence_breakdown_matches_score(self):
        signals = extract_signals(
            "refactor auth module to support OAuth2 and jwt tokens in production"
        )
        score = score_complexity(signals)
        breakdown = confidence_breakdown(signals)
        assert breakdown["total_score"] == score

    def test_is_reasoning_task_integration(self):
        assert is_reasoning_task("analyse the trade-offs") is True
        assert is_reasoning_task("list files") is False
        assert is_reasoning_task("quick: analyse the trade-offs") is False

    def test_strip_escape_prefix_integration(self):
        clean, had = strip_escape_prefix("simple:  do this")
        assert had is True
        assert clean == "do this"

    def test_logging_on_cap(self):
        """score_to_tier logs when capping to a lower tier."""
        with patch("logging.Logger.info") as mock_log:
            score_to_tier(95, allowed_tiers=["LOW", "STANDARD"])
            mock_log.assert_called_once()
            assert "capped" in mock_log.call_args[0][0].lower() or " capped" in str(mock_log.call_args)
