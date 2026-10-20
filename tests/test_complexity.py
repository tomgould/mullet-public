"""Tests for complexity analysis and routing."""

from mullet.core.router.complexity import (
    Tier, ContextSignals, extract_signals, is_simple_message,
    score_complexity, score_to_tier, strip_escape_prefix,
)
from mullet.core.router.decision import Router, ProviderEntry


# --- Complexity scoring tests ---

def test_simple_message():
    assert is_simple_message("list files") is True
    assert is_simple_message("what is the status?") is True


def test_complex_message():
    assert is_simple_message("refactor the authentication system to support OAuth2") is False
    assert is_simple_message("x " * 100) is False  # too long


def test_code_blocks_not_simple():
    assert is_simple_message("fix this ```code```") is False


def test_url_not_simple():
    assert is_simple_message("check https://example.com") is False


def test_multiline_not_simple():
    assert is_simple_message("do this\nthen that\nthen more") is False


def test_extract_signals_simple():
    signals = extract_signals("list the files")
    assert signals.lexical.has_simple_keywords is True
    assert signals.lexical.has_architecture_keywords is False
    assert signals.lexical.word_count == 3


def test_extract_signals_complex():
    prompt = "refactor the authentication system to use OAuth2 with JWT tokens and handle all edge cases in production"
    signals = extract_signals(prompt)
    assert signals.lexical.has_architecture_keywords is True
    assert signals.lexical.has_risk_keywords is True
    assert signals.structural.domain_specificity == "security"


def test_extract_signals_file_paths():
    signals = extract_signals("fix the bug in src/auth.py and tests/test_auth.py")
    assert signals.lexical.file_path_count >= 2
    assert signals.structural.cross_file_dependencies is True


def test_score_simple_task():
    signals = extract_signals("list files")
    score = score_complexity(signals)
    assert score < 20  # Should be LOW


def test_score_medium_task():
    signals = extract_signals("implement input validation for the user registration form")
    score = score_complexity(signals)
    tier = score_to_tier(score)
    assert tier in (Tier.LOW, Tier.STANDARD)


def test_score_complex_task():
    prompt = ("refactor the authentication system to support OAuth2 with token refresh, "
              "handle all edge cases, add migration for the database schema, "
              "and ensure production deployment is safe")
    signals = extract_signals(prompt)
    score = score_complexity(signals)
    tier = score_to_tier(score)
    assert tier != Tier.LOW


def test_score_to_tier_thresholds():
    assert score_to_tier(5) == Tier.LOW
    assert score_to_tier(29) == Tier.LOW
    assert score_to_tier(30) == Tier.STANDARD
    assert score_to_tier(75) == Tier.STANDARD
    assert score_to_tier(76) == Tier.ADVANCED
    assert score_to_tier(89) == Tier.ADVANCED
    assert score_to_tier(90) == Tier.CRITICAL
    assert score_to_tier(100) == Tier.CRITICAL


def test_question_depth():
    signals = extract_signals("why does this crash?")
    assert signals.lexical.question_depth == "why"

    signals = extract_signals("how do I fix this?")
    assert signals.lexical.question_depth == "how"

    signals = extract_signals("what is the error?")
    assert signals.lexical.question_depth == "what"


# --- Escape hatch tests ---

def test_escape_prefix_quick():
    text, had = strip_escape_prefix("quick: fix the bug")
    assert text == "fix the bug"
    assert had is True


def test_escape_prefix_simple():
    text, had = strip_escape_prefix("simple: list files")
    assert text == "list files"
    assert had is True


def test_escape_prefix_just():
    text, had = strip_escape_prefix("just: echo hello")
    assert text == "echo hello"
    assert had is True


def test_escape_no_prefix():
    text, had = strip_escape_prefix("refactor the auth module")
    assert text == "refactor the auth module"
    assert had is False


def test_escape_forces_simple():
    """Escape prefix should make is_simple_message return True even for complex text."""
    # Without prefix: complex task scores high
    assert not is_simple_message("refactor the entire authentication system")
    # With prefix: forced simple
    assert is_simple_message("quick: refactor the entire authentication system")


# --- Router tests ---

def _make_router(**kwargs) -> Router:
    providers = {
        "local": ProviderEntry(name="local", tier="free", models={"default": "mistral-nemo", "low": "mistral-nemo"}),
        "cheap_cloud": ProviderEntry(name="cheap_cloud", tier="cheap", models={"default": "haiku", "medium": "haiku"},
                                     cost_per_1k_input=0.0001, cost_per_1k_output=0.0004),
        "premium_cloud": ProviderEntry(name="premium_cloud", tier="premium", models={"default": "sonnet", "high": "opus"},
                                       cost_per_1k_input=0.003, cost_per_1k_output=0.015),
    }
    return Router(providers=providers, **kwargs)


def test_route_simple_to_local():
    router = _make_router()
    decision = router.route("list files")
    assert decision.tier == Tier.LOW
    assert decision.provider == "local"


def test_route_complex_to_premium():
    router = _make_router()
    decision = router.route(
        "refactor the entire authentication system to support OAuth2 with JWT "
        "token refresh, handle all security edge cases, migrate the database schema, "
        "and ensure production deployment safety"
    )
    assert decision.tier == Tier.ADVANCED
    assert decision.provider == "premium_cloud"


def test_route_agent_override():
    router = _make_router()
    decision = router.route("do something simple", agent_type="planner")
    assert decision.tier == Tier.ADVANCED
    assert "agent override" in decision.reasons[0]


def test_route_agent_adaptive():
    router = _make_router()
    decision = router.route("list files", agent_type="debugger")
    assert decision.tier == Tier.LOW  # Simple task, no override


def test_route_force_tier():
    router = _make_router()
    decision = router.route("list files", force_tier="ADVANCED")
    assert decision.tier == Tier.ADVANCED


def test_route_force_provider():
    router = _make_router()
    decision = router.route("anything", force_provider="premium_cloud")
    assert decision.provider == "premium_cloud"


def test_route_local_only():
    router = _make_router(local_only=True)
    decision = router.route("refactor the entire codebase")
    assert decision.tier == Tier.LOW
    assert decision.provider == "local"
    assert "local-only" in decision.reasons[0]


def test_route_fallback_on_failure():
    router = _make_router()
    router.mark_failure("local", RuntimeError("down"))
    decision = router.route("list files")
    # LOCAL is down, should fall back to cheap_cloud (next in LOW tier map)
    assert decision.provider == "cheap_cloud"


def test_route_delegation():
    router = _make_router()
    decision = router.route_delegation("ADVANCED", "summarize this log file")
    # Should be one tier below parent (MEDIUM), but simple task → LOW wins
    assert decision.tier in (Tier.LOW, Tier.STANDARD)


def test_route_delegation_tier_comparison():
    """Verify Tier comparison is numeric, not alphabetical.

    Bug: 'HIGH'.value <= 'LOW'.value was True because 'H' < 'L' in ASCII.
    Fixed: uses numeric tier_order dict.
    """
    router = _make_router()
    # A complex child task from a HIGH parent should stay at MEDIUM (one tier down),
    # not get downgraded to LOW because of alphabetical string comparison
    decision = router.route_delegation(
        "ADVANCED",
        "refactor the authentication module to support OAuth2 with JWT tokens and handle security edge cases"
    )
    # This is a complex task — should be at least STANDARD, not LOW
    assert decision.tier in (Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL)


def test_mark_success_resets_failure():
    router = _make_router()
    router.mark_failure("local", RuntimeError("down"))
    assert router.providers["local"].healthy is False
    router.mark_success("local")
    assert router.providers["local"].healthy is True


# --- Context signal tests ---

def test_context_signals_default():
    signals = extract_signals("list files")
    assert signals.context.previous_failures == 0
    assert signals.context.conversation_turns == 0
    assert signals.context.agent_chain_depth == 0


def test_context_failures_escalate():
    """Previous failures should push the score up."""
    base = extract_signals("list files")
    base_score = score_complexity(base)

    with_failures = extract_signals(
        "list files",
        context=ContextSignals(previous_failures=2),
    )
    fail_score = score_complexity(with_failures)
    assert fail_score > base_score


def test_context_deep_chain_escalates():
    """Deep delegation chains should increase complexity."""
    base = extract_signals("summarize this")
    base_score = score_complexity(base)

    deep = extract_signals(
        "summarize this",
        context=ContextSignals(agent_chain_depth=3),
    )
    deep_score = score_complexity(deep)
    assert deep_score > base_score


def test_context_complex_plan_escalates():
    """Complex plans should push score higher."""
    ctx = ContextSignals(plan_complexity=7)
    signals = extract_signals("do the next step", context=ctx)
    score = score_complexity(signals)
    assert score >= 15  # plan_complexity >= 5 adds 15


def test_budget_blocks_premium_uses_fallback():
    """When budget blocks premium, router should use cheaper provider from tier map."""
    from unittest.mock import MagicMock
    budget = MagicMock()
    # Premium too expensive, cheap and local OK
    def mock_can_afford(provider_name, model, est_tokens):
        return provider_name != "premium_cloud"
    budget.can_afford.side_effect = mock_can_afford
    budget.estimate_cost = MagicMock(return_value=0.001)

    router = _make_router(budget_tracker=budget)
    decision = router.route(
        "refactor the entire authentication system to support OAuth2 with JWT "
        "token refresh, handle all security edge cases, migrate the database schema"
    )
    # Should fall back to cheap_cloud since premium is unaffordable
    assert decision.provider != "premium_cloud"
    assert decision.provider == "cheap_cloud"
    assert any("budget" in r for r in decision.reasons)


def test_budget_auto_downgrade_all_blocked():
    """When ALL providers in a tier are budget-blocked, auto-downgrade to cheaper tier."""
    from unittest.mock import MagicMock
    budget = MagicMock()
    # Only local is affordable
    def mock_can_afford(provider_name, model, est_tokens):
        return provider_name == "local"
    budget.can_afford.side_effect = mock_can_afford
    budget.estimate_cost = MagicMock(return_value=0.0)

    router = _make_router(budget_tracker=budget)
    decision = router.route(
        "refactor the entire authentication system to support OAuth2 with JWT "
        "token refresh, handle all security edge cases, migrate the database schema"
    )
    # HIGH tier fully blocked, MEDIUM blocked — should downgrade to LOCAL
    assert decision.provider == "local"
    assert any("budget downgrade" in r for r in decision.reasons)


def test_context_escalation_changes_tier():
    """Enough context escalation should push a LOW task to MEDIUM."""
    # Neutral task (no simple keywords) normally routes LOW
    simple = extract_signals("update the config values")
    assert score_to_tier(score_complexity(simple)) == Tier.LOW

    # Same task but with 2 failures + deep chain → should escalate
    escalated = extract_signals(
        "update the config values",
        context=ContextSignals(previous_failures=2, agent_chain_depth=2),
    )
    tier = score_to_tier(score_complexity(escalated))
    assert tier in (Tier.STANDARD, Tier.ADVANCED)


# --- G15: allowlist tier capping ------------------------------------------

from mullet.core.router.complexity import Tier as _Tier_G15, score_to_tier as _score_to_tier_G15


def test_score_to_tier_caps_down_to_allowed_max():
    tier = _score_to_tier_G15(200.0, allowed_tiers=["LOW", "STANDARD"])
    assert tier == _Tier_G15.STANDARD


def test_score_to_tier_caps_up_to_allowed_min():
    tier = _score_to_tier_G15(0.0, allowed_tiers=["ADVANCED", "CRITICAL"])
    assert tier == _Tier_G15.ADVANCED


def test_score_to_tier_no_allowlist_is_unchanged():
    assert _score_to_tier_G15(200.0) == _Tier_G15.CRITICAL
    assert _score_to_tier_G15(0.0) == _Tier_G15.LOW


def test_score_to_tier_recommended_already_in_allowlist():
    tier = _score_to_tier_G15(50.0, allowed_tiers=["STANDARD", "ADVANCED"])
    assert tier == _Tier_G15.STANDARD


class TestRecommendTier:
    def test_simple_lookup_is_low(self):
        from mullet.core.router.complexity import recommend_tier, Tier
        tier = recommend_tier("what is the version?")
        assert tier == Tier.LOW

    def test_multi_file_refactor_is_standard_or_above(self):
        from mullet.core.router.complexity import recommend_tier, Tier
        tier = recommend_tier("refactor the auth module and update all callers", file_count=5)
        assert tier in (Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL)

    def test_security_task_above_low(self):
        from mullet.core.router.complexity import recommend_tier, Tier
        tier = recommend_tier("audit authentication and JWT token validation for security vulnerabilities")
        assert tier != Tier.LOW

    def test_file_count_escalates_tier(self):
        from mullet.core.router.complexity import recommend_tier, Tier
        tier_no_files = recommend_tier("update the config")
        tier_many_files = recommend_tier("update the config", file_count=10)
        # More files → same or higher tier
        tiers = [Tier.LOW, Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL]
        assert tiers.index(tier_many_files) >= tiers.index(tier_no_files)


class TestComplexityHelpers:
    """Direct unit tests for private helper functions."""

    def test_detect_domain_security(self):
        from mullet.core.router.complexity import _detect_domain
        assert _detect_domain("add jwt token authentication") == "security"

    def test_detect_domain_infra(self):
        from mullet.core.router.complexity import _detect_domain
        assert _detect_domain("deploy to kubernetes cluster") == "infrastructure"

    def test_detect_domain_general(self):
        from mullet.core.router.complexity import _detect_domain
        assert _detect_domain("update config values") == "general"

    def test_detect_question_depth_why(self):
        from mullet.core.router.complexity import _detect_question_depth
        assert _detect_question_depth("why does this fail?") == "why"

    def test_detect_question_depth_how(self):
        from mullet.core.router.complexity import _detect_question_depth
        assert _detect_question_depth("how does the router work?") == "how"

    def test_detect_question_depth_what(self):
        from mullet.core.router.complexity import _detect_question_depth
        assert _detect_question_depth("what is oauth?") == "what"

    def test_detect_question_depth_none(self):
        from mullet.core.router.complexity import _detect_question_depth
        assert _detect_question_depth("list the files") == "none"

    def test_count_subtasks_numbered_list(self):
        from mullet.core.router.complexity import _count_subtasks
        prompt = "1. Add auth\n2. Update DB\n3. Write tests"
        assert _count_subtasks(prompt) >= 3

    def test_count_subtasks_single_task(self):
        from mullet.core.router.complexity import _count_subtasks
        assert _count_subtasks("list files") >= 1

    def test_assess_reversibility_difficult(self):
        from mullet.core.router.complexity import _assess_reversibility
        assert _assess_reversibility("run migration and drop old table") == "difficult"

    def test_assess_reversibility_moderate(self):
        from mullet.core.router.complexity import _assess_reversibility
        assert _assess_reversibility("refactor the auth module") == "moderate"

    def test_assess_reversibility_easy(self):
        from mullet.core.router.complexity import _assess_reversibility
        assert _assess_reversibility("add a comment to the function") == "easy"

    def test_assess_impact_system_wide(self):
        from mullet.core.router.complexity import _assess_impact
        assert _assess_impact("update entire codebase to python 3.12", 0) == "system-wide"

    def test_assess_impact_function_level(self):
        from mullet.core.router.complexity import _assess_impact
        assert _assess_impact("fix this function", 1) == "function"

    def test_assess_impact_line_level(self):
        from mullet.core.router.complexity import _assess_impact
        assert _assess_impact("add a line to the config", 0) == "line"


def test_is_reasoning_task_detects_analysis_prompts():
    from mullet.core.router.complexity import is_reasoning_task
    assert is_reasoning_task("analyse the trade-offs between sync and async I/O")
    assert is_reasoning_task("review this implementation for correctness")
    assert is_reasoning_task("explain why this approach is better")
    assert not is_reasoning_task("list the files")
    assert not is_reasoning_task("fix the typo in README")


def test_simple_message_skips_type_floor():
    """is_simple_message prompts must not be floored to STANDARD by task-type table."""
    # We can't test _resolve_provider directly without the full MCP setup,
    # so test the helper behaviour directly.
    from mullet.core.router.complexity import is_simple_message
    assert is_simple_message("list the files in this directory")
    assert is_simple_message("what is 2+2")


# ---------------------------------------------------------------------------
# apply_speed_mode
# ---------------------------------------------------------------------------

from mullet.core.router.complexity import apply_speed_mode, speed_mode_badge


class TestApplySpeedMode:
    def test_economy_keeps_low(self):
        assert apply_speed_mode(Tier.LOW, "economy") == Tier.LOW

    def test_economy_keeps_critical(self):
        assert apply_speed_mode(Tier.CRITICAL, "economy") == Tier.CRITICAL

    def test_balanced_promotes_low_to_standard(self):
        assert apply_speed_mode(Tier.LOW, "balanced") == Tier.STANDARD

    def test_balanced_keeps_standard(self):
        assert apply_speed_mode(Tier.STANDARD, "balanced") == Tier.STANDARD

    def test_balanced_keeps_advanced(self):
        assert apply_speed_mode(Tier.ADVANCED, "balanced") == Tier.ADVANCED

    def test_fast_promotes_low_to_standard(self):
        assert apply_speed_mode(Tier.LOW, "fast") == Tier.STANDARD

    def test_fast_keeps_advanced(self):
        assert apply_speed_mode(Tier.ADVANCED, "fast") == Tier.ADVANCED

    def test_max_promotes_low_to_advanced(self):
        assert apply_speed_mode(Tier.LOW, "max") == Tier.ADVANCED

    def test_max_promotes_standard_to_advanced(self):
        assert apply_speed_mode(Tier.STANDARD, "max") == Tier.ADVANCED

    def test_max_keeps_critical(self):
        assert apply_speed_mode(Tier.CRITICAL, "max") == Tier.CRITICAL

    def test_unknown_mode_no_change(self):
        assert apply_speed_mode(Tier.LOW, "turbo") == Tier.LOW


class TestSpeedModeBadge:
    def test_economy_badge(self):
        assert speed_mode_badge("economy") == "[eco]"

    def test_balanced_badge(self):
        assert speed_mode_badge("balanced") == "[bal]"

    def test_fast_badge(self):
        assert speed_mode_badge("fast") == "[fast]"

    def test_max_badge(self):
        assert speed_mode_badge("max") == "[max]"

    def test_unknown_mode_wrapped_in_brackets(self):
        assert speed_mode_badge("turbo") == "[turbo]"


# ---------------------------------------------------------------------------
# confidence_breakdown
# ---------------------------------------------------------------------------

from mullet.core.router.complexity import (
    confidence_breakdown, ComplexitySignals, LexicalSignals, StructuralSignals, ContextSignals
)


class TestConfidenceBreakdown:
    def test_empty_signals_returns_zeros(self):
        signals = ComplexitySignals()
        result = confidence_breakdown(signals)
        assert result["total_score"] == 0.0
        assert result["lexical_signals"] == {}
        assert result["structural_signals"] == {}
        assert result["context_signals"] == {}

    def test_lexical_high_word_count(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(word_count=250)
        )
        result = confidence_breakdown(signals)
        assert "word_count_200+" in result["lexical_signals"]
        assert result["total_score"] > 0

    def test_lexical_architecture_keywords(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(has_architecture_keywords=True)
        )
        result = confidence_breakdown(signals)
        assert "architecture_keywords" in result["lexical_signals"]
        assert result["lexical_signals"]["architecture_keywords"] == 20

    def test_lexical_simple_keywords_negative(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(has_simple_keywords=True)
        )
        result = confidence_breakdown(signals)
        assert "simple_keywords" in result["lexical_signals"]
        assert result["lexical_signals"]["simple_keywords"] == -15

    def test_total_score_never_negative(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(has_simple_keywords=True)  # -15
        )
        result = confidence_breakdown(signals)
        assert result["total_score"] >= 0  # clamped to 0

    def test_structural_security_domain(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(domain_specificity="security")
        )
        result = confidence_breakdown(signals)
        assert "domain_security" in result["structural_signals"]
        assert result["structural_signals"]["domain_security"] == 20

    def test_structural_reversibility_difficult(self):
        signals = ComplexitySignals(
            structural=StructuralSignals(reversibility="difficult")
        )
        result = confidence_breakdown(signals)
        assert "reversibility_difficult" in result["structural_signals"]

    def test_context_previous_failures(self):
        signals = ComplexitySignals(
            context=ContextSignals(previous_failures=2)
        )
        result = confidence_breakdown(signals)
        assert "previous_failures_2+" in result["context_signals"]
        assert result["context_signals"]["previous_failures_2+"] == 20

    def test_context_plan_complexity_3plus(self):
        signals = ComplexitySignals(
            context=ContextSignals(plan_complexity=3)
        )
        result = confidence_breakdown(signals)
        assert "plan_complexity_3+" in result["context_signals"]

    def test_combined_signals_accumulate(self):
        signals = ComplexitySignals(
            lexical=LexicalSignals(has_architecture_keywords=True, has_risk_keywords=True),
            structural=StructuralSignals(domain_specificity="security"),
        )
        result = confidence_breakdown(signals)
        # architecture(20) + risk(20) + security(20) = 60
        assert result["total_score"] == 60.0


# ---------------------------------------------------------------------------
# Phase A — reasoning keyword scoring (TDD: RED first, then GREEN)
# ---------------------------------------------------------------------------

class TestPhaseAReasoningScoring:
    """New tests for Phase A: reasoning keyword bump + prose sophistication bump."""

    def test_realistic_prose_critique_reaches_standard_or_above(self):
        """A realistic plan-critique prompt (~60+ words, reasoning keyword, no code/paths)
        should score STANDARD or higher — was LOW before Phase A.
        """
        prompt = (
            "Please critique this implementation plan and analyse whether the approach "
            "is sound. The plan proposes migrating from a monolithic architecture to "
            "microservices. I need a thorough review of the trade-offs, risks, and "
            "whether the sequencing makes sense given our team size and current "
            "technical debt. Provide your honest assessment of the viability."
        )
        signals = extract_signals(prompt)
        score = score_complexity(signals)
        tier = score_to_tier(score)
        assert tier in (Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL), (
            f"Expected STANDARD or above for prose critique, got {tier} (score={score})"
        )

    def test_list_files_still_low(self):
        """Simple task must stay LOW after Phase A changes."""
        signals = extract_signals("list files")
        score = score_complexity(signals)
        tier = score_to_tier(score)
        assert tier == Tier.LOW, f"Expected LOW for 'list files', got {tier} (score={score})"

    def test_bare_two_word_reasoning_stays_low(self):
        """A bare 2-word 'critique this' must stay LOW.
        +15 bump alone is < 30 threshold — correct per spec.
        """
        signals = extract_signals("critique this")
        score = score_complexity(signals)
        tier = score_to_tier(score)
        assert tier == Tier.LOW, (
            f"Bare 'critique this' should stay LOW (score={score}), got {tier}"
        )

    def test_130_word_plain_prose_gets_prose_bump(self):
        """A 130-word plain prose analysis prompt (no code, no file paths) should
        get the +10 prose bump for the 120–200 word band.
        When combined with a reasoning keyword (+15), total must reach STANDARD.
        """
        # ~130 words, contains "analyse", no code blocks, no file paths
        words = "analyse " + ("the system performance characteristics and determine ") * 20
        # Trim to ~130 words
        word_list = words.split()[:130]
        prompt = " ".join(word_list)
        assert 120 < len(prompt.split()) <= 200, f"Word count out of band: {len(prompt.split())}"

        signals = extract_signals(prompt)
        score = score_complexity(signals)
        # +15 reasoning keyword + +10 prose bump = 25 base; may combine with other signals
        # Assert score includes prose bump contribution (score >= 25)
        assert score >= 25, f"Expected score >= 25 (reasoning + prose bump), got {score}"
        tier = score_to_tier(score)
        assert tier in (Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL), (
            f"130-word prose with reasoning keyword should reach STANDARD, got {tier} (score={score})"
        )

    def test_extract_signals_sets_has_reasoning_keywords_true(self):
        """extract_signals must set has_reasoning_keywords=True for prompts with keywords."""
        prompt = "please review this implementation for correctness"
        signals = extract_signals(prompt)
        assert signals.lexical.has_reasoning_keywords is True, (
            "Expected has_reasoning_keywords=True for prompt containing 'review'"
        )

    def test_extract_signals_sets_has_reasoning_keywords_false(self):
        """extract_signals must set has_reasoning_keywords=False when no reasoning keywords."""
        prompt = "list the files in the directory"
        signals = extract_signals(prompt)
        assert signals.lexical.has_reasoning_keywords is False, (
            "Expected has_reasoning_keywords=False for prompt without reasoning keywords"
        )
