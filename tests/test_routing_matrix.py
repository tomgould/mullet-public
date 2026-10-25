"""Tests for mullet.core.router.routing_matrix model selector."""
import pytest
from mullet.core.router.routing_matrix import select_kimi_model, select_gemini_model, select_local_model
from mullet.core.router.preflight import PreflightResult
from mullet.core.router.complexity import Tier


def make_preflight(
    input_tokens=1000,
    expected_output_tokens=500,
    complexity_tier=Tier.STANDARD,
    needs_reasoning=False,
    speed_mode="balanced",
    confidence=0.9,
    analyzer="local_llm",
) -> PreflightResult:
    return PreflightResult(
        input_tokens=input_tokens,
        expected_output_tokens=expected_output_tokens,
        complexity_tier=complexity_tier,
        needs_reasoning=needs_reasoning,
        confidence=confidence,
        analyzer=analyzer,
        latency_ms=50,
        cached=False,
    )


# ---------------------------------------------------------------------------
# select_kimi_model — parametrised
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expected_model", [
    # Reasoning → kimi-k2.7-code (K2.7 over K2.6; thinking* discontinued 2026-05-25)
    ({"needs_reasoning": True}, "kimi-k2.7-code"),
    ({"needs_reasoning": True, "speed_mode": "fast"}, "kimi-k2.7-code"),
    ({"needs_reasoning": True, "speed_mode": "max"}, "kimi-k2.7-code"),
    # Max speed → kimi-k2.7-code (kimi-k2-turbo-preview discontinued 2026-05-25)
    ({"complexity_tier": Tier.STANDARD, "speed_mode": "max"}, "kimi-k2.7-code"),
    ({"complexity_tier": Tier.ADVANCED, "speed_mode": "max"}, "kimi-k2.7-code"),
    # High output → k2.7-code
    ({"expected_output_tokens": 10000}, "kimi-k2.7-code"),
    # Critical → k2.7-code
    ({"complexity_tier": Tier.CRITICAL}, "kimi-k2.7-code"),
    # Advanced → k2.7-code (same price as k2.6 but thinking always ON; better quality)
    ({"complexity_tier": Tier.ADVANCED}, "kimi-k2.7-code"),
    # Large context → kimi-k2.5
    ({"input_tokens": 200_000}, "kimi-k2.5"),
    # Default low/standard → kimi-k2.5
    ({}, "kimi-k2.5"),
    ({"complexity_tier": Tier.LOW}, "kimi-k2.5"),
])
def test_select_kimi_model(kwargs, expected_model):
    pf = make_preflight(**{k: v for k, v in kwargs.items() if k != "speed_mode"})
    speed = kwargs.get("speed_mode", "balanced")
    model_id, reason = select_kimi_model(pf, speed_mode=speed)
    assert model_id == expected_model, f"Expected {expected_model!r}, got {model_id!r} (reason: {reason})"
    assert isinstance(reason, str) and len(reason) > 0


# ---------------------------------------------------------------------------
# select_gemini_model — parametrised
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expected_model", [
    # Max speed → flash
    ({"speed_mode": "max"}, "gemini-2.5-flash"),
    # Critical → pro
    ({"complexity_tier": Tier.CRITICAL}, "gemini-3.1-pro"),
    # High output → pro
    ({"expected_output_tokens": 10000}, "gemini-3.1-pro"),
    # Advanced → pro (routing-v3 value picker: ADVANCED quality floor=55 is
    # only cleared by gemini-3.1-pro (coding=67); this aligns select_gemini_model
    # with _best_gemini's static fallback which already mapped ADVANCED→pro)
    ({"complexity_tier": Tier.ADVANCED}, "gemini-3.1-pro"),
    # Economy → flash-lite
    ({"speed_mode": "economy"}, "gemini-2.5-flash-lite"),
    # LOW → flash-lite
    ({"complexity_tier": Tier.LOW}, "gemini-2.5-flash-lite"),
    # Default STANDARD → flash
    ({}, "gemini-2.5-flash"),
])
def test_select_gemini_model(kwargs, expected_model):
    pf = make_preflight(**{k: v for k, v in kwargs.items() if k != "speed_mode"})
    speed = kwargs.get("speed_mode", "balanced")
    model_id, reason = select_gemini_model(pf, speed_mode=speed)
    assert model_id == expected_model, f"Expected {expected_model!r}, got {model_id!r} (reason: {reason})"


# ---------------------------------------------------------------------------
# estimated_cost_usd
# ---------------------------------------------------------------------------

def test_estimated_cost_usd_kimi():
    from mullet.core.router.routing_matrix import estimated_cost_usd
    cost = estimated_cost_usd("kimi-k2.6", input_tokens=1000, output_tokens=500)
    assert cost > 0.0


def test_estimated_cost_usd_unknown_model():
    from mullet.core.router.routing_matrix import estimated_cost_usd
    cost = estimated_cost_usd("nonexistent-model", input_tokens=1000, output_tokens=500)
    assert cost == 0.0


# ── select_local_model tests ──────────────────────────────────────────────────

SAMPLE_MODELS = [
    "qwen3.5-2b-kimi-and-opus-distillation",   # tiny, reasoning
    "qwen2.5-coder-3b-instruct",               # tiny, coder
    "qwen2.5-coder-7b-instruct",               # small, coder
    "qwen/qwen3.5-9b",                         # medium, general
    "text-embedding-nomic-embed-text-v1.5",    # embedding — excluded
]


def test_select_local_no_models():
    pf = make_preflight()
    model, reason = select_local_model(pf, [])
    assert model is None


def test_select_local_excludes_embedding_only():
    pf = make_preflight()
    model, reason = select_local_model(pf, ["text-embedding-nomic-embed-text-v1.5"])
    assert model is None


def test_select_local_low_tier_picks_smallest():
    pf = make_preflight(complexity_tier=Tier.LOW)
    model, reason = select_local_model(pf, SAMPLE_MODELS)
    assert model is not None
    # Should pick a tiny/small model — param count ≤ 3B
    from mullet.core.router.routing_matrix import _local_param_count
    assert _local_param_count(model) <= 3.0


def test_select_local_reasoning_prefers_reasoning_model():
    pf = make_preflight(needs_reasoning=True)
    model, reason = select_local_model(pf, SAMPLE_MODELS)
    # qwen3.5-2b-kimi-and-opus-distillation has "distillation" → reasoning
    assert model is not None
    assert "distillation" in model or "reasoning" in model or "thinking" in model


def test_select_local_advanced_picks_larger_model():
    pf = make_preflight(complexity_tier=Tier.ADVANCED)
    model, reason = select_local_model(pf, SAMPLE_MODELS)
    from mullet.core.router.routing_matrix import _local_param_count
    assert model is not None
    # Should prefer 9B over 2-3B for ADVANCED
    assert _local_param_count(model) >= 7.0


def test_select_local_critical_with_no_large_model_returns_none():
    # All models are tiny — CRITICAL should return None (too small)
    tiny_models = ["qwen3.5-2b-test", "qwen2.5-3b-test"]
    pf = make_preflight(complexity_tier=Tier.CRITICAL)
    model, reason = select_local_model(pf, tiny_models)
    assert model is None


# Real June-2026 local arsenal (lms keys) for tier-sizing assertions.
LOCAL_ARSENAL = [
    "qwen3-coder-30b-a3b-instruct",
    "qwen2.5-coder-14b-instruct",
    "qwen2.5-coder-7b-instruct",
    "qwen3.5-9b",
    "qwen3-4b-instruct-2507",
    "qwen3-4b-thinking-2507",
    "qwen3-vl-8b-instruct",
]


def test_local_standard_code_picks_small_coder():
    pf = make_preflight(complexity_tier=Tier.STANDARD)
    model, _ = select_local_model(pf, LOCAL_ARSENAL)
    assert model == "qwen2.5-coder-7b-instruct"


def test_local_advanced_code_picks_14b_coder():
    """ADVANCED must escalate to the 14B coder, not stay on the 7B."""
    pf = make_preflight(complexity_tier=Tier.ADVANCED)
    model, _ = select_local_model(pf, LOCAL_ARSENAL)
    assert model == "qwen2.5-coder-14b-instruct"


def test_local_critical_code_picks_30b_coder():
    """CRITICAL must use the heavy 30B-A3B coder — the whole point of having it."""
    pf = make_preflight(complexity_tier=Tier.CRITICAL)
    model, _ = select_local_model(pf, LOCAL_ARSENAL)
    assert model == "qwen3-coder-30b-a3b-instruct"


def test_local_reasoning_prefers_fast_small_thinking():
    """Local reasoning stays on the fast 4B-thinking model."""
    pf = make_preflight(needs_reasoning=True)
    model, _ = select_local_model(pf, LOCAL_ARSENAL)
    assert model == "qwen3-4b-thinking-2507"


def test_classify_local_model_embedding_returns_none_tier():
    from mullet.core.router.routing_matrix import _classify_local_model
    result = _classify_local_model("text-embedding-nomic-embed-text-v1.5")
    assert result["tier"] is None


def test_local_param_count_extracts_correctly():
    from mullet.core.router.routing_matrix import _local_param_count
    assert _local_param_count("qwen2.5-7b-instruct") == 7.0
    assert _local_param_count("llama-3-70b") == 70.0
    assert _local_param_count("unknown-model") == 7.0  # default


@pytest.mark.parametrize("model_id,expected_tier", [
    # Modern MoE names: total size drives the tier, not the "aNb" active suffix.
    ("qwen3.6-35b-a3b", "CRITICAL"),
    ("qwen3-coder-next-80b-a3b", "CRITICAL"),
    ("qwen3.5-27b", "CRITICAL"),
    ("devstral-small-24b", "CRITICAL"),
    # Regressions — existing conventions still classify correctly.
    ("qwen2.5-coder-14b-instruct", "ADVANCED"),
    ("qwen2.5-coder-7b-instruct", "STANDARD"),
    ("qwen3.5-2b-distillation", "LOW"),
])
def test_classify_local_model_modern_moe_sizes(model_id, expected_tier):
    from mullet.core.router.routing_matrix import _classify_local_model
    assert _classify_local_model(model_id)["tier"] == expected_tier


def test_classify_local_moe_suffix_preserves_coder_flag():
    from mullet.core.router.routing_matrix import _classify_local_model
    c = _classify_local_model("qwen3-coder-next-80b-a3b")
    assert c["tier"] == "CRITICAL"
    assert c["coder"] is True


@pytest.mark.parametrize("model_id,expected_tier", [
    ("ornith-1.5-35b-a3b", "CRITICAL"),
    ("ornith-1.5-9b", "ADVANCED"),
])
def test_classify_local_ornith_is_a_coder_family(model_id, expected_tier):
    from mullet.core.router.routing_matrix import _classify_local_model
    c = _classify_local_model(model_id)
    assert c["coder"] is True
    assert c["tier"] == expected_tier


def test_select_local_code_task_picks_ornith_as_coder():
    """A CRITICAL code task keeps the Ornith MoE over a same-size non-coder."""
    pf = make_preflight(complexity_tier=Tier.CRITICAL)
    model, reason = select_local_model(pf, ["ornith-1.5-35b-a3b", "qwen3.6-35b-a3b"])
    assert model == "ornith-1.5-35b-a3b"
    assert "coder" in reason.lower()


# ---------------------------------------------------------------------------
# FIX A/B regression tests — reasoning deprioritisation & known param counts
# ---------------------------------------------------------------------------

def test_select_local_non_reasoning_task_prefers_coder_over_reasoning_model():
    """Non-reasoning STANDARD task must pick the plain coder, not a reasoning model."""
    pf = make_preflight(complexity_tier=Tier.STANDARD, needs_reasoning=False)
    models = [
        "qwen3-4b-thinking-2507",        # reasoning
        "qwen2.5-coder-7b-instruct",     # plain coder
    ]
    model, reason = select_local_model(pf, models)
    assert model == "qwen2.5-coder-7b-instruct"
    assert "coder" in reason.lower()


def test_select_local_non_reasoning_advanced_prefers_coder_over_reasoning_model():
    """Non-reasoning ADVANCED task must pick the plain coder, not a reasoning model."""
    pf = make_preflight(complexity_tier=Tier.ADVANCED, needs_reasoning=False)
    models = [
        "qwen3-4b-thinking-2507",        # reasoning
        "qwen2.5-coder-14b-instruct",    # plain coder
    ]
    model, reason = select_local_model(pf, models)
    assert model == "qwen2.5-coder-14b-instruct"
    assert "coder" in reason.lower()


def test_select_local_reasoning_task_still_picks_reasoning_model():
    """needs_reasoning=True must still prefer a reasoning model (regression guard)."""
    pf = make_preflight(needs_reasoning=True)
    models = [
        "qwen3-4b-thinking-2507",        # reasoning
        "qwen2.5-coder-7b-instruct",     # plain coder
    ]
    model, reason = select_local_model(pf, models)
    assert model == "qwen3-4b-thinking-2507"
    assert "reasoning" in reason.lower()


def test_select_local_non_reasoning_only_reasoning_models_fallback():
    """If only reasoning models are available for a non-reasoning task, do not empty the set."""
    pf = make_preflight(complexity_tier=Tier.STANDARD, needs_reasoning=False)
    models = [
        "qwen3-4b-thinking-2507",
    ]
    model, reason = select_local_model(pf, models)
    assert model is not None
    assert "local" in reason.lower()


def test_local_param_count_known_model_without_size_token():
    """_KNOWN_LOCAL_PARAMS lookup must work when no clean size token is present."""
    from mullet.core.router.routing_matrix import _local_param_count
    assert _local_param_count("deepseek-coder-v2-lite-instruct") == 16.0
    assert _local_param_count("deepseek-coder-v2-lite") == 16.0


def test_local_param_count_regex_fallback_still_works():
    """Existing regex extraction must continue to work for normal model names."""
    from mullet.core.router.routing_matrix import _local_param_count
    assert _local_param_count("qwen2.5-coder-14b-instruct") == 14.0
    assert _local_param_count("qwen2.5-coder-7b-instruct") == 7.0
