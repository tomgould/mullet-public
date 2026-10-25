"""3-D routing matrix: (input_tokens × output_tokens × complexity) → model_id.

Model rosters (kept here for quick reference):

Kimi – see select_kimi_model() below.

Gemini:
| model_id                | $/1M in | $/1M out | max_context | notes      |
|-------------------------|---------|----------|-------------|------------|
| gemini-2.5-flash-lite   | $0.10   | $0.40    | 1048576     | ultra-cheap|
| gemini-2.5-flash        | $0.30   | $2.50    | 1048576     | preferred  |
| gemini-3.5-flash        | $1.50   | $9.00    | 1048576     | available (5x cost, not auto-picked) |
| gemini-3.1-pro          | $2.00   | $12.00   | 2000000     | preferred  |
| gemini-2.5-pro          | $2.00   | $12.00   | 1048576     | legacy     |

Note: gemini-3.5-flash is a real, verified model but NOT auto-selected — its
input/output cost is 5x/3.6x gemini-2.5-flash for a tier whose whole point is
being cheap. It stays addressable by explicit request. gemini-3.1-pro IS
promoted (same price as 2.5-pro, 2x context) since that's a cost-free upgrade.
Gemini generally sits behind Kimi/OpenRouter in provider priority — see
cost-class notes in select_openrouter_model() below.
"""
from __future__ import annotations
from mullet.core.router.preflight import PreflightResult
from mullet.core.router.complexity import Tier
from mullet.core.providers.value_picker_constants import (
    KIMI_VALUE_PICK_CANDIDATES,
    OPENROUTER_FREE_TIER_CANDIDATES,
    OPENROUTER_VALUE_PICK_CANDIDATES,
)
import logging

logger = logging.getLogger(__name__)

_LARGE_CONTEXT_THRESHOLD = 131_072   # above this, 131K models are ruled out
_HIGH_OUTPUT_THRESHOLD = 8_000       # above this, use premium-quality model

# Tier is a str enum — alphabetical order doesn't match severity order (ADVANCED < STANDARD).
# Use this rank map for all >= / > comparisons.
_TIER_RANK: dict[str, int] = {"LOW": 0, "STANDARD": 1, "ADVANCED": 2, "CRITICAL": 3}


def _tier_gte(tier: Tier, threshold: Tier) -> bool:
    return _TIER_RANK.get(tier.value, 0) >= _TIER_RANK.get(threshold.value, 0)


_GEMINI_HIGH_OUTPUT_THRESHOLD = 8_000     # above this, use pro
_GEMINI_DEPRECATED = {
    "gemini-2.0-flash": "2026-06-01",
    "gemini-2.0-flash-lite": "2026-06-01",
}


def _kimi_value_pick(
    preflight: PreflightResult,
    speed_mode: str,
) -> tuple[str, str] | None:
    """routing-v3 Phase 3b: try the value-based picker for the tier-based
    branches of select_kimi_model.

    Only ever consulted from the elif chain below -- i.e. never when
    preflight.needs_reasoning is True, since that early-return in
    select_kimi_model happens first and is untouched by this function.
    Returns None (never raises) on any exception, on a None pick from
    pick_by_value, or when an existing static override (speed_mode="max",
    high expected output) should take precedence -- the caller then falls
    through to the existing static ladder unchanged.
    """
    try:
        from mullet.core.providers.value_picker import pick_by_value
        candidates = KIMI_VALUE_PICK_CANDIDATES
        domain = "coding"
        tier = preflight.complexity_tier.value
        value_pick = pick_by_value(
            candidates, tier, preflight.input_tokens, domain,
            context_tokens=preflight.input_tokens,
            prefer_reasoning=False,
        )
        if value_pick is not None and speed_mode != "max":
            # speed_mode="max" is an explicit user override for latency that
            # the existing ladder already special-cases (max -> k2.7-code
            # for STANDARD+ tiers) -- respect that override rather than
            # letting the value picker second-guess an explicit speed
            # request. Also preserve the existing high-output heuristic
            # (k2.7-code for >8K expected output) since the value picker
            # lacks output-volume signal and would otherwise under-pick on
            # quality for long generations.
            if preflight.expected_output_tokens <= _HIGH_OUTPUT_THRESHOLD:
                return value_pick, f"value picker: domain={domain} tier={tier}"
    except Exception as exc:
        logger.warning(
            "select_kimi_model: value picker failed (%s) -- falling back to static ladder",
            exc,
        )
    return None


def select_kimi_model(
    preflight: PreflightResult,
    *,
    speed_mode: str = "balanced",
) -> tuple[str, str]:
    """Return (model_id, human-readable reason) for this preflight result."""
    if preflight.needs_reasoning:
        # kimi-k2-thinking* discontinued 2026-05-25 → K2.7-Code has integrated
        # reasoning with ~30% fewer reasoning tokens than K2.6.
        model_id = "kimi-k2.7-code"
        reason = "reasoning task → kimi-k2.7-code (thinking series discontinued)"
    elif (_value_pick := _kimi_value_pick(preflight, speed_mode)) is not None:
        model_id, reason = _value_pick
    elif speed_mode == "max" and _tier_gte(preflight.complexity_tier, Tier.STANDARD):
        # kimi-k2-turbo-preview discontinued 2026-05-25 → kimi-k2.7-code
        model_id = "kimi-k2.7-code"
        reason = "max speed — kimi-k2.7-code (turbo-preview discontinued)"
    elif preflight.expected_output_tokens > _HIGH_OUTPUT_THRESHOLD:
        model_id = "kimi-k2.7-code"
        reason = "high output volume — premium model for quality"
    elif preflight.complexity_tier == Tier.CRITICAL:
        model_id = "kimi-k2.7-code"
        reason = "critical complexity"
    elif preflight.complexity_tier == Tier.ADVANCED:
        model_id = "kimi-k2.7-code"
        reason = "advanced task — K2.7-Code (thinking ON, same cost as K2.6, better quality)"
    elif preflight.input_tokens > _LARGE_CONTEXT_THRESHOLD:
        model_id = "kimi-k2.5"
        reason = "large context, standard model"
    else:
        model_id = "kimi-k2.5"
        reason = "standard task"

    from mullet.core.providers.kimi import _DEPRECATED_KIMI_MODELS
    if model_id in _DEPRECATED_KIMI_MODELS:
        replacement, eol_date = _DEPRECATED_KIMI_MODELS[model_id]
        logger.warning(
            "routing_matrix selected deprecated Kimi model %r (EOL %s) — "
            "using %r instead. This should not happen; check routing logic.",
            model_id, eol_date, replacement,
        )
        model_id = replacement

    logger.debug(
        "routing_matrix: %s → %s (%s) [in=%d out=%d tier=%s reason=%s speed=%s]",
        "kimi", model_id, model_id, preflight.input_tokens, preflight.expected_output_tokens,
        preflight.complexity_tier.value, reason, speed_mode,
    )

    return model_id, reason


def select_gemini_model(
    preflight: PreflightResult,
    *,
    speed_mode: str = "balanced",
) -> tuple[str, str]:
    """Return (model_id, reason) for the optimal Gemini model given this preflight."""
    # routing-v3 Phase 3 (Gemini only): try the value-based picker first.
    # Any exception or None result falls through to the existing static
    # if/elif ladder below -- this must never break routing, only improve it.
    try:
        from mullet.core.providers.value_picker import pick_by_value
        candidates: list[tuple[str, int]] = [
            ("gemini-3.1-pro",        2_000_000),
            ("gemini-2.5-pro",        1_000_000),
            ("gemini-3.5-flash",      1_048_576),
            ("gemini-2.5-flash",      1_000_000),
            ("gemini-2.5-flash-lite", 1_000_000),
        ]
        domain = "reasoning" if preflight.needs_reasoning else "coding"
        tier = preflight.complexity_tier.value
        value_pick = pick_by_value(
            candidates, tier, preflight.input_tokens, domain,
            context_tokens=preflight.input_tokens,
            prefer_reasoning=preflight.needs_reasoning,
        )
        if value_pick is not None and speed_mode not in ("max", "economy"):
            # speed_mode="max"/"economy" are explicit user overrides for
            # latency/cost that the existing ladder already special-cases
            # (max -> flash for lowest latency, economy -> flash-lite) --
            # respect those overrides rather than letting the value picker
            # second-guess an explicit speed request.
            # Also preserve the existing high-output heuristic (pro for >8K
            # expected output) since the value picker lacks output-volume
            # signal and would otherwise under-pick on quality for long
            # generations.
            if preflight.expected_output_tokens <= _GEMINI_HIGH_OUTPUT_THRESHOLD:
                return value_pick, f"value picker: domain={domain} tier={tier}"
    except Exception as exc:
        logger.warning(
            "select_gemini_model: value picker failed (%s) -- falling back to static ladder",
            exc,
        )

    if speed_mode == "max":
        model_id = "gemini-2.5-flash"
        reason = "max speed — flash for lowest latency"
    elif preflight.complexity_tier == Tier.CRITICAL or preflight.expected_output_tokens > _GEMINI_HIGH_OUTPUT_THRESHOLD:
        model_id = "gemini-3.1-pro"
        reason = "critical task or high output volume"
    elif preflight.complexity_tier == Tier.ADVANCED:
        model_id = "gemini-2.5-flash"
        reason = "advanced task"
    elif speed_mode == "economy" or preflight.complexity_tier == Tier.LOW:
        model_id = "gemini-2.5-flash-lite"
        reason = "low complexity / economy mode — cheapest model"
    else:
        model_id = "gemini-2.5-flash"
        reason = "standard task"

    logger.debug(
        "routing_matrix: %s → %s (%s) [in=%d out=%d tier=%s reason=%s speed=%s]",
        "gemini", model_id, model_id, preflight.input_tokens, preflight.expected_output_tokens,
        preflight.complexity_tier.value, reason, speed_mode,
    )

    return model_id, reason


def select_openrouter_model(
    preflight: PreflightResult,
    *,
    speed_mode: str = "balanced",
) -> tuple[str, str]:
    """Return (model_id, reason) for the optimal OpenRouter model given this preflight.

    Routing priorities (verified 2026-07-05):
      - needs_reasoning  → deepseek-v4-pro ($0.435/$0.87) — strong reasoning, cheap.
                           speed_mode=premium promotes to gpt-5 / claude-opus-4.8.
      - input >500K tok  → qwen3-coder (1M ctx, $0.22/$1.80) or
                           claude-sonnet-5 (1M ctx, $2/$10) in premium mode.
      - CRITICAL         → claude-opus-4.8 (premium) or claude-sonnet-5 (balanced).
      - ADVANCED         → deepseek-v4-pro (premium claude-sonnet-5) — V4 Pro is ~3x
                           cheaper than qwen3.7-max; the benchmark edge didn't justify
                           the cost for a non-premium tier. qwen3.7-max stays available.
      - STANDARD/LOW     → deepseek-v4-flash ($0.098/$0.196) — best $/quality, 1M ctx.
    """
    is_premium = speed_mode in ("max", "premium")

    # Free-tier-first for LOW complexity: try genuinely-free OpenRouter models
    # before the paid value-picker ladder below. Scoped tightly to LOW only --
    # free models have no SLA (OpenRouter: shared/best-effort capacity, 20
    # req/min + daily quota, provider-level throttling during peak times) so
    # this must never leak into STANDARD/ADVANCED/CRITICAL where reliability
    # matters more. Any exception or None result falls through unchanged to
    # the existing paid logic below -- this must never break routing, only
    # add a cheaper option at the bottom tier.
    if preflight.complexity_tier == Tier.LOW and not preflight.needs_reasoning:
        try:
            from mullet.core.providers.model_availability import without_unavailable
            from mullet.core.providers.value_picker import pick_by_value
            free_pick = pick_by_value(
                without_unavailable("openrouter", OPENROUTER_FREE_TIER_CANDIDATES),
                "LOW", preflight.input_tokens, "coding",
                context_tokens=preflight.input_tokens, prefer_reasoning=False,
            )
            if free_pick is not None:
                return free_pick, "free-tier pick for LOW complexity"
        except Exception as exc:
            logger.warning(
                "select_openrouter_model: free-tier pick failed (%s) -- falling back to paid ladder",
                exc,
            )

    # routing-v3 Phase 3a (OpenRouter): try the value-based picker first for
    # the balanced-mode, non-reasoning, non-large-context path. Any exception
    # or None result falls through to the existing static if/elif ladder
    # below unchanged -- this must never break routing, only improve it.
    # Skipped entirely when: needs_reasoning (routed via the reasoning branch
    # below -- only deepseek/deepseek-v4-pro is reasoning-tuned among the 8
    # candidates), input >500K tokens (a context-driven choice the 8-candidate
    # picker doesn't model), or an explicit premium/max speed_mode request for
    # CRITICAL/ADVANCED (those overrides are preserved untouched, mirroring
    # Gemini's explicit speed-override preservation).
    if (
        not preflight.needs_reasoning
        and preflight.input_tokens <= 500_000
        and not (is_premium and preflight.complexity_tier in (Tier.CRITICAL, Tier.ADVANCED))
    ):
        try:
            from mullet.core.providers.model_availability import without_unavailable
            from mullet.core.providers.value_picker import pick_by_value
            candidates = without_unavailable("openrouter", OPENROUTER_VALUE_PICK_CANDIDATES)
            tier = preflight.complexity_tier.value
            value_pick = pick_by_value(
                candidates, tier, preflight.input_tokens, "coding",
                context_tokens=preflight.input_tokens, prefer_reasoning=False,
            )
            if value_pick is not None:
                return value_pick, f"value picker: domain=coding tier={tier}"
        except Exception as exc:
            logger.warning(
                "select_openrouter_model: value picker failed (%s) -- falling back to static ladder",
                exc,
            )

    if preflight.needs_reasoning:
        if is_premium:
            model_id = "openai/gpt-5"
            reason = "reasoning, premium tier"
        else:
            model_id = "deepseek/deepseek-v4-pro"
            reason = "reasoning, chain-of-thought"
    elif preflight.input_tokens > 500_000:
        if is_premium:
            model_id = "anthropic/claude-sonnet-5"
            reason = "very large context, premium quality (1M ctx)"
        else:
            model_id = "qwen/qwen3-coder"
            reason = "very large context, cheap (1M ctx, code-tuned)"
    elif preflight.complexity_tier == Tier.CRITICAL:
        if is_premium:
            model_id = "anthropic/claude-opus-4.8"
            reason = "critical task, premium synthesis"
        else:
            model_id = "anthropic/claude-sonnet-5"
            reason = "critical task, balanced (1M ctx)"
    elif preflight.complexity_tier == Tier.ADVANCED:
        if is_premium:
            model_id = "anthropic/claude-sonnet-5"
            reason = "advanced task, premium"
        else:
            model_id = "deepseek/deepseek-v4-pro"
            reason = "advanced task — DeepSeek V4 Pro (1M ctx, ~3x cheaper than qwen3.7-max)"
    elif preflight.input_tokens > 100_000:
        model_id = "deepseek/deepseek-v4-flash"
        reason = "large context — DeepSeek V4 Flash (1M ctx, cheapest)"
    else:
        model_id = "deepseek/deepseek-v4-flash"
        reason = "standard task — DeepSeek V4 Flash, best $/quality"

    logger.debug(
        "routing_matrix: %s → %s (%s) [in=%d out=%d tier=%s reason=%s speed=%s]",
        "openrouter", model_id, model_id, preflight.input_tokens,
        preflight.expected_output_tokens, preflight.complexity_tier.value,
        reason, speed_mode,
    )
    return model_id, reason


# Patterns for classifying local models by parameter count (from model ID string)
_LOCAL_EMBEDDING_PATTERNS = ("embed", "embedding", "nomic", "bge-", "e5-")
_LOCAL_TINY_PATTERNS = ("0.5b", "1b", "1.5b", "2b", "3b")      # analysis only
_LOCAL_SMALL_PATTERNS = ("4b", "6b", "7b", "8b")                # standard tasks
_LOCAL_MEDIUM_PATTERNS = ("9b", "10b", "11b", "12b", "13b", "14b")  # advanced
_LOCAL_LARGE_PATTERNS = (
    "20b", "21b", "22b", "24b", "27b", "30b", "32b", "34b", "35b",
    "70b", "72b", "80b", "106b", "120b", "235b", "480b",
)  # critical — includes modern MoE total sizes (e.g. 35B-A3B, 80B-A3B)
_LOCAL_CODER_PATTERNS = ("coder", "code", "starcoder", "deepseek-coder", "codellama", "ornith")
_LOCAL_VISION_PATTERNS = ("vl", "vision", "internvl", "llava", "moondream")
_LOCAL_REASONING_PATTERNS = ("thinking", "reasoning", "r1", "qwq", "distillation")

# Known parameter counts for model IDs that lack a clean size token.
_KNOWN_LOCAL_PARAMS: dict[str, float] = {"deepseek-coder-v2-lite": 16.0}  # 16B MoE


def _classify_local_model(model_id: str) -> dict:
    """Return capability profile for a local model based on its ID string."""
    import re
    mid = model_id.lower()
    # Skip embedding models
    if any(p in mid for p in _LOCAL_EMBEDDING_PATTERNS):
        return {"tier": None, "coder": False, "vision": False, "reasoning": False}

    is_coder = any(p in mid for p in _LOCAL_CODER_PATTERNS)
    is_vision = any(p in mid for p in _LOCAL_VISION_PATTERNS)
    is_reasoning = any(p in mid for p in _LOCAL_REASONING_PATTERNS)

    # Strip the MoE active-parameter suffix (e.g. "-a3b", "-a22b") before tier
    # matching so total size drives the tier — otherwise "qwen3.6-35b-a3b" would
    # false-match the "3b" tiny pattern and be misrouted to LOW.
    sized = re.sub(r"[-_.]a\d+(?:\.\d+)?b\b", "", mid)

    if any(p in sized for p in _LOCAL_LARGE_PATTERNS):
        tier = "CRITICAL"
    elif any(p in sized for p in _LOCAL_MEDIUM_PATTERNS):
        tier = "ADVANCED"
    elif any(p in sized for p in _LOCAL_SMALL_PATTERNS):
        tier = "STANDARD"
    elif any(p in sized for p in _LOCAL_TINY_PATTERNS):
        tier = "LOW"
    else:
        tier = "STANDARD"  # unknown size — assume standard

    return {"tier": tier, "coder": is_coder, "vision": is_vision, "reasoning": is_reasoning}


def _local_param_count(model_id: str) -> float:
    """Extract approximate parameter count (in billions) from a model ID string.
    Returns 7.0 as default when not parseable."""
    import re
    mid = model_id.lower()
    for key, params in _KNOWN_LOCAL_PARAMS.items():
        if key in mid:
            return params
    m = re.search(r'(\d+(?:\.\d+)?)\s*b\b', mid)
    return float(m.group(1)) if m else 7.0


def select_local_model(
    preflight: PreflightResult,
    available_models: list[str],
    *,
    speed_mode: str = "balanced",
) -> tuple[str, str] | tuple[None, str]:
    """Return (model_id, reason) for the best available local model, or (None, reason) if none fits.

    Works on any system — classifies models by inferring capability from model ID strings.
    """
    # 1. Filter out embedding models and classify the rest.
    usable = [(m, _classify_local_model(m)) for m in available_models]
    usable = [(m, c) for m, c in usable if c["tier"] is not None]

    # 2. If no usable models.
    if not usable:
        return None, "no usable local models available"

    # 3. If preflight.needs_reasoning: prefer models with is_reasoning=True.
    if preflight.needs_reasoning:
        reasoning_models = [(m, c) for m, c in usable if c["reasoning"]]
        if reasoning_models:
            reasoning_models.sort(
                key=lambda mc: (-_TIER_RANK.get(mc[1]["tier"], 0), _local_param_count(mc[0]))
            )
            model_id = reasoning_models[0][0]
            reason = "reasoning task — local reasoning model"
            logger.debug(
                "routing_matrix: %s → %s (%s) [in=%d out=%d tier=%s reason=%s speed=%s]",
                "local", model_id, model_id, preflight.input_tokens, preflight.expected_output_tokens,
                preflight.complexity_tier.value, reason, speed_mode,
            )
            return model_id, reason

    # 5. Select by tier matching against preflight.complexity_tier.
    tier_val = preflight.complexity_tier.value
    if tier_val == "LOW":
        target_tiers = ["LOW", "STANDARD"]
    elif tier_val == "STANDARD":
        target_tiers = ["STANDARD", "ADVANCED"]
    elif tier_val == "ADVANCED":
        target_tiers = ["ADVANCED", "STANDARD"]
    elif tier_val == "CRITICAL":
        target_tiers = ["ADVANCED", "CRITICAL"]
    else:
        target_tiers = ["STANDARD"]

    candidates = [(m, c) for m, c in usable if c["tier"] in target_tiers]

    if tier_val == "CRITICAL" and not candidates:
        return None, "local models too small for critical task"

    if not candidates:
        candidates = usable

    # 4. If task is code-related: prefer coder models at appropriate tier.
    is_code_task = _tier_gte(preflight.complexity_tier, Tier.STANDARD) and any(
        c["coder"] for m, c in candidates
    )
    if is_code_task:
        coder_candidates = [(m, c) for m, c in candidates if c["coder"]]
        if coder_candidates:
            candidates = coder_candidates

    # 5. Deprioritise reasoning models for non-reasoning tasks.
    if not preflight.needs_reasoning:
        non_reasoning = [(m, c) for m, c in candidates if not c["reasoning"]]
        if non_reasoning:
            candidates = non_reasoning

    # 6. Pick by tier intent. Higher tiers (ADVANCED/CRITICAL) want the most
    #    capable model that qualifies — pick the LARGEST so the heavy local
    #    coders (e.g. 14B, 30B-A3B) actually get used. Lower tiers (LOW/STANDARD)
    #    want the smallest for speed/efficiency.
    candidates.sort(key=lambda mc: _local_param_count(mc[0]))
    if _tier_gte(preflight.complexity_tier, Tier.ADVANCED):
        model_id, profile = candidates[-1]
    else:
        model_id, profile = candidates[0]

    if is_code_task and profile["coder"]:
        reason = f"code task — local {profile['tier'].lower()} coder model"
    else:
        reason = f"local {profile['tier'].lower()} model"

    logger.debug(
        "routing_matrix: %s → %s (%s) [in=%d out=%d tier=%s reason=%s speed=%s]",
        "local", model_id, model_id, preflight.input_tokens, preflight.expected_output_tokens,
        preflight.complexity_tier.value, reason, speed_mode,
    )

    return model_id, reason


def estimated_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Rough cost estimate using KNOWN_PRICING from model_registry.

    Works for any model_id present in the registry (Kimi, Gemini, etc.).
    """
    from mullet.core.providers.model_registry import KNOWN_PRICING

    pricing = KNOWN_PRICING.get(model_id)
    if pricing is None:
        return 0.0
    input_rate, output_rate = pricing
    return (input_tokens * input_rate + output_tokens * output_rate) / 1000.0
