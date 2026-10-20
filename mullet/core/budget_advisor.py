"""Budget Advisor — pre-flight lane recommendations for autonomous loops.

Wraps check_budget() from the HUD guard and produces a human-readable
lane recommendation (premium/max/balanced/economy/fast) with estimated
cost and a formatted preflight banner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mullet.hud.guard import check_budget

Lane = Literal["premium", "max", "balanced", "economy", "fast"]

def _get_pricing():
    from mullet.core.providers.model_registry import KNOWN_PRICING
    return KNOWN_PRICING


def _get_provider_balances() -> dict:
    """Fetch cached provider balances — fast (5-min cache), never blocks routing."""
    try:
        from mullet.core.providers.budget_probe import get_all_balances
        return get_all_balances()
    except Exception:
        return {}

# Lane → (primary, fallback)
_LANE_PROVIDERS: dict[Lane, tuple[str, str]] = {
    "premium":  ("opus",   "sonnet"),
    "max":      ("sonnet", "haiku"),
    "balanced": ("sonnet", "haiku"),
    "fast":     ("haiku",  "kimi"),
    "economy":  ("kimi",   "local"),
}

# Map model/tier names to budget-probe provider keys for balance checks
_MODEL_TO_PROVIDER_KEY: dict[str, str] = {
    "opus": "anthropic", "sonnet": "anthropic", "haiku": "anthropic",
    "kimi": "kimi", "local": "local", "gemini": "gemini",
    "openrouter": "openrouter", "anthropic": "anthropic",
}


@dataclass
class LaneRecommendation:
    """A routing recommendation: lane, primary+fallback provider, cost estimate, budget snapshot."""

    lane: Lane
    primary_provider: str       # e.g. "opus"
    fallback_provider: str      # next step if pressure rises
    rationale: str              # one-line human-readable reason
    est_input_cost_usd: float   # rough cost for this task
    est_output_cost_usd: float
    budget_snapshot: str        # formatted "5h: 1% | weekly: 23% | pressure: 0.14"


# ---------------------------------------------------------------------------
# Token / cost helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(task_size_chars: int, loop_iterations: int) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) for the full run."""
    input_tokens = (task_size_chars // 4) * loop_iterations
    output_tokens = int(input_tokens * 0.3)
    return input_tokens, output_tokens


def _estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> tuple[float, float]:
    """Return (input_cost_usd, output_cost_usd) for the given provider."""
    lower = provider.lower()
    if lower == "local":
        return 0.0, 0.0

    # Adapter: iterate KNOWN_PRICING tuples and convert per-1K → per-1M
    rates = None
    for key, (inp, out) in _get_pricing().items():
        if lower in key or key in lower:
            rates = (inp, out)
            break

    if rates is None:
        # Fallback to first kimi model
        for key, (inp, out) in _get_pricing().items():
            if "kimi" in key:
                rates = (inp, out)
                break

    if rates is None:
        rates = (0.0, 0.0)

    in_cost = (input_tokens / 1_000_000) * (rates[0] * 1000)
    out_cost = (output_tokens / 1_000_000) * (rates[1] * 1000)
    return in_cost, out_cost


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_preflight(
    task_size_chars: int,
    loop_iterations: int,
    is_orchestration: bool = False,
) -> bool:
    """Return True when a pre-flight budget check should be shown.

    Triggers when the task is large (>20 000 chars), multi-iteration
    (>=3 loops), or is an orchestration session.
    """
    return (
        task_size_chars > 20_000
        or loop_iterations >= 3
        or is_orchestration
    )


def recommend_lane(
    task_size_chars: int,
    loop_iterations: int = 1,
    user_preference: Lane | None = None,
) -> LaneRecommendation:
    """Return a LaneRecommendation for the given task parameters.

    If *user_preference* is given it is honoured verbatim; the budget
    snapshot is still populated so the caller can display pressure context.
    """
    budget = check_budget()
    pressure   = budget.pressure
    five_pct   = budget.five_hour_pct
    weekly_pct = budget.weekly_pct
    snapshot   = (
        f"5h: {five_pct:.0f}% | weekly: {weekly_pct:.0f}% | pressure: {pressure:.2f}"
    )

    # ------------------------------------------------------------------ lane
    if user_preference is not None:
        lane: Lane = user_preference
        rationale = (
            f"User-specified lane '{lane}' (budget: {snapshot})"
        )
    elif pressure < 0.20 and weekly_pct < 50:
        lane = "premium"
        rationale = (
            f"Low pressure ({pressure:.2f}) + weekly headroom ({weekly_pct:.0f}%) "
            "→ full-capability Opus run"
        )
    elif pressure < 0.40:
        lane = "balanced"
        rationale = (
            f"Moderate pressure ({pressure:.2f}) → Sonnet keeps quality high "
            "while preserving Anthropic quota"
        )
    elif pressure < 0.70:
        lane = "economy"
        rationale = (
            f"Elevated pressure ({pressure:.2f}) → Haiku reduces Anthropic spend"
        )
    else:
        lane = "economy"
        rationale = (
            f"High pressure ({pressure:.2f}) → Kimi/local to avoid throttling"
        )

    primary, fallback = _LANE_PROVIDERS[lane]

    # Override primary provider at high pressure even within economy lane
    if pressure >= 0.70 and lane == "economy":
        primary  = "kimi"
        fallback = "local"

    # --- Multi-provider budget check ---
    balances = _get_provider_balances()
    if balances:
        # Build ordered candidate list: [primary, fallback, all others by priority_score]
        all_providers = ["anthropic", "gemini", "openrouter", "kimi", "local"]
        ordered = sorted(
            all_providers,
            key=lambda k: balances.get(k).priority_score if balances.get(k) else 0.5,
            reverse=True,
        )

        # If chosen primary is broke or unavailable, pick highest-scoring alternative.
        # Map model names (opus/sonnet/haiku) to provider keys for balance lookup.
        primary_key = _MODEL_TO_PROVIDER_KEY.get(primary, primary)
        primary_bal = balances.get(primary_key)
        if primary_bal is not None and not primary_bal.has_budget:
            for candidate in ordered:
                cand_bal = balances.get(candidate)
                if cand_bal and cand_bal.has_budget and candidate != primary_key:
                    rationale += f" [auto-switched from {primary} (no budget) to {candidate}]"
                    primary = candidate
                    break

        # snapshot stays single-line (it's used as a banner header row)
        # Provider detail is shown separately in format_preflight_message()

    input_tokens, output_tokens = _estimate_tokens(task_size_chars, loop_iterations)
    in_cost, out_cost = _estimate_cost(primary, input_tokens, output_tokens)

    return LaneRecommendation(
        lane=lane,
        primary_provider=primary,
        fallback_provider=fallback,
        rationale=rationale,
        est_input_cost_usd=in_cost,
        est_output_cost_usd=out_cost,
        budget_snapshot=snapshot,
    )


def format_preflight_message(
    rec: LaneRecommendation,
    alternatives: list[LaneRecommendation],
) -> str:
    """Return a concise (<20 lines, <1200 chars) preflight banner.

    Includes the budget snapshot, recommended lane + cost, and up to
    2 alternative lanes as quick-select hints.
    """
    total = rec.est_input_cost_usd + rec.est_output_cost_usd
    lines: list[str] = [
        "┌─ Budget Pre-flight ─────────────────────────────────────────┐",
        f"│ {rec.budget_snapshot:<61}│",
        "├──────────────────────────────────────────────────────────────┤",
        f"│ Recommended lane : {rec.lane:<10}  provider: {rec.primary_provider:<10}         │",
        f"│ Est. cost        : ${total:.4f} USD "
        f"(in=${rec.est_input_cost_usd:.4f} out=${rec.est_output_cost_usd:.4f})   │",
        f"│ Rationale        : {rec.rationale[:59]:<59}│",
    ]

    if alternatives:
        lines.append("├──────────────────────────────────────────────────────────────┤")
        lines.append("│ Alternatives:                                                │")
        for alt in alternatives[:2]:
            alt_total = alt.est_input_cost_usd + alt.est_output_cost_usd
            hint = f"  • [{alt.lane}] via {alt.primary_provider} — ${alt_total:.4f} USD"
            lines.append(f"│{hint:<62}│")

    # Multi-provider status block
    balances = _get_provider_balances()
    if balances:
        lines.append("├──────────────────────────────────────────────────────────────┤")
        lines.append("│ Provider status:                                              │")
        for k in ["anthropic", "gemini", "openrouter", "kimi", "local"]:
            b = balances.get(k)
            if b:
                icon = "✓" if b.has_budget else "✗"
                score = f"{b.priority_score:.2f}"
                summary = b.summary_line()[:35]
                entry = f"  {icon} {b.name[:20]:<20} {score}  {summary}"
                lines.append(f"│{entry:<62}│")

    lines.append("└──────────────────────────────────────────────────────────────┘")
    return "\n".join(lines)
