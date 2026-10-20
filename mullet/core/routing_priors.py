"""Routing priors — re-rank provider picks using historical success rates.

Consult :mod:`mullet.core.routing_history` to decide whether the router's
first-pick provider has a poor track record for this task signature, and if
so swap in a better-performing candidate.

Usage::

    from mullet.core.routing_priors import pick_better_provider
    new_provider, reason = pick_better_provider(task, tier, picked, candidates)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from mullet.core.routing_history import make_signature, success_rate, iter_recent

logger = logging.getLogger(__name__)


def pick_better_provider(
    task: str,
    tier: str,
    picked_provider: str,
    candidates: list[str],
    *,
    min_pick_rate: float = 0.50,
    good_enough_rate: float = 0.75,
    min_samples: int = 3,
    base: Optional[Path] = None,
) -> tuple[str, str | None]:
    """Return *(new_provider, reason_or_None)*.

    If priors don't suggest a change, returns *(picked_provider, None)*.

    Algorithm:
      1. Compute ``sig = make_signature(task, "run", tier)``
      2. ``pick_rate = success_rate(sig, picked_provider, min_samples=min_samples)``
      3. If ``pick_rate`` is None (not enough data) OR ``pick_rate >= min_pick_rate`` → keep picked
      4. Otherwise walk ``candidates`` in order; return first candidate whose
         ``success_rate >= good_enough_rate``
      5. If no candidate qualifies → return picked_provider (no change)
    """
    try:
        sig = make_signature(task, "run", tier)
        pick_rate = success_rate(sig, picked_provider, base=base, min_samples=min_samples)

        # Not enough data — don't second-guess the router
        if pick_rate is None:
            return picked_provider, None

        # Pick is performing well enough — keep it
        if pick_rate >= min_pick_rate:
            return picked_provider, None

        # Picked provider is underperforming; look for a better alternative
        for candidate in candidates:
            if candidate == picked_provider:
                continue
            cand_rate = success_rate(sig, candidate, base=base, min_samples=min_samples)
            if cand_rate is not None and cand_rate >= good_enough_rate:
                reason = (
                    f"pick_rate({picked_provider})={pick_rate:.2f} < {min_pick_rate}; "
                    f"{candidate} rate={cand_rate:.2f} >= {good_enough_rate}"
                )
                return candidate, reason

        # No suitable alternative found — keep original pick
        return picked_provider, None

    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.warning("routing_priors: pick_better_provider error (non-fatal): %s", exc)
        return picked_provider, None


def summarize_priors(
    task: str,
    tier: str,
    base: Optional[Path] = None,
    min_samples: int = 1,
) -> dict[str, tuple[float | None, int]]:
    """Return ``{provider: (success_rate_or_None, sample_count)}`` for *task/tier*.

    Sorted by sample_count descending.  Useful for logging and debug.
    """
    try:
        sig = make_signature(task, "run", tier)
        # Aggregate counts from history
        counts: dict[str, list[bool]] = {}
        for outcome in iter_recent(limit=2000, base=base):
            if outcome.task_signature == sig:
                counts.setdefault(outcome.provider, []).append(outcome.success)

        result: dict[str, tuple[float | None, int]] = {}
        for provider, outcomes in counts.items():
            n = len(outcomes)
            rate: float | None = (sum(outcomes) / n) if n >= min_samples else None
            result[provider] = (rate, n)

        return dict(sorted(result.items(), key=lambda kv: kv[1][1], reverse=True))

    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.warning("routing_priors: summarize_priors error (non-fatal): %s", exc)
        return {}


def pick_better_provider_semantic(
    task: str,
    tier: str,
    picked_provider: str,
    candidates: list[str],
    *,
    semantic_threshold: float = 0.75,
    min_pick_rate: float = 0.50,
    good_enough_rate: float = 0.75,
    min_samples: int = 3,
    base: Optional[Path] = None,
    **kwargs,
) -> tuple[str, str | None]:
    """Semantic wrapper — re-rank using weighted success from near-signature history.

    Algorithm:
      1. Call ``nearest_signatures(task, threshold=semantic_threshold)`` to find
         semantically similar past signatures.
      2. If no near-sigs found, delegate to exact-signature
         :func:`pick_better_provider`.
      3. Otherwise iterate recent history once; build
         ``{(sig, provider): [success_bool, ...]}`` for near-sigs only.
      4. Compute weighted success rate per provider::

             weighted_rate[p] = sum(w_s * sr_s * n_s) / sum(w_s * n_s)

         where ``w_s`` = cosine similarity, ``sr_s`` = raw success rate,
         ``n_s`` = sample count for *(sig, p)*.  Only pairs with
         ``n_s >= min_samples`` are included.
      5. If picked has no weighted data → fall back to exact-sig path.
      6. If ``weighted_rate[picked] < min_pick_rate`` AND some candidate has
         ``weighted_rate >= good_enough_rate`` → swap.
      7. Fail-open: returns *(picked_provider, None)* on any exception.
    """
    try:
        from mullet.core.semantic_sig import nearest_signatures  # lazy import

        near_sigs = nearest_signatures(task, threshold=semantic_threshold, base=base)
        if not near_sigs:
            # No semantic evidence — fall back to exact-signature logic
            return pick_better_provider(
                task,
                tier,
                picked_provider,
                candidates,
                min_pick_rate=min_pick_rate,
                good_enough_rate=good_enough_rate,
                min_samples=min_samples,
                base=base,
            )

        # Build per-(sig, provider) outcome lists from recent history
        near_sig_set = {s for s, _ in near_sigs}
        history: dict[tuple[str, str], list[bool]] = {}
        for outcome in iter_recent(limit=2000, base=base):
            if outcome.task_signature in near_sig_set:
                key = (outcome.task_signature, outcome.provider)
                history.setdefault(key, []).append(outcome.success)

        # Aggregate weighted success rate per provider
        all_providers = [picked_provider] + [c for c in candidates if c != picked_provider]
        numerator: dict[str, float] = {}
        denominator: dict[str, float] = {}

        for p in all_providers:
            for sig, w_s in near_sigs:
                records = history.get((sig, p), [])
                n_s = len(records)
                if n_s < min_samples:
                    continue
                sr = sum(records) / n_s
                numerator[p] = numerator.get(p, 0.0) + w_s * sr * n_s
                denominator[p] = denominator.get(p, 0.0) + w_s * n_s

        weighted_rate: dict[str, float] = {
            p: numerator[p] / denominator[p]
            for p in numerator
            if denominator.get(p, 0.0) > 0.0
        }

        # No data for picked → fall back to exact-sig
        if picked_provider not in weighted_rate:
            return pick_better_provider(
                task,
                tier,
                picked_provider,
                candidates,
                min_pick_rate=min_pick_rate,
                good_enough_rate=good_enough_rate,
                min_samples=min_samples,
                base=base,
            )

        pick_rate = weighted_rate[picked_provider]
        if pick_rate >= min_pick_rate:
            return picked_provider, None

        # Look for a sufficiently better candidate
        for candidate in candidates:
            if candidate == picked_provider:
                continue
            cand_rate = weighted_rate.get(candidate)
            if cand_rate is not None and cand_rate >= good_enough_rate:
                reason = (
                    f"semantic weighted_rate({picked_provider})={pick_rate:.2f} < {min_pick_rate}; "
                    f"{candidate} weighted_rate={cand_rate:.2f} >= {good_enough_rate} "
                    f"(across {len(near_sigs)} near-sig(s))"
                )
                return candidate, reason

        return picked_provider, None

    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.warning("routing_priors: pick_better_provider_semantic error (non-fatal): %s", exc)
        return picked_provider, None
