"""The intelligent compute router — routes tasks to optimal provider/model.

Combines OMC complexity analysis, multi-provider selection, budget
tracking, and Hermes-style fallback chains.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mullet.core.router.complexity import (
    Tier, extract_signals, is_reasoning_task, is_simple_message, score_complexity, score_to_tier,
)

import json
from pathlib import Path

if TYPE_CHECKING:
    from mullet.core.providers.health import HealthTracker
    from mullet.core.router.budget import BudgetTracker

logger = logging.getLogger(__name__)

# Quality-routing thresholds — importable so CLI display stays in sync
QUALITY_THRESHOLD: float = 0.60
QUALITY_MIN_SAMPLES: int = 3
QUALITY_USER_THRESHOLD: float = 0.35  # importable so CLI display stays in sync


def _provider_allowed(provider: str, cached_balance) -> bool:
    """Return True if provider has budget AND is within any configured spend cap."""
    if cached_balance is not None and not cached_balance.has_budget:
        return False
    try:
        from mullet.core.providers.spend_caps import check_spend_cap, load_budget_config
        config = load_budget_config()
        # Pass spend amount (used), not remaining, so cap comparison is correct
        balance_usd = cached_balance.credits_used_usd if cached_balance else None
        allowed, _reason = check_spend_cap(provider, balance_usd, config)
        return allowed
    except Exception:
        return True  # fail-open on any error


def _apply_probe_filter(provider: str, fallback_chain: list[str], reasons: list[str]) -> str:
    """Check cached probe results and skip providers with zero balance or exceeded caps.

    Pure synchronous cache read — no HTTP, no asyncio.  Fail-open when
    the cache has no entry or when anything goes wrong.
    """
    try:
        from mullet.core.providers import budget_probe

        cached = budget_probe._cache.get(provider)
        if not _provider_allowed(provider, cached):
            for fallback in fallback_chain:
                fb_cached = budget_probe._cache.get(fallback)
                if _provider_allowed(fallback, fb_cached):
                    reasons.append(f"probe: {provider} blocked → {fallback}")
                    return fallback
    except Exception:
        pass
    return provider


# Agent tier overrides — some agents always run at a fixed tier
AGENT_TIER_OVERRIDES: dict[str, Tier | None] = {
    "explore": Tier.LOW,
    "writer": Tier.LOW,
    "executor": Tier.STANDARD,
    "test-engineer": Tier.STANDARD,
    "designer": Tier.STANDARD,
    "researcher": Tier.STANDARD,
    "planner": Tier.ADVANCED,
    "architect": Tier.ADVANCED,
    "code-reviewer": Tier.CRITICAL,
    "analyst": Tier.CRITICAL,
    "debugger": None,  # adaptive
    "verifier": None,  # adaptive
}


@dataclass
class RoutingDecision:
    """Result of routing a task to a provider."""
    provider: str
    model: str
    tier: Tier
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    fallback_chain: list[str] = field(default_factory=list)
    allowed_providers: frozenset[str] | None = None


@dataclass
class ProviderEntry:
    """A registered provider with its tier and cost info."""
    name: str
    tier: str  # "free" | "cheap" | "premium"
    models: dict[str, str] = field(default_factory=dict)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    healthy: bool = True
    last_failure: float = 0.0
    backoff_until: float = 0.0


def _quality_history_path() -> Path:
    """Return path to routing_history.jsonl — injectable for tests."""
    from mullet.core.routing_history import history_path
    return history_path()


def _quality_tracker_path() -> Path:
    """Return path to provider_quality.json — injectable for tests."""
    from mullet.core.paths import config_dir
    return config_dir() / "provider_quality.json"


def _load_user_quality_scores(path: Path, min_samples: int = 3) -> dict[str, float]:
    """Return {provider: approval_rate} for providers with >= min_samples user ratings.

    Reads from provider_quality.json (written by QualityTracker).
    Fail-open: returns empty dict if file absent or unreadable.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, float] = {}
        for provider, stats in raw.items():
            up = stats.get("up", 0)
            down = stats.get("down", 0)
            total = up + down
            if total >= min_samples:
                result[provider] = up / total
        return result
    except Exception as exc:
        logger.debug("quality routing: failed to load user scores: %s", exc)
        return {}


def _load_provider_success_rates(path: Path, min_samples: int = 3) -> dict[str, float]:
    """Return {provider: success_rate} for providers with >= min_samples records.

    Fail-open: returns empty dict if file absent or unreadable.
    """
    if not path.exists():
        return {}
    try:
        data: dict[str, list[bool]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = d.get("provider", "")
            if p:
                data.setdefault(p, []).append(bool(d.get("success", False)))
        return {
            p: sum(outcomes) / len(outcomes)
            for p, outcomes in data.items()
            if len(outcomes) >= min_samples
        }
    except Exception as exc:
        logger.debug("quality routing: failed to load history: %s", exc)
        return {}


# Rate-limit down-weighting — look at the last RATE_LIMIT_WINDOW calls per provider
RATE_LIMIT_WINDOW: int = 20
RATE_LIMIT_THRESHOLD: float = 0.40  # skip if >40% of last 20 calls were rate-limit errors
RATE_LIMIT_WATCHLIST_THRESHOLD: float = 0.20  # warn (but don't skip) at ≥20% rate


def _load_recent_rate_limit_rates(
    path: Path, window: int = RATE_LIMIT_WINDOW, min_samples: int = 5
) -> dict[str, float]:
    """Return {provider: rate_limit_rate} based on the last `window` calls per provider.

    Reads routing_history.jsonl in forward order, keeps only the last `window`
    outcomes per provider, and returns the fraction that had error_class="rate_limit".
    Fail-open: returns empty dict if file absent or unreadable.
    """
    if not path.exists():
        return {}
    try:
        per_provider: dict[str, list[str | None]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = d.get("provider", "")
            if not p or p == "unknown":
                continue
            ec = d.get("error_class")
            per_provider.setdefault(p, []).append(ec)
        result: dict[str, float] = {}
        for p, ecs in per_provider.items():
            recent = ecs[-window:]
            if len(recent) < min_samples:
                continue
            rl_count = sum(1 for ec in recent if ec == "rate_limit")
            result[p] = rl_count / len(recent)
        return result
    except Exception as exc:
        logger.debug("rate-limit routing: failed to load history: %s", exc)
        return {}


def _load_provider_quality_data(
    path: Path,
    min_success_samples: int = 3,
    rl_window: int = RATE_LIMIT_WINDOW,
    min_rl_samples: int = 5,
) -> tuple[dict[str, float], dict[str, float]]:
    """Read routing_history.jsonl once and return (success_rates, rl_rates).

    success_rates: {provider: all-time success rate} for providers with >= min_success_samples records.
    rl_rates: {provider: rate_limit_rate} based on last rl_window calls with >= min_rl_samples.
    Fail-open: returns ({}, {}) on any error.
    """
    if not path.exists():
        return {}, {}
    try:
        success_data: dict[str, list[bool]] = {}
        rl_data: dict[str, list[str | None]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = d.get("provider", "")
            if p:
                success_data.setdefault(p, []).append(bool(d.get("success", False)))
            if p and p != "unknown":
                rl_data.setdefault(p, []).append(d.get("error_class"))
        success_rates = {
            p: sum(outcomes) / len(outcomes)
            for p, outcomes in success_data.items()
            if len(outcomes) >= min_success_samples
        }
        rl_rates: dict[str, float] = {}
        for p, ecs in rl_data.items():
            recent = ecs[-rl_window:]
            if len(recent) < min_rl_samples:
                continue
            rl_count = sum(1 for ec in recent if ec == "rate_limit")
            rl_rates[p] = rl_count / len(recent)
        return success_rates, rl_rates
    except Exception as exc:
        logger.debug("quality routing: failed to load combined quality data: %s", exc)
        return {}, {}


class Router:
    """The intelligent compute router.

    Combines complexity analysis, provider management,
    and budget tracking into a single routing decision.
    """

    def __init__(
        self,
        providers: dict[str, ProviderEntry] | None = None,
        tier_map: dict[str, list[str]] | None = None,
        thresholds: dict | None = None,
        local_only: bool = False,
        health_tracker: HealthTracker | None = None,
        budget_tracker: BudgetTracker | None = None,
        use_pressure_routing: bool = False,
    ):
        self.providers = providers or {}
        self.tier_map = tier_map or {
            "LOW": ["local"],
            "STANDARD": ["cheap_cloud", "local"],
            "ADVANCED": ["premium_cloud", "cheap_cloud"],
            "CRITICAL": ["premium_cloud"],
        }
        self.thresholds = thresholds or {"local_max": 30, "standard_max": 76, "advanced_max": 90}
        self.local_only = local_only
        self.health = health_tracker
        self.budget = budget_tracker
        self.use_pressure_routing = use_pressure_routing

    # G8/G9 — provider priority when falling back inside an allowlist (cheapest first)
    # Phase E: gemini demoted to last-resort (not cheap); openrouter added after kimi
    _ALLOWLIST_FALLBACK_ORDER: list[str] = [
        "local", "cheap_cloud", "kimi", "openrouter",
        "haiku", "sonnet", "premium_cloud", "opus", "opus-4-7", "opus-4-8", "gemini",
    ]
    # G9 — preferred order when picking a tier-capable provider inside allowlist
    # Phase E: gemini demoted to last; openrouter added after kimi
    _ALLOWLIST_TIER_PICK_ORDER: list[str] = [
        "kimi", "openrouter", "sonnet", "opus-4-8", "opus", "opus-4-7", "haiku",
        "cheap_cloud", "premium_cloud", "local", "gemini",
    ]

    def _pick_allowlisted_provider(
        self,
        allowed: frozenset[str],
        tier_value: str | None = None,
    ) -> str | None:
        """Pick highest-priority provider from allowlist; optionally tier-capable."""
        from mullet.core.router.complexity import PROVIDER_TIER_CAPABILITIES

        def _eligible(name: str) -> bool:
            if name not in allowed or name not in self.providers:
                return False
            if tier_value is not None:
                caps = PROVIDER_TIER_CAPABILITIES.get(name, set())
                if tier_value not in caps:
                    return False
            return True

        order = self._ALLOWLIST_TIER_PICK_ORDER if tier_value else self._ALLOWLIST_FALLBACK_ORDER
        for name in order:
            if not _eligible(name):
                continue
            if name == "kimi" and _eligible("openrouter"):
                return self._pick_cheap_cloud_peer("kimi", "openrouter")
            return name
        return None

    def _pick_cheap_cloud_peer(self, a: str, b: str) -> str:
        """Break a tie between two equally-cheap providers using live history.

        Fail-open: falls back to *a* (today's static-order default) when there's
        no data yet or the signals are tied -- never raises, never blocks routing.
        """
        quality, rl = _load_provider_quality_data(
            _quality_history_path(), min_success_samples=QUALITY_MIN_SAMPLES,
        )
        a_rl, b_rl = rl.get(a, 0.0), rl.get(b, 0.0)
        # Rate-limit safety first -- a provider actively getting 429'd is a worse
        # pick regardless of historical success rate.
        if a_rl > RATE_LIMIT_THRESHOLD and b_rl <= RATE_LIMIT_THRESHOLD:
            return b
        if b_rl > RATE_LIMIT_THRESHOLD and a_rl <= RATE_LIMIT_THRESHOLD:
            return a
        a_q, b_q = quality.get(a), quality.get(b)
        if a_q is not None and b_q is not None and a_q != b_q:
            return a if a_q > b_q else b
        return a

    def pick_provider_for_tier(
        self,
        tier_value: str,
        allowed_providers: frozenset[str] | None,
        pressure: float,
    ) -> str | None:
        """Choose the best provider for a given tier under the current budget pressure.

        Phase B: thin wrapper around recommend_provider — preserves the headroom
        pick (Haiku at STANDARD<0.6, Sonnet at ADVANCED<0.3) while also honouring
        allowed_providers and PROVIDER_TIER_CAPABILITIES.

        Returns None when nothing in the allowlist can serve the tier; the caller
        must fall through to the existing tier_map / quality / fallback logic.

        Args:
            tier_value: "LOW" | "STANDARD" | "ADVANCED" | "CRITICAL"
            allowed_providers: Preset allowlist (None = no restriction).
            pressure: 0.0–1.0 budget pressure from compute_pressure().
        """
        from mullet.hud.guard import recommend_provider
        from mullet.core.router.complexity import PROVIDER_TIER_CAPABILITIES

        def _ok(p: str) -> bool:
            return (
                p in self.providers
                and (allowed_providers is None or p in allowed_providers)
                and tier_value in PROVIDER_TIER_CAPABILITIES.get(p, set())
            )

        rec = recommend_provider(pressure, tier_value)
        # S4: preserve headroom pick — use recommend_provider's result when it fits
        if rec != "pause" and _ok(rec):
            return rec

        # rec doesn't fit the allowlist/providers — fall back to best tier-capable
        # allowed provider via the existing static priority helper
        if allowed_providers is not None:
            return self._pick_allowlisted_provider(allowed_providers, tier_value=tier_value)

        # No allowlist: scan _ALLOWLIST_TIER_PICK_ORDER for any registered + tier-capable provider
        return next((p for p in self._ALLOWLIST_TIER_PICK_ORDER if _ok(p)), None)

    def route(
        self,
        task_prompt: str,
        agent_type: str | None = None,
        force_tier: Tier | str | None = None,
        force_provider: str | None = None,
        allowed_providers: frozenset[str] | None = None,
    ) -> RoutingDecision:
        """Route a task to the optimal provider and model, honouring allowed_providers if set."""
        reasons = []

        # Local-only mode
        if self.local_only:
            local = self._find_provider("local")
            if local:
                return RoutingDecision(
                    provider=local.name,
                    model=local.models.get("default", "default"),
                    tier=Tier.LOW,
                    reasons=["local-only mode"],
                    allowed_providers=allowed_providers,
                )
            reasons.append("local-only but no local provider found")
            return RoutingDecision(
                provider="none",
                model="none",
                tier=Tier.LOW,
                confidence=0.0,
                reasons=reasons,
                allowed_providers=allowed_providers,
            )

        # Force overrides
        if force_provider and force_provider in self.providers:
            if allowed_providers is not None and force_provider not in allowed_providers:
                logger.error(
                    "force_provider=%s rejected — not in allowlist %s",
                    force_provider, sorted(allowed_providers),
                )
                return RoutingDecision(
                    provider=None,
                    model="",
                    tier=force_tier if isinstance(force_tier, Tier) else Tier.STANDARD,
                    confidence=0.0,
                    reasons=[f"force_provider={force_provider} rejected by allowlist={sorted(allowed_providers)}"],
                    allowed_providers=allowed_providers,
                )
            p = self.providers[force_provider]
            return RoutingDecision(
                provider=p.name,
                model=p.models.get("default", "default"),
                tier=force_tier if isinstance(force_tier, Tier) else Tier.STANDARD,
                reasons=[f"forced provider: {force_provider}"],
                allowed_providers=allowed_providers,
            )

        # Determine tier
        if force_tier:
            tier = Tier(force_tier) if isinstance(force_tier, str) else force_tier
            reasons.append(f"forced tier: {tier.value}")
        elif agent_type and agent_type in AGENT_TIER_OVERRIDES:
            override = AGENT_TIER_OVERRIDES[agent_type]
            if override is not None:
                tier = override
                reasons.append(f"agent override: {agent_type} → {tier.value}")
            else:
                tier = self._analyze_complexity(task_prompt, reasons)
        else:
            tier = self._analyze_complexity(task_prompt, reasons)

            # Check routing learning — similar past tasks may suggest a better tier
            try:
                from mullet.core.router.learning import suggest_tier
                learned = suggest_tier(task_prompt)
                if learned:
                    learned_tier = Tier(learned)
                    tier_order = {Tier.LOW: 0, Tier.STANDARD: 1, Tier.ADVANCED: 2, Tier.CRITICAL: 3}
                    # Only downgrade (save money), never upgrade blindly
                    if tier_order.get(learned_tier, 1) < tier_order.get(tier, 1):
                        tier = learned_tier
                        reasons.append(f"learned: similar tasks succeeded at {learned}")
            except Exception as _exc:
                logger.warning("decision: learned tier suggestion failed: %s", _exc)

        # Phase B: reasoning-task tier floor — if the prompt calls for deep analysis and
        # the score placed it at LOW, raise the effective tier to STANDARD so genuine
        # reasoning work routes to at least cheap-cloud / haiku (not just local).
        # Guarded by is_reasoning_task(), which already returns False for simple messages,
        # so trivial keyword-bearing prompts ("list the status") are not escalated.
        if tier == Tier.LOW and not force_tier and is_reasoning_task(task_prompt):
            tier = Tier.STANDARD
            reasons.append("reasoning task → floor STANDARD")

        # Budget-pressure override: use cheaper/better model based on live usage
        # Only active when use_pressure_routing=True (set by build_default_router, not tests)
        if self.use_pressure_routing:
            try:
                from mullet.hud.guard import compute_pressure
                from mullet.hud.usage import fetch_usage
                limits = fetch_usage()
                if limits:
                    pressure = compute_pressure(limits)
                    # Phase B: delegate provider selection to pick_provider_for_tier, which is
                    # a thin wrapper around recommend_provider that also honours the allowlist
                    # and PROVIDER_TIER_CAPABILITIES.  Returns None when nothing fits.
                    chosen = self.pick_provider_for_tier(tier.value, allowed_providers, pressure)
                    if chosen is not None:
                        # Probe-cache filter: skip provider if cached balance is zero
                        fallback_candidates = [
                            p for p in ["haiku", "kimi", "local"]
                            if p in self.providers and (allowed_providers is None or p in allowed_providers)
                        ]
                        chosen = _apply_probe_filter(
                            chosen,
                            [p for p in fallback_candidates if p != chosen],
                            reasons,
                        )
                        entry = self.providers[chosen]
                        model = entry.models.get("default", "default")
                        reasons.append(f"budget-pressure {pressure:.2f} -> {chosen}")
                        return RoutingDecision(
                            provider=entry.name,
                            model=model,
                            tier=tier,
                            confidence=0.9,
                            reasons=reasons,
                            fallback_chain=[p for p in fallback_candidates if p != chosen],
                            allowed_providers=allowed_providers,
                        )
                    # pick returned None — no provider in allowlist can serve this tier
                    # Fall through to the existing tier_map / allowlist / quality logic below
                    logger.debug(
                        "decision: budget-pressure pick returned None for tier=%s allowlist=%s — falling through",
                        tier.value, sorted(allowed_providers) if allowed_providers else None,
                    )
            except Exception as _exc:
                logger.warning("decision: budget pressure routing failed: %s", _exc)

        # G9 — when force_tier or any tier-based selection, honour allowlist first
        provider_names = self.tier_map.get(tier.value, [])
        if allowed_providers is not None:
            fallback = self._pick_allowlisted_provider(allowed_providers, tier_value=tier.value)
            if fallback is None:
                logger.error(
                    "No provider in allowlist %s can serve tier %s",
                    sorted(allowed_providers), tier.value,
                )
                return RoutingDecision(
                    provider=None,
                    model="",
                    tier=tier,
                    confidence=0.0,
                    reasons=reasons + [
                        f"force_tier={tier.value} but allowlist={sorted(allowed_providers)} cannot serve tier"
                    ],
                    allowed_providers=allowed_providers,
                )
            entry = self._find_provider(fallback)
            if entry and self._is_healthy(entry) and self._check_budget(entry):
                model = entry.models.get(tier.value.lower(), entry.models.get("default", "default"))
                reasons.append(f"allowlist tier pick: {fallback}")
                return RoutingDecision(
                    provider=entry.name,
                    model=model,
                    tier=tier,
                    reasons=reasons,
                    estimated_cost=self._estimate_cost(entry, 4000),
                    fallback_chain=[n for n in provider_names if n != fallback and n in allowed_providers],
                    allowed_providers=allowed_providers,
                )

        # Load quality success rates once — fail-open, returns {} when file absent
        _quality_rates, _rl_rates = _load_provider_quality_data(
            _quality_history_path(),
            min_success_samples=QUALITY_MIN_SAMPLES,
        )
        _user_quality = _load_user_quality_scores(_quality_tracker_path(), min_samples=QUALITY_MIN_SAMPLES)
        _quality_skipped: list[str] = []

        for pname in provider_names:
            entry = self._find_provider(pname)
            if not entry or not self._is_healthy(entry) or not self._check_budget(entry):
                if entry and not self._check_budget(entry):
                    reasons.append(f"budget: {pname} too expensive")
                continue
            rate = _quality_rates.get(pname)
            if rate is not None and rate < QUALITY_THRESHOLD:
                reasons.append(
                    f"quality skip: {pname} success_rate={rate:.0%} < {QUALITY_THRESHOLD:.0%}"
                )
                _quality_skipped.append(pname)
                continue
            user_score = _user_quality.get(pname)
            if user_score is not None and user_score < QUALITY_USER_THRESHOLD:
                reasons.append(
                    f"user-rating skip: {pname} approval={user_score:.0%} < {QUALITY_USER_THRESHOLD:.0%}"
                )
                _quality_skipped.append(pname)
                continue
            rl_rate = _rl_rates.get(pname)
            if rl_rate is not None and rl_rate > RATE_LIMIT_THRESHOLD:
                reasons.append(
                    f"rate-limit skip: {pname} rl_rate={rl_rate:.0%} > {RATE_LIMIT_THRESHOLD:.0%}"
                )
                _quality_skipped.append(pname)
                continue
            model = entry.models.get(tier.value.lower(), entry.models.get("default", "default"))
            fallbacks = [n for n in provider_names if n != pname]
            return RoutingDecision(
                provider=entry.name,
                model=model,
                tier=tier,
                reasons=reasons,
                estimated_cost=self._estimate_cost(entry, 4000),
                fallback_chain=fallbacks,
                allowed_providers=allowed_providers,
            )

        # Second pass: all quality-filtered — fall back to least-bad skipped provider
        for pname in _quality_skipped:
            entry = self._find_provider(pname)
            if entry and self._is_healthy(entry) and self._check_budget(entry):
                model = entry.models.get(tier.value.lower(), entry.models.get("default", "default"))
                fallbacks = [n for n in provider_names if n != pname]
                return RoutingDecision(
                    provider=entry.name,
                    model=model,
                    tier=tier,
                    reasons=reasons + ["quality fallback: all candidates below threshold"],
                    estimated_cost=self._estimate_cost(entry, 4000),
                    fallback_chain=fallbacks,
                    allowed_providers=allowed_providers,
                )

        # Budget auto-downgrade: if budget blocked all providers, try cheaper tiers
        tier_order = {Tier.LOW: 0, Tier.STANDARD: 1, Tier.ADVANCED: 2, Tier.CRITICAL: 3}
        downgrade_map = {
            Tier.CRITICAL: [Tier.ADVANCED, Tier.STANDARD, Tier.LOW],
            Tier.ADVANCED: [Tier.STANDARD, Tier.LOW],
            Tier.STANDARD: [Tier.LOW],
        }
        downgrade_chain = downgrade_map.get(tier, [])
        for down_tier in downgrade_chain:
            down_providers = self.tier_map.get(down_tier.value, [])
            for pname in down_providers:
                entry = self._find_provider(pname)
                if entry and self._is_healthy(entry) and self._check_budget(entry):
                    if allowed_providers is not None and entry.name not in allowed_providers:
                        continue
                    model = entry.models.get(down_tier.value.lower(), entry.models.get("default", "default"))
                    reasons.append(f"budget downgrade: {tier.value} -> {down_tier.value}")
                    return RoutingDecision(
                        provider=entry.name, model=model, tier=down_tier,
                        reasons=reasons,
                        estimated_cost=self._estimate_cost(entry, 4000),
                        allowed_providers=allowed_providers,
                    )

        # No healthy provider — try any available (still honour allowlist if set)
        for entry in self.providers.values():
            if allowed_providers is not None and entry.name not in allowed_providers:
                continue
            if self._is_healthy(entry):
                return RoutingDecision(
                    provider=entry.name,
                    model=entry.models.get("default", "default"),
                    tier=tier,
                    reasons=reasons + ["fallback: no healthy provider for requested tier"],
                    allowed_providers=allowed_providers,
                )

        return RoutingDecision(
            provider="none",
            model="none",
            tier=tier,
            confidence=0.0,
            reasons=reasons + ["ERROR: no providers available"],
            allowed_providers=allowed_providers,
        )

    def route_delegation(
        self,
        parent_tier: str,
        child_task: str,
        allowed_providers: frozenset[str] | None = None,
        allowed_tiers: list[str] | None = None,
    ) -> RoutingDecision:
        """Route a delegated subtask — default one tier below parent, honouring allowed_providers.

        When ``allowed_providers`` is set but ``allowed_tiers`` is not, the
        list of tiers those providers can serve (per
        ``complexity.PROVIDER_TIER_CAPABILITIES``) is auto-inferred and
        passed into ``score_to_tier`` so the child's analysed complexity
        cannot exceed what the allowlist can satisfy. This prevents the
        delegation router from picking ADVANCED/CRITICAL for child tasks
        whose only allowed providers are LOW/STANDARD-capable, then
        bouncing on the downstream allowlist check.
        """
        tier_down = {"CRITICAL": Tier.ADVANCED, "ADVANCED": Tier.STANDARD, "STANDARD": Tier.LOW, "LOW": Tier.LOW}
        default_tier = tier_down.get(parent_tier, Tier.LOW)

        # Auto-infer tier cap from allowed_providers when caller didn't
        # specify allowed_tiers explicitly. Explicit always wins.
        if allowed_providers and not allowed_tiers:
            from mullet.core.router.complexity import tiers_for_providers
            inferred = tiers_for_providers(allowed_providers)
            if inferred:
                allowed_tiers = inferred
                logger.info(
                    "route_delegation: inferred allowed_tiers=%s from allowed_providers=%s",
                    sorted(allowed_tiers), sorted(allowed_providers),
                )

        # Check if child task warrants same tier as parent — capped by
        # allowed_tiers when present (explicit or inferred).
        signals = extract_signals(child_task)
        score = score_complexity(signals)
        child_tier = score_to_tier(score, self.thresholds, allowed_tiers=allowed_tiers)

        # Use the lower of: parent's downgraded tier or child's analyzed tier
        tier_order = {Tier.LOW: 0, Tier.STANDARD: 1, Tier.ADVANCED: 2, Tier.CRITICAL: 3}
        final_tier = default_tier if tier_order.get(child_tier, 0) <= tier_order.get(default_tier, 0) else child_tier
        return self.route(child_task, force_tier=final_tier, allowed_providers=allowed_providers)

    def mark_failure(self, provider_name: str, error: Exception):
        """Record a provider failure with backoff."""
        entry = self.providers.get(provider_name)
        if entry:
            entry.healthy = False
            entry.last_failure = time.time()
            entry.backoff_until = time.time() + 60  # 60s backoff

    def mark_success(self, provider_name: str):
        """Record a provider success — reset failure state."""
        entry = self.providers.get(provider_name)
        if entry:
            entry.healthy = True
            entry.backoff_until = 0

    def record_usage(self, provider_name: str, tokens: int, cost: float, model: str = "unknown"):
        """Record token usage for budget tracking."""
        if self.budget:
            self.budget.record_usage(provider_name, model, tokens, 0)

    def _analyze_complexity(self, prompt: str, reasons: list[str]) -> Tier:
        """Run complexity analysis on the prompt."""
        if is_simple_message(prompt):
            reasons.append("simple message (fast pre-check)")
            return Tier.LOW

        signals = extract_signals(prompt)
        score = score_complexity(signals)
        tier = score_to_tier(score, self.thresholds)
        reasons.append(f"complexity score: {score:.1f} → {tier.value}")
        return tier

    def _find_provider(self, name: str) -> ProviderEntry | None:
        return self.providers.get(name)

    def _is_healthy(self, entry: ProviderEntry) -> bool:
        if self.health:
            return self.health.is_available(entry.name)
        # Fallback to simple flag
        if entry.backoff_until and time.time() < entry.backoff_until:
            return False
        return entry.healthy

    def _check_budget(self, entry: ProviderEntry, est_tokens: int = 4000) -> bool:
        """Check if budget allows this provider."""
        if not self.budget:
            return True
        model = entry.models.get("default", "default")
        return self.budget.can_afford(entry.name, model, est_tokens)

    def _estimate_cost(self, entry: ProviderEntry, est_tokens: int) -> float:
        if self.budget:
            model = entry.models.get("default", "default")
            return self.budget.estimate_cost(entry.name, model, est_tokens)
        return (est_tokens / 1000) * (entry.cost_per_1k_input + entry.cost_per_1k_output) / 2
