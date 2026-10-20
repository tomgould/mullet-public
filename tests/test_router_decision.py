"""Tests for Router.route — agent overrides, force flags, local-only, healthy gating."""



from mullet.core.router.complexity import Tier
from mullet.core.router.decision import (
    AGENT_TIER_OVERRIDES,
    ProviderEntry,
    Router,
    RoutingDecision,
)


def _make_router(**kwargs) -> Router:
    """Build a Router with minimal provider set for testing."""
    default_providers = {
        "local":           ProviderEntry(name="local", tier="free",
                                         models={"default": "llama-8b"}),
        "cheap_cloud":     ProviderEntry(name="cheap_cloud", tier="cheap",
                                         models={"default": "kimi-k2.5"}),
        "premium_cloud":   ProviderEntry(name="premium_cloud", tier="premium",
                                         models={"default": "claude-sonnet-4-6"}),
    }
    providers = kwargs.pop("providers", default_providers)
    return Router(providers=providers, **kwargs)


# --- agent tier overrides --------------------------------------------------


def test_agent_tier_override_explore_is_low():
    """agent_type='explore' must force Tier.LOW regardless of prompt."""
    router = _make_router()
    decision = router.route("complex multi-step refactor", agent_type="explore")
    assert decision.tier == Tier.LOW
    assert any("explore" in r for r in decision.reasons)


def test_agent_tier_override_code_reviewer_is_critical():
    """agent_type='code-reviewer' must force Tier.CRITICAL."""
    router = _make_router()
    decision = router.route("quick check", agent_type="code-reviewer")
    assert decision.tier == Tier.CRITICAL


def test_agent_tier_override_adaptive_agents_use_complexity():
    """Adaptive agents (debugger, verifier) skip override and analyze complexity."""
    router = _make_router()
    decision = router.route("hello world", agent_type="debugger")
    # Adaptive → falls back to complexity analysis; simple prompt → LOW or STANDARD
    assert decision.tier in (Tier.LOW, Tier.STANDARD)


def test_agent_tier_override_unknown_agent_analyzes_complexity():
    """Unknown agent_type falls back to complexity analysis."""
    router = _make_router()
    decision = router.route("simple task", agent_type="unknown-type-xyz")
    assert isinstance(decision, RoutingDecision)


# --- force_tier / force_provider -------------------------------------------


def test_force_tier_string_accepted():
    """force_tier='ADVANCED' works as a string."""
    router = _make_router()
    decision = router.route("irrelevant", force_tier="ADVANCED")
    assert decision.tier == Tier.ADVANCED


def test_force_tier_enum_accepted():
    """force_tier=Tier.CRITICAL works as an enum."""
    router = _make_router()
    decision = router.route("irrelevant", force_tier=Tier.CRITICAL)
    assert decision.tier == Tier.CRITICAL


def test_force_provider_skips_tier_selection():
    """force_provider returns that provider regardless of tier logic."""
    router = _make_router()
    decision = router.route("anything", force_provider="premium_cloud")
    assert decision.provider == "premium_cloud"
    assert any("forced provider" in r for r in decision.reasons)


def test_force_provider_unknown_ignored():
    """Unknown force_provider falls back to normal routing."""
    router = _make_router()
    # "nonexistent" is not in providers → falls back to complexity-based routing
    decision = router.route("simple task", force_provider="nonexistent")
    assert decision.provider != "nonexistent"


# --- local_only mode -------------------------------------------------------


def test_local_only_with_local_provider():
    """local_only=True always routes to local."""
    router = _make_router(local_only=True)
    decision = router.route("very complex multi-day refactor with 50 files")
    assert decision.provider == "local"
    assert decision.tier == Tier.LOW
    assert any("local-only" in r for r in decision.reasons)


def test_local_only_without_local_provider():
    """local_only=True with no local provider returns none/0 confidence."""
    router = _make_router(
        providers={
            "cheap_cloud": ProviderEntry(name="cheap_cloud", tier="cheap",
                                         models={"default": "kimi-k2.5"})
        },
        local_only=True,
    )
    decision = router.route("anything")
    assert decision.provider == "none"
    assert decision.confidence == 0.0


# --- AGENT_TIER_OVERRIDES sanity -------------------------------------------


def test_agent_tier_overrides_all_known_agents_have_tier():
    """Every agent in AGENT_TIER_OVERRIDES has either a Tier or None (adaptive)."""
    for agent, override in AGENT_TIER_OVERRIDES.items():
        assert override is None or isinstance(override, Tier), (
            f"{agent} override must be Tier or None; got {type(override)}"
        )


# --- health / budget hooks -------------------------------------------------


def test_mark_failure_records_failure_time():
    """mark_failure sets backoff_until on the provider."""
    router = _make_router()
    router.mark_failure("cheap_cloud", RuntimeError("network"))
    entry = router.providers["cheap_cloud"]
    assert entry.last_failure > 0
    assert entry.backoff_until > entry.last_failure


def test_mark_success_resets_failure_state():
    """mark_success clears backoff_until."""
    router = _make_router()
    router.mark_failure("cheap_cloud", RuntimeError("x"))
    router.mark_success("cheap_cloud")
    entry = router.providers["cheap_cloud"]
    assert entry.backoff_until == 0.0
    assert entry.healthy is True


# --- G8 / G9: allowlist-aware routing -------------------------------------

def _make_router_with_kimi_haiku():
    providers = {
        "kimi":  ProviderEntry(name="kimi",  tier="cheap",
                               models={"default": "kimi-k2.5"}),
        "haiku": ProviderEntry(name="haiku", tier="cheap",
                               models={"default": "claude-haiku-4-5"}),
        "local": ProviderEntry(name="local", tier="free",
                               models={"default": "llama-8b"}),
    }
    return Router(providers=providers)


def test_force_tier_critical_with_haiku_only_returns_none_reason():
    """Haiku cannot serve CRITICAL — allowlist=[haiku] must refuse, not silently pick."""
    router = _make_router_with_kimi_haiku()
    d = router.route("x", force_tier="CRITICAL", allowed_providers=frozenset({"haiku"}))
    assert d.provider is None
    assert any("cannot serve tier" in r for r in d.reasons)


def test_force_tier_critical_with_kimi_returns_kimi():
    """Kimi serves CRITICAL — allowlist=[kimi] must pick kimi."""
    router = _make_router_with_kimi_haiku()
    d = router.route("x", force_tier="CRITICAL", allowed_providers=frozenset({"kimi"}))
    assert d.provider == "kimi"
    assert d.tier == Tier.CRITICAL


def test_allowed_providers_stored_on_decision():
    """allowed_providers must round-trip onto the returned RoutingDecision."""
    router = _make_router_with_kimi_haiku()
    allow = frozenset({"kimi", "haiku"})
    d = router.route("x", force_tier="STANDARD", allowed_providers=allow)
    assert d.allowed_providers == allow


def test_force_provider_rejected_by_allowlist():
    """force_provider outside allowlist must be refused loudly, not silently honoured."""
    router = _make_router_with_kimi_haiku()
    d = router.route(
        "x",
        force_provider="haiku",
        allowed_providers=frozenset({"kimi"}),
    )
    assert d.provider is None
    assert any("rejected by allowlist" in r for r in d.reasons)


def test_allowlist_tier_fallback_picks_from_allowed_only():
    """With kimi+haiku registered and STANDARD tier, allowlist={haiku} picks haiku."""
    router = _make_router_with_kimi_haiku()
    d = router.route("x", force_tier="STANDARD", allowed_providers=frozenset({"haiku"}))
    assert d.provider == "haiku"


# --- G14: Router.route_delegation honours allowed_providers ----------------


def test_route_delegation_honours_allowlist():
    """route_delegation forwards allowed_providers and refuses providers outside it."""
    router = _make_router_with_kimi_haiku()
    d = router.route_delegation(
        parent_tier="CRITICAL",
        child_task="x",
        allowed_providers=frozenset({"kimi"}),
    )
    assert d.allowed_providers == frozenset({"kimi"})
    # Child must land on kimi (the only allowed provider with capacity)
    assert d.provider in ("kimi", None)


def test_route_delegation_no_allowlist_behaves_as_before():
    """route_delegation without allowed_providers preserves legacy behaviour."""
    router = _make_router_with_kimi_haiku()
    d = router.route_delegation(parent_tier="STANDARD", child_task="x")
    assert d.allowed_providers is None
    assert d.provider is not None


def test_route_delegation_stores_allowlist_on_decision():
    """allowed_providers round-trips onto the returned RoutingDecision."""
    router = _make_router_with_kimi_haiku()
    allow = frozenset({"kimi", "haiku"})
    d = router.route_delegation(parent_tier="STANDARD", child_task="x", allowed_providers=allow)
    assert d.allowed_providers == allow
