"""Gap-filling tests for mullet/core/router/decision.py — push coverage toward ≥85%."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.router.complexity import Tier
from mullet.core.router.decision import (
    _apply_probe_filter,
    _load_provider_quality_data,
    _load_provider_success_rates,
    _load_recent_rate_limit_rates,
    _load_user_quality_scores,
    _provider_allowed,
    ProviderEntry,
    Router,
    RoutingDecision,
)


# ---------------------------------------------------------------------------
# _provider_allowed
# ---------------------------------------------------------------------------


def test_provider_allowed_no_balance():
    """No cached balance → allowed."""
    assert _provider_allowed("kimi", None) is True


def test_provider_allowed_has_budget_false():
    """cached_balance.has_budget=False → blocked."""
    bal = MagicMock()
    bal.has_budget = False
    bal.credits_used_usd = 1.0
    assert _provider_allowed("kimi", bal) is False


def test_provider_allowed_spend_cap_blocks():
    """Spend cap exceeded → blocked."""
    bal = MagicMock()
    bal.has_budget = True
    bal.credits_used_usd = 10.0
    with patch(
        "mullet.core.providers.spend_caps.load_budget_config",
        return_value={"monthly_limit_usd": 5.0},
    ):
        with patch(
            "mullet.core.providers.spend_caps.check_spend_cap",
            return_value=(False, "cap exceeded"),
        ):
            assert _provider_allowed("kimi", bal) is False


def test_provider_allowed_exception_fails_open():
    """Exception inside spend-cap check → allowed (fail-open)."""
    bal = MagicMock()
    bal.has_budget = True
    bal.credits_used_usd = 1.0
    with patch(
        "mullet.core.providers.spend_caps.load_budget_config",
        side_effect=RuntimeError("boom"),
    ):
        assert _provider_allowed("kimi", bal) is True


# ---------------------------------------------------------------------------
# _apply_probe_filter
# ---------------------------------------------------------------------------


def test_probe_filter_blocks_and_fallback():
    """Primary provider blocked, fallback allowed → returns fallback."""
    blocked = MagicMock()
    blocked.has_budget = False
    allowed = MagicMock()
    allowed.has_budget = True

    import mullet.core.providers.budget_probe as _bp

    orig = getattr(_bp, "_cache", {})
    try:
        _bp._cache = {"kimi": blocked, "haiku": allowed}
        reasons: list[str] = []
        result = _apply_probe_filter("kimi", ["haiku"], reasons)
        assert result == "haiku"
        assert any("probe: kimi blocked → haiku" in r for r in reasons)
    finally:
        _bp._cache = orig


def test_probe_filter_allows_provider():
    """Primary provider allowed → returned unchanged."""
    allowed = MagicMock()
    allowed.has_budget = True

    import mullet.core.providers.budget_probe as _bp

    orig = getattr(_bp, "_cache", {})
    try:
        _bp._cache = {"kimi": allowed}
        reasons: list[str] = []
        result = _apply_probe_filter("kimi", ["haiku"], reasons)
        assert result == "kimi"
    finally:
        _bp._cache = orig


def test_probe_filter_exception_fails_open():
    """Exception in probe filter → returns original provider."""
    import mullet.core.providers.budget_probe as _bp

    orig = getattr(_bp, "_cache", {})
    try:
        _bp._cache = MagicMock()
        _bp._cache.get.side_effect = RuntimeError("boom")
        reasons: list[str] = []
        result = _apply_probe_filter("kimi", ["haiku"], reasons)
        assert result == "kimi"
    finally:
        _bp._cache = orig


# ---------------------------------------------------------------------------
# _load_user_quality_scores
# ---------------------------------------------------------------------------


def test_load_user_quality_missing_file(tmp_path):
    """Missing file → empty dict."""
    path = tmp_path / "no_such_file.json"
    assert _load_user_quality_scores(path) == {}


def test_load_user_quality_valid_data(tmp_path):
    """Valid data with enough samples → approval rates."""
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps({
            "kimi": {"up": 8, "down": 2},
            "haiku": {"up": 1, "down": 1},
        })
    )
    result = _load_user_quality_scores(path, min_samples=3)
    assert result == {"kimi": 0.8}


def test_load_user_quality_exception(tmp_path, caplog):
    """Corrupt file → empty dict and debug log."""
    path = tmp_path / "quality.json"
    path.write_text("not json")
    with caplog.at_level("DEBUG", logger="mullet"):
        assert _load_user_quality_scores(path) == {}
    assert "failed to load user scores" in caplog.text


# ---------------------------------------------------------------------------
# _load_provider_success_rates
# ---------------------------------------------------------------------------


def test_load_provider_success_missing_file(tmp_path):
    """Missing file → empty dict."""
    path = tmp_path / "no_such_file.jsonl"
    assert _load_provider_success_rates(path) == {}


def test_load_provider_success_valid(tmp_path):
    """Valid JSONL lines → success rates."""
    path = tmp_path / "history.jsonl"
    lines = [
        json.dumps({"provider": "kimi", "success": True}),
        json.dumps({"provider": "kimi", "success": True}),
        json.dumps({"provider": "kimi", "success": False}),
        json.dumps({"provider": "haiku", "success": True}),
    ]
    path.write_text("\n".join(lines))
    result = _load_provider_success_rates(path, min_samples=3)
    assert pytest.approx(result["kimi"]) == 2 / 3
    assert "haiku" not in result  # only 1 sample < min_samples


def test_load_provider_success_bad_json_skipped(tmp_path):
    """Bad JSON lines are skipped."""
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps({"provider": "kimi", "success": True}) + "\n"
        "not json\n"
        + json.dumps({"provider": "kimi", "success": True}) + "\n"
        + json.dumps({"provider": "kimi", "success": False}) + "\n"
    )
    result = _load_provider_success_rates(path, min_samples=3)
    assert pytest.approx(result["kimi"]) == 2 / 3


def test_load_provider_success_exception(tmp_path, caplog):
    """Unreadable file → empty dict and debug log."""
    path = tmp_path / "history.jsonl"
    path.write_text("x")
    path.chmod(0o000)
    try:
        with caplog.at_level("DEBUG", logger="mullet"):
            assert _load_provider_success_rates(path) == {}
        assert "failed to load history" in caplog.text
    finally:
        path.chmod(0o644)


# ---------------------------------------------------------------------------
# _load_recent_rate_limit_rates
# ---------------------------------------------------------------------------


def test_load_rl_rates_missing_file(tmp_path):
    """Missing file → empty dict."""
    path = tmp_path / "no_such_file.jsonl"
    assert _load_recent_rate_limit_rates(path) == {}


def test_load_rl_rates_valid(tmp_path):
    """Valid JSONL with rate_limit entries → rates."""
    path = tmp_path / "history.jsonl"
    lines = [
        json.dumps({"provider": "kimi", "error_class": None}),
        json.dumps({"provider": "kimi", "error_class": "rate_limit"}),
        json.dumps({"provider": "kimi", "error_class": "rate_limit"}),
        json.dumps({"provider": "kimi", "error_class": None}),
        json.dumps({"provider": "kimi", "error_class": None}),
    ]
    path.write_text("\n".join(lines))
    result = _load_recent_rate_limit_rates(path, window=20, min_samples=3)
    assert pytest.approx(result["kimi"]) == 2 / 5


def test_load_rl_rates_unknown_skipped(tmp_path):
    """Provider 'unknown' is skipped."""
    path = tmp_path / "history.jsonl"
    lines = [
        json.dumps({"provider": "unknown", "error_class": "rate_limit"}),
        json.dumps({"provider": "kimi", "error_class": None}),
        json.dumps({"provider": "kimi", "error_class": None}),
        json.dumps({"provider": "kimi", "error_class": "rate_limit"}),
    ]
    path.write_text("\n".join(lines))
    result = _load_recent_rate_limit_rates(path, window=20, min_samples=3)
    assert pytest.approx(result["kimi"]) == 1 / 3
    assert "unknown" not in result


def test_load_rl_rates_exception(tmp_path, caplog):
    """Unreadable file → empty dict and debug log."""
    path = tmp_path / "history.jsonl"
    path.write_text("x")
    path.chmod(0o000)
    try:
        with caplog.at_level("DEBUG", logger="mullet"):
            assert _load_recent_rate_limit_rates(path) == {}
        assert "rate-limit routing: failed to load history" in caplog.text
    finally:
        path.chmod(0o644)


# ---------------------------------------------------------------------------
# _load_provider_quality_data
# ---------------------------------------------------------------------------


def test_load_quality_data_missing_file(tmp_path):
    """Missing file → ({}, {})."""
    path = tmp_path / "no_such_file.jsonl"
    assert _load_provider_quality_data(path) == ({}, {})


def test_load_quality_data_empty_line_and_bad_json(tmp_path):
    """Empty lines and bad JSON are skipped."""
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps({"provider": "kimi", "success": True, "error_class": None}) + "\n"
        "\n"
        "bad json\n"
        + json.dumps({"provider": "kimi", "success": True, "error_class": None}) + "\n"
        + json.dumps({"provider": "kimi", "success": False, "error_class": "rate_limit"}) + "\n"
    )
    success, rl = _load_provider_quality_data(path, min_success_samples=2, min_rl_samples=2)
    assert pytest.approx(success["kimi"]) == 2 / 3
    assert pytest.approx(rl["kimi"]) == 1 / 3


def test_load_quality_data_exception(tmp_path, caplog):
    """Exception → ({}, {}) and debug log."""
    path = tmp_path / "history.jsonl"
    path.write_text("x")
    path.chmod(0o000)
    try:
        with caplog.at_level("DEBUG", logger="mullet"):
            assert _load_provider_quality_data(path) == ({}, {})
        assert "failed to load combined quality data" in caplog.text
    finally:
        path.chmod(0o644)


# ---------------------------------------------------------------------------
# Router.route — learned tier downgrade + exception handling
# ---------------------------------------------------------------------------


def _make_router(**kwargs) -> Router:
    default_providers = {
        "local": ProviderEntry(name="local", tier="free", models={"default": "llama-8b"}),
        "cheap_cloud": ProviderEntry(name="cheap_cloud", tier="cheap", models={"default": "kimi-k2.5"}),
        "premium_cloud": ProviderEntry(name="premium_cloud", tier="premium", models={"default": "claude-sonnet-4-6"}),
        "haiku": ProviderEntry(name="haiku", tier="cheap", models={"default": "claude-haiku-4-5"}),
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "moonshot-v1"}),
    }
    providers = kwargs.pop("providers", default_providers)
    return Router(providers=providers, **kwargs)


def test_route_learned_tier_downgrade():
    """suggest_tier returns a lower tier → router downgrades."""
    router = _make_router()
    with patch(
        "mullet.core.router.learning.suggest_tier",
        return_value="LOW",
    ):
        # Use a complex prompt that would normally score STANDARD or higher
        decision = router.route(
            "refactor the entire codebase with architecture redesign and migration",
        )
        assert decision.tier == Tier.LOW
        assert any("learned" in r for r in decision.reasons)


def test_route_learned_tier_exception_handled():
    """suggest_tier raises → routing continues normally."""
    router = _make_router()
    with patch(
        "mullet.core.router.learning.suggest_tier",
        side_effect=RuntimeError("learning broken"),
    ):
        decision = router.route("simple task")
        assert decision.provider is not None


# ---------------------------------------------------------------------------
# Router.route — budget-pressure routing
# ---------------------------------------------------------------------------


def test_route_pressure_routing_picks_budget_provider():
    """use_pressure_routing=True with no allowlist picks budget provider."""
    router = _make_router(use_pressure_routing=True)
    with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
        with patch("mullet.hud.guard.compute_pressure", return_value=0.75):
            with patch(
                "mullet.hud.guard.recommend_provider",
                return_value="haiku",
            ):
                decision = router.route("some task")
                assert decision.provider == "haiku"
                assert any("budget-pressure" in r for r in decision.reasons)


def test_route_pressure_routing_allowlist_fallback():
    """Budget provider not in allowlist → falls back to allowlisted provider.

    Phase B: pick_provider_for_tier returns the best tier-capable allowlisted provider
    when recommend_provider's suggestion (premium_cloud) is not in the allowlist.
    The old 'budget-pressure fallback to allowlist' reason was replaced by
    pick_provider_for_tier silently returning the best allowed provider, so the reason
    string is now 'budget-pressure <pressure> -> <provider>' — provider is from allowlist.
    """
    router = _make_router(use_pressure_routing=True)
    with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
        with patch("mullet.hud.guard.compute_pressure", return_value=0.75):
            with patch(
                "mullet.hud.guard.recommend_provider",
                return_value="premium_cloud",
            ):
                decision = router.route(
                    "some task",
                    allowed_providers=frozenset({"haiku", "kimi"}),
                )
                assert decision.provider in ("haiku", "kimi")
                # Phase B: reason now says "budget-pressure <p> -> <provider>" directly
                assert any("budget-pressure" in r for r in decision.reasons)


def test_route_pressure_routing_allowlist_no_match():
    """Phase E update: local is now CRITICAL-capable, so allowlist={local} + CRITICAL → local.

    Before Phase E: local could not serve CRITICAL → provider=None.
    After Phase E: local can serve every tier → provider='local'.

    The test name is kept for history; the behavior has changed intentionally.
    """
    router = _make_router(use_pressure_routing=True)
    with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
        with patch("mullet.hud.guard.compute_pressure", return_value=0.75):
            with patch(
                "mullet.hud.guard.recommend_provider",
                return_value="premium_cloud",
            ):
                # local is now CRITICAL-capable (Phase E) — must return local, not None
                decision = router.route(
                    "some task",
                    force_tier="CRITICAL",
                    allowed_providers=frozenset({"local"}),
                )
                assert decision.provider == "local", (
                    f"Phase E: local is CRITICAL-capable; expected 'local', got {decision.provider!r}"
                )


def test_route_pressure_routing_probe_fallback():
    """Budget provider blocked by probe → falls back to next candidate."""
    router = _make_router(use_pressure_routing=True)
    blocked = MagicMock()
    blocked.has_budget = False
    allowed = MagicMock()
    allowed.has_budget = True

    with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
        with patch("mullet.hud.guard.compute_pressure", return_value=0.75):
            with patch(
                "mullet.hud.guard.recommend_provider",
                return_value="kimi",
            ):
                import mullet.core.providers.budget_probe as _bp

                orig = getattr(_bp, "_cache", {})
                try:
                    _bp._cache = {"kimi": blocked, "haiku": allowed}
                    decision = router.route(
                        "some task",
                        allowed_providers=frozenset({"kimi", "haiku"}),
                    )
                    assert decision.provider == "haiku"
                finally:
                    _bp._cache = orig


def test_route_pressure_routing_exception_handled():
    """Exception in pressure routing block → handled gracefully."""
    router = _make_router(use_pressure_routing=True)
    with patch(
        "mullet.hud.usage.fetch_usage",
        side_effect=RuntimeError("usage broken"),
    ):
        decision = router.route("some task")
        assert decision.provider is not None


# ---------------------------------------------------------------------------
# Router.route — quality / rate-limit / budget skips (no allowlist)
# ---------------------------------------------------------------------------


def _make_router_for_quality_skip() -> Router:
    """Router whose STANDARD-tier providers are cheap_cloud + local."""
    providers = {
        "cheap_cloud": ProviderEntry(name="cheap_cloud", tier="cheap", models={"default": "kimi-k2.5"}),
        "local": ProviderEntry(name="local", tier="free", models={"default": "llama-8b"}),
    }
    return Router(providers=providers)


def test_route_quality_skip_and_second_pass():
    """Low success rate skips all STANDARD providers, second pass falls back."""
    router = _make_router_for_quality_skip()
    history_path = Path(__file__).with_suffix(".tmp_history.jsonl")
    history_path.write_text(
        "\n".join([
            json.dumps({"provider": "cheap_cloud", "success": False}),
            json.dumps({"provider": "cheap_cloud", "success": False}),
            json.dumps({"provider": "cheap_cloud", "success": False}),
            json.dumps({"provider": "local", "success": False}),
            json.dumps({"provider": "local", "success": False}),
            json.dumps({"provider": "local", "success": False}),
        ])
    )
    try:
        with patch(
            "mullet.core.router.decision._quality_history_path",
            return_value=history_path,
        ):
            with patch(
                "mullet.core.router.decision._quality_tracker_path",
                return_value=Path("/nonexistent"),
            ):
                decision = router.route("task", force_tier="STANDARD")
                assert decision.provider in ("cheap_cloud", "local")
                assert any("quality fallback" in r for r in decision.reasons)
    finally:
        history_path.unlink(missing_ok=True)


def test_route_user_rating_skip():
    """Low user-approval rating skips provider."""
    router = _make_router_for_quality_skip()
    quality_path = Path(__file__).with_suffix(".tmp_quality.json")
    quality_path.write_text(
        json.dumps({
            "cheap_cloud": {"up": 1, "down": 9},
            "local": {"up": 5, "down": 5},
        })
    )
    try:
        with patch(
            "mullet.core.router.decision._quality_history_path",
            return_value=Path("/nonexistent"),
        ):
            with patch(
                "mullet.core.router.decision._quality_tracker_path",
                return_value=quality_path,
            ):
                decision = router.route("task", force_tier="STANDARD")
                assert decision.provider == "local"
                assert any("user-rating skip: cheap_cloud" in r for r in decision.reasons)
    finally:
        quality_path.unlink(missing_ok=True)


def test_route_rate_limit_skip():
    """High rate-limit rate skips provider."""
    router = _make_router_for_quality_skip()
    history_path = Path(__file__).with_suffix(".tmp_history.jsonl")
    lines = [
        json.dumps({"provider": "cheap_cloud", "success": True, "error_class": "rate_limit"})
        for _ in range(10)
    ] + [
        json.dumps({"provider": "local", "success": True, "error_class": None})
        for _ in range(5)
    ]
    history_path.write_text("\n".join(lines))
    try:
        with patch(
            "mullet.core.router.decision._quality_history_path",
            return_value=history_path,
        ):
            with patch(
                "mullet.core.router.decision._quality_tracker_path",
                return_value=Path("/nonexistent"),
            ):
                decision = router.route("task", force_tier="STANDARD")
                assert decision.provider == "local"
                assert any("rate-limit skip: cheap_cloud" in r for r in decision.reasons)
    finally:
        history_path.unlink(missing_ok=True)


def test_route_budget_too_expensive_reason():
    """Budget tracker blocks provider → 'budget: too expensive' in reasons."""
    router = _make_router_for_quality_skip()
    budget = MagicMock()
    budget.can_afford.return_value = False
    router.budget = budget
    decision = router.route("task", force_tier="STANDARD")
    assert any("budget: cheap_cloud too expensive" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# Router.route — budget downgrade
# ---------------------------------------------------------------------------


def test_route_budget_downgrade():
    """All CRITICAL providers blocked by budget → downgrade to cheaper tier."""
    router = Router(
        providers={
            "premium_cloud": ProviderEntry(name="premium_cloud", tier="premium", models={"default": "claude-sonnet-4-6"}),
            "cheap_cloud": ProviderEntry(name="cheap_cloud", tier="cheap", models={"default": "kimi-k2.5"}),
            "local": ProviderEntry(name="local", tier="free", models={"default": "llama-8b"}),
        },
        budget_tracker=MagicMock(),
    )
    router.budget.can_afford.side_effect = lambda p, m, t: p == "local"
    decision = router.route("task", force_tier="CRITICAL")
    assert decision.provider == "local"
    # Downgrade hits STANDARD first (local is in STANDARD tier_map)
    assert decision.tier == Tier.STANDARD
    assert any("budget downgrade: CRITICAL -> STANDARD" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# Router.route — final return "none"
# ---------------------------------------------------------------------------


def test_route_final_none_when_no_healthy_provider():
    """All providers unhealthy and no downgrade → 'none' provider."""
    router = Router(
        providers={
            "premium_cloud": ProviderEntry(name="premium_cloud", tier="premium", models={"default": "claude-sonnet-4-6"}, healthy=False),
            "cheap_cloud": ProviderEntry(name="cheap_cloud", tier="cheap", models={"default": "kimi-k2.5"}, healthy=False),
        },
    )
    decision = router.route("task", force_tier="CRITICAL")
    assert decision.provider == "none"
    assert decision.confidence == 0.0
    assert any("ERROR: no providers available" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# Extra coverage for _load_provider_success_rates empty lines
# ---------------------------------------------------------------------------


def test_load_provider_success_empty_line_skipped(tmp_path):
    """Empty lines inside JSONL are skipped."""
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps({"provider": "kimi", "success": True}) + "\n"
        "\n"
        + json.dumps({"provider": "kimi", "success": True}) + "\n"
        + json.dumps({"provider": "kimi", "success": False}) + "\n"
    )
    result = _load_provider_success_rates(path, min_samples=3)
    assert pytest.approx(result["kimi"]) == 2 / 3


# ---------------------------------------------------------------------------
# Extra coverage for _load_recent_rate_limit_rates empty / short
# ---------------------------------------------------------------------------


def test_load_rl_rates_empty_line_and_short_window(tmp_path):
    """Empty lines skipped; window with < min_samples yields no entry."""
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps({"provider": "kimi", "error_class": None}) + "\n"
        "\n"
        "bad json\n"
        + json.dumps({"provider": "kimi", "error_class": None}) + "\n"
    )
    result = _load_recent_rate_limit_rates(path, window=20, min_samples=5)
    assert result == {}


# ---------------------------------------------------------------------------
# Router.record_usage + _is_healthy branches
# ---------------------------------------------------------------------------


def test_record_usage_with_budget_tracker():
    """record_usage delegates to budget_tracker when present."""
    router = _make_router()
    budget = MagicMock()
    router.budget = budget
    router.record_usage("kimi", 1000, 0.05, model="moonshot-v1")
    budget.record_usage.assert_called_once_with("kimi", "moonshot-v1", 1000, 0)


def test_is_healthy_with_health_tracker():
    """_is_healthy delegates to health tracker when present."""
    router = _make_router()
    health = MagicMock()
    health.is_available.return_value = False
    router.health = health
    entry = ProviderEntry(name="kimi", tier="cheap")
    assert router._is_healthy(entry) is False
    health.is_available.assert_called_once_with("kimi")


def test_is_healthy_backoff_active():
    """Provider in active backoff period → unhealthy."""
    router = _make_router()
    entry = ProviderEntry(name="kimi", tier="cheap", backoff_until=time.time() + 60)
    assert router._is_healthy(entry) is False


# ---------------------------------------------------------------------------
# Router._analyze_complexity — non-simple message branch
# ---------------------------------------------------------------------------


def test_analyze_complexity_non_simple():
    """Complex prompt triggers full signal extraction and scoring."""
    router = _make_router()
    reasons: list[str] = []
    tier = router._analyze_complexity(
        "refactor the entire codebase with architecture redesign and migration across multiple files",
        reasons,
    )
    assert tier in (Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL)
    assert any("complexity score:" in r for r in reasons)


def test_route_local_only_mode():
    """local_only=True routes to local provider."""
    router = Router(
        providers={"local": ProviderEntry(name="local", tier="free", models={"default": "llama-8b"})},
        local_only=True,
    )
    decision = router.route("task")
    assert decision.provider == "local"
    assert decision.tier == Tier.LOW


def test_route_local_only_mode_no_local_provider():
    """local_only=True with no local provider → none."""
    router = Router(providers={}, local_only=True)
    decision = router.route("task")
    assert decision.provider == "none"
    assert decision.confidence == 0.0


def test_route_force_provider():
    """force_provider bypasses complexity analysis."""
    router = _make_router()
    decision = router.route("task", force_provider="cheap_cloud")
    assert decision.provider == "cheap_cloud"
    assert any("forced provider: cheap_cloud" in r for r in decision.reasons)


def test_route_force_provider_rejected_by_allowlist():
    """force_provider rejected when not in allowlist."""
    router = _make_router()
    decision = router.route("task", force_provider="cheap_cloud", allowed_providers=frozenset({"kimi"}))
    assert decision.provider is None
    assert any("rejected by allowlist" in r for r in decision.reasons)


def test_route_agent_type_override():
    """Agent type mapped to a fixed tier."""
    router = _make_router()
    decision = router.route("task", agent_type="explore")
    assert decision.tier == Tier.LOW
    assert any("agent override: explore" in r for r in decision.reasons)


def test_route_delegation_basic():
    """Delegation routes one tier below parent."""
    router = _make_router()
    decision = router.route_delegation("CRITICAL", "simple task")
    assert decision.tier == Tier.ADVANCED


def test_mark_failure_and_success():
    """mark_failure sets healthy=False and backoff; mark_success resets."""
    router = _make_router()
    entry = router.providers["kimi"]
    router.mark_failure("kimi", RuntimeError("boom"))
    assert entry.healthy is False
    assert entry.backoff_until > time.time()
    router.mark_success("kimi")
    assert entry.healthy is True
    assert entry.backoff_until == 0


def test_route_budget_downgrade_skips_non_allowlisted():
    """Budget downgrade candidates not in allowed_providers are skipped."""
    router = Router(
        providers={
            "premium_cloud": ProviderEntry(name="premium_cloud", tier="premium", models={"default": "claude-sonnet-4-6"}, healthy=False),
            "cheap_cloud": ProviderEntry(name="cheap_cloud", tier="cheap", models={"default": "kimi-k2.5"}, healthy=True),
            "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "moonshot-v1"}, healthy=False),
        },
    )
    decision = router.route("task", force_tier="CRITICAL", allowed_providers=frozenset({"kimi"}))
    assert decision.provider == "none"
    assert any("ERROR: no providers available" in r for r in decision.reasons)


def test_route_no_healthy_provider_fallback_skips_non_allowlisted():
    """No-healthy-provider fallback skips providers outside allowed_providers."""
    router = Router(
        providers={
            "premium_cloud": ProviderEntry(name="premium_cloud", tier="premium", models={"default": "claude-sonnet-4-6"}, healthy=False),
            "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "moonshot-v1"}, healthy=True),
        },
    )
    decision = router.route("task", force_tier="CRITICAL", allowed_providers=frozenset({"premium_cloud"}))
    assert decision.provider == "none"
    assert any("ERROR: no providers available" in r for r in decision.reasons)


def test_route_agent_type_override_none():
    """Agent type mapped to None override falls back to complexity analysis."""
    router = _make_router()
    decision = router.route("task", agent_type="debugger")
    assert not any("agent override" in r for r in decision.reasons)
    assert any("simple message (fast pre-check)" in r for r in decision.reasons)


def test_route_g9_allowlist_no_match():
    """Phase E update: local is now CRITICAL-capable, so {local} for CRITICAL → local.

    Before Phase E: local could not serve CRITICAL → provider=None with "allowlist" reason.
    After Phase E: local can serve every tier → provider='local'.

    To get a true "no match" scenario, use a provider that genuinely has no CRITICAL capability.
    """
    router = _make_router()
    # After Phase E, local IS CRITICAL-capable. Use a different, non-tier-capable provider.
    # Add a dummy entry that has no PROVIDER_TIER_CAPABILITIES entry.
    from mullet.core.router.decision import ProviderEntry
    router.providers["unknown_cloud"] = ProviderEntry(name="unknown_cloud", tier="cheap", models={"default": "x"})
    decision = router.route("task", force_tier="CRITICAL", allowed_providers=frozenset({"unknown_cloud"}))
    assert decision.provider is None, (
        f"unknown_cloud has no CRITICAL capability, expected None, got {decision.provider!r}"
    )
    assert any("allowlist" in r for r in decision.reasons)


def test_route_g9_allowlist_tier_pick_success():
    """Allowlisted provider can serve tier and is healthy → picked."""
    router = _make_router()
    decision = router.route("task", force_tier="CRITICAL", allowed_providers=frozenset({"kimi"}))
    assert decision.provider == "kimi"
    assert any("allowlist tier pick" in r for r in decision.reasons)


def test_route_delegation_inferred_allowed_tiers(caplog):
    """route_delegation infers allowed_tiers from allowed_providers and logs it."""
    router = _make_router()
    with caplog.at_level("INFO", logger="mullet"):
        decision = router.route_delegation("STANDARD", "task", allowed_providers=frozenset({"local"}))
    assert "inferred allowed_tiers" in caplog.text
    assert decision.provider == "local"
