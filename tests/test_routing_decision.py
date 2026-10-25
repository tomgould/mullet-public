import json
import time
from unittest.mock import MagicMock, patch


from mullet.core.router.decision import (
    ProviderEntry,
    Router,
    RATE_LIMIT_WINDOW,
    _load_provider_quality_data,
)


def _make_router(**kwargs):
    default_providers = {
        "local": ProviderEntry(name="local", tier="free", models={"default": "llama-8b"}),
        "gemini": ProviderEntry(name="gemini", tier="cheap", models={"default": "gemini-flash"}),
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.5"}),
        "haiku": ProviderEntry(name="haiku", tier="cheap", models={"default": "claude-haiku"}),
        "sonnet": ProviderEntry(name="sonnet", tier="premium", models={"default": "claude-sonnet"}),
    }
    providers = kwargs.pop("providers", default_providers)
    return Router(providers=providers, **kwargs)


def test_find_provider_existing():
    router = _make_router()
    assert router._find_provider("kimi").name == "kimi"


def test_find_provider_missing():
    router = _make_router()
    assert router._find_provider("missing") is None


def test_is_healthy_with_backoff():
    router = _make_router()
    entry = router.providers["kimi"]
    entry.healthy = True
    entry.backoff_until = time.time() + 3600
    assert router._is_healthy(entry) is False


def test_is_healthy_backoff_expired():
    router = _make_router()
    entry = router.providers["kimi"]
    entry.healthy = True
    entry.backoff_until = time.time() - 1
    assert router._is_healthy(entry) is True


def test_is_healthy_with_health_tracker():
    router = _make_router()
    mock_health = MagicMock()
    mock_health.is_available.return_value = False
    router.health = mock_health
    entry = router.providers["kimi"]
    assert router._is_healthy(entry) is False
    mock_health.is_available.assert_called_once_with("kimi")


def test_check_budget_no_budget():
    router = _make_router()
    entry = router.providers["kimi"]
    assert router._check_budget(entry) is True


def test_check_budget_denied():
    router = _make_router()
    mock_budget = MagicMock()
    mock_budget.can_afford.return_value = False
    router.budget = mock_budget
    entry = router.providers["kimi"]
    assert router._check_budget(entry) is False


def test_estimate_cost_without_budget():
    router = _make_router()
    entry = router.providers["kimi"]
    entry.cost_per_1k_input = 1.0
    entry.cost_per_1k_output = 3.0
    assert router._estimate_cost(entry, 2000) == 4.0


def test_estimate_cost_with_budget():
    router = _make_router()
    mock_budget = MagicMock()
    mock_budget.estimate_cost.return_value = 0.5
    router.budget = mock_budget
    entry = router.providers["kimi"]
    assert router._estimate_cost(entry, 2000) == 0.5
    mock_budget.estimate_cost.assert_called_once_with("kimi", "kimi-k2.5", 2000)


def test_pick_allowlisted_provider_basic():
    router = _make_router()
    result = router._pick_allowlisted_provider(frozenset({"kimi", "sonnet"}))
    assert result == "kimi"


def test_pick_allowlisted_provider_tier_filtered():
    router = _make_router()
    result = router._pick_allowlisted_provider(frozenset({"sonnet"}), tier_value="CRITICAL")
    assert result == "sonnet"


def test_pick_allowlisted_provider_no_match():
    # Phase E: local is now CRITICAL-capable, so {gemini, local} for CRITICAL returns 'local'.
    # gemini is also CRITICAL-capable? No — gemini caps at ADVANCED. But local now caps at CRITICAL.
    # So this is no longer a "no match" case — local matches. Test updated to reflect new behavior.
    router = _make_router()
    result = router._pick_allowlisted_provider(frozenset({"gemini", "local"}), tier_value="CRITICAL")
    # local is CRITICAL-capable after Phase E, so it must be returned
    assert result == "local", (
        f"Phase E: local is CRITICAL-capable; expected 'local' for {{gemini, local}} @ CRITICAL, "
        f"got {result!r}"
    )


def test_record_usage_no_budget():
    router = _make_router()
    router.record_usage("kimi", 100, 0.01)
    assert router.budget is None


def test_record_usage_with_budget():
    router = _make_router()
    mock_budget = MagicMock()
    router.budget = mock_budget
    router.record_usage("kimi", 100, 0.01, model="kimi-k2.5")
    mock_budget.record_usage.assert_called_once_with("kimi", "kimi-k2.5", 100, 0)


def test_load_provider_quality_data_empty(tmp_path):
    path = tmp_path / "routing_history.jsonl"
    path.write_text("")
    assert _load_provider_quality_data(path) == ({}, {})


def test_load_provider_quality_data_success_rates(tmp_path):
    path = tmp_path / "routing_history.jsonl"
    lines = [
        json.dumps({"provider": "kimi", "success": True}),
        json.dumps({"provider": "kimi", "success": False}),
        json.dumps({"provider": "kimi", "success": True}),
        json.dumps({"provider": "sonnet", "success": True}),
    ]
    path.write_text("\n".join(lines) + "\n")
    success_rates, rl_rates = _load_provider_quality_data(path, min_success_samples=3, min_rl_samples=5)
    assert success_rates == {"kimi": 2 / 3}
    assert rl_rates == {}


def test_load_provider_quality_data_rl_window(tmp_path):
    path = tmp_path / "routing_history.jsonl"
    lines = []
    for i in range(RATE_LIMIT_WINDOW + 5):
        if i < 5:
            ec = "rate_limit"
        else:
            ec = None
        lines.append(json.dumps({"provider": "kimi", "success": True, "error_class": ec}))
    path.write_text("\n".join(lines) + "\n")
    success_rates, rl_rates = _load_provider_quality_data(path, min_success_samples=3, min_rl_samples=5)
    assert success_rates == {"kimi": 1.0}
    # last window = RATE_LIMIT_WINDOW items, items 5..end have no rate limit -> 0 rate limit in window
    assert rl_rates == {"kimi": 0.0}


def test_load_provider_quality_data_rl_counts(tmp_path):
    path = tmp_path / "routing_history.jsonl"
    lines = []
    for i in range(10):
        ec = "rate_limit" if i >= 8 else None
        lines.append(json.dumps({"provider": "kimi", "success": True, "error_class": ec}))
    path.write_text("\n".join(lines) + "\n")
    success_rates, rl_rates = _load_provider_quality_data(path, min_success_samples=3, min_rl_samples=5)
    assert success_rates == {"kimi": 1.0}
    # last 10 items: 2 rate limits -> 0.2
    assert rl_rates == {"kimi": 0.2}


def test_load_provider_quality_data_unknown_filtered(tmp_path):
    path = tmp_path / "routing_history.jsonl"
    lines = [
        json.dumps({"provider": "unknown", "success": False, "error_class": "rate_limit"}),
        json.dumps({"provider": "unknown", "success": False, "error_class": "rate_limit"}),
        json.dumps({"provider": "unknown", "success": False, "error_class": "rate_limit"}),
        json.dumps({"provider": "kimi", "success": True, "error_class": None}),
        json.dumps({"provider": "kimi", "success": True, "error_class": None}),
        json.dumps({"provider": "kimi", "success": True, "error_class": "rate_limit"}),
    ]
    path.write_text("\n".join(lines) + "\n")
    success_rates, rl_rates = _load_provider_quality_data(path, min_success_samples=3, min_rl_samples=3)
    # unknown should appear in success_rates because _load_provider_success_rates includes it
    assert "unknown" in success_rates
    assert success_rates["unknown"] == 0.0
    # unknown should NOT appear in rl_rates
    assert "unknown" not in rl_rates
    assert rl_rates == {"kimi": 1 / 3}


# ---------------------------------------------------------------------------
# _pick_cheap_cloud_peer
# ---------------------------------------------------------------------------


def test_pick_cheap_cloud_peer_no_history():
    router = _make_router()
    with patch("mullet.core.router.decision._load_provider_quality_data", return_value=({}, {})):
        assert router._pick_cheap_cloud_peer("kimi", "openrouter") == "kimi"


def test_pick_cheap_cloud_peer_kimi_rate_limited():
    router = _make_router()
    with patch(
        "mullet.core.router.decision._load_provider_quality_data",
        return_value=({"kimi": 0.9, "openrouter": 0.9}, {"kimi": 0.5, "openrouter": 0.0}),
    ):
        assert router._pick_cheap_cloud_peer("kimi", "openrouter") == "openrouter"


def test_pick_cheap_cloud_peer_openrouter_rate_limited():
    router = _make_router()
    with patch(
        "mullet.core.router.decision._load_provider_quality_data",
        return_value=({"kimi": 0.9, "openrouter": 0.9}, {"kimi": 0.0, "openrouter": 0.5}),
    ):
        assert router._pick_cheap_cloud_peer("kimi", "openrouter") == "kimi"


def test_pick_cheap_cloud_peer_openrouter_better_quality():
    router = _make_router()
    with patch(
        "mullet.core.router.decision._load_provider_quality_data",
        return_value=({"kimi": 0.5, "openrouter": 0.9}, {"kimi": 0.0, "openrouter": 0.0}),
    ):
        assert router._pick_cheap_cloud_peer("kimi", "openrouter") == "openrouter"


def test_pick_cheap_cloud_peer_kimi_better_quality():
    router = _make_router()
    with patch(
        "mullet.core.router.decision._load_provider_quality_data",
        return_value=({"kimi": 0.9, "openrouter": 0.5}, {"kimi": 0.0, "openrouter": 0.0}),
    ):
        assert router._pick_cheap_cloud_peer("kimi", "openrouter") == "kimi"


def test_pick_cheap_cloud_peer_tied_quality():
    router = _make_router()
    with patch(
        "mullet.core.router.decision._load_provider_quality_data",
        return_value=({"kimi": 0.9, "openrouter": 0.9}, {"kimi": 0.0, "openrouter": 0.0}),
    ):
        assert router._pick_cheap_cloud_peer("kimi", "openrouter") == "kimi"


# ---------------------------------------------------------------------------
# _pick_allowlisted_provider with cheap-cloud peer tie-break
# ---------------------------------------------------------------------------


def test_pick_allowlisted_provider_only_kimi_eligible():
    router = _make_router()
    with patch("mullet.core.router.decision._load_provider_quality_data", return_value=({}, {})):
        result = router._pick_allowlisted_provider(frozenset({"kimi", "sonnet"}))
        assert result == "kimi"


def test_pick_allowlisted_provider_openrouter_not_registered():
    router = _make_router(providers={
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.5"}),
        "sonnet": ProviderEntry(name="sonnet", tier="premium", models={"default": "claude-sonnet"}),
    })
    with patch("mullet.core.router.decision._load_provider_quality_data", return_value=({}, {})):
        result = router._pick_allowlisted_provider(frozenset({"kimi", "sonnet", "openrouter"}))
        assert result == "kimi"


def test_pick_allowlisted_provider_openrouter_not_tier_capable():
    router = _make_router()
    # openrouter is not CRITICAL-capable
    with patch("mullet.core.router.decision._load_provider_quality_data", return_value=({}, {})):
        result = router._pick_allowlisted_provider(frozenset({"kimi", "openrouter"}), tier_value="CRITICAL")
        assert result == "kimi"


def test_pick_allowlisted_provider_both_eligible_no_history():
    router = _make_router(providers={
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.5"}),
        "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek-v4"}),
    })
    with patch("mullet.core.router.decision._load_provider_quality_data", return_value=({}, {})):
        result = router._pick_allowlisted_provider(frozenset({"kimi", "openrouter"}))
        assert result == "kimi"


def test_pick_allowlisted_provider_both_eligible_kimi_rate_limited():
    router = _make_router(providers={
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.5"}),
        "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek-v4"}),
    })
    with patch(
        "mullet.core.router.decision._load_provider_quality_data",
        return_value=({"kimi": 0.9, "openrouter": 0.9}, {"kimi": 0.5, "openrouter": 0.0}),
    ):
        result = router._pick_allowlisted_provider(frozenset({"kimi", "openrouter"}))
        assert result == "openrouter"


def test_pick_allowlisted_provider_both_eligible_openrouter_rate_limited():
    router = _make_router(providers={
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.5"}),
        "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek-v4"}),
    })
    with patch(
        "mullet.core.router.decision._load_provider_quality_data",
        return_value=({"kimi": 0.9, "openrouter": 0.9}, {"kimi": 0.0, "openrouter": 0.5}),
    ):
        result = router._pick_allowlisted_provider(frozenset({"kimi", "openrouter"}))
        assert result == "kimi"


def test_pick_allowlisted_provider_both_eligible_openrouter_better_quality():
    router = _make_router(providers={
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.5"}),
        "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek-v4"}),
    })
    with patch(
        "mullet.core.router.decision._load_provider_quality_data",
        return_value=({"kimi": 0.5, "openrouter": 0.9}, {"kimi": 0.0, "openrouter": 0.0}),
    ):
        result = router._pick_allowlisted_provider(frozenset({"kimi", "openrouter"}))
        assert result == "openrouter"


def test_pick_allowlisted_provider_both_eligible_tied_quality():
    router = _make_router(providers={
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.5"}),
        "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek-v4"}),
    })
    with patch(
        "mullet.core.router.decision._load_provider_quality_data",
        return_value=({"kimi": 0.9, "openrouter": 0.9}, {"kimi": 0.0, "openrouter": 0.0}),
    ):
        result = router._pick_allowlisted_provider(frozenset({"kimi", "openrouter"}))
        assert result == "kimi"
