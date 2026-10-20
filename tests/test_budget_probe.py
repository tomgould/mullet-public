"""Tests for mullet/core/providers/budget_probe.py."""
from __future__ import annotations

import asyncio
import time
import unittest.mock as mock
from unittest.mock import AsyncMock, patch

import pytest

from mullet.core.providers.budget_probe import (
    ProviderBalance,
    _BudgetCache,
    _cache,
    get_available_providers,
    parallel_provider_pairs,
    probe_anthropic,
    probe_kimi,
    probe_local,
    probe_openrouter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_balance(
    *,
    credits_remaining_usd: float | None = None,
    is_subscription: bool = False,
    is_free: bool = False,
    subscription_pressure: float | None = None,
    billing_model: str | None = None,
    error: str | None = None,
    key: str = "test",
    name: str = "Test",
) -> ProviderBalance:
    # Auto-derive billing_model from flags when not specified
    if billing_model is None:
        billing_model = "free" if is_free else ("subscription_cli" if is_subscription else "api")
    return ProviderBalance(
        name=name,
        key=key,
        credits_remaining_usd=credits_remaining_usd,
        credits_used_usd=None,
        credits_total_usd=None,
        is_subscription=is_subscription,
        is_free=is_free,
        billing_model=billing_model,
        subscription_pressure=subscription_pressure,
        error=error,
    )


# ---------------------------------------------------------------------------
# 1. ProviderBalance dataclass logic — pure unit tests
# ---------------------------------------------------------------------------


class TestProviderBalanceHasBudget:
    """Tests for ProviderBalance.has_budget property."""

    def test_has_budget_true_when_credits_remaining_positive(self):
        bal = _make_balance(credits_remaining_usd=10.0)
        assert bal.has_budget is True

    def test_has_budget_false_when_credits_remaining_zero(self):
        bal = _make_balance(credits_remaining_usd=0.0)
        assert bal.has_budget is False

    def test_has_budget_false_when_credits_remaining_negative(self):
        bal = _make_balance(credits_remaining_usd=-5.0)
        assert bal.has_budget is False

    def test_has_budget_true_when_is_free_regardless_of_credits(self):
        bal = _make_balance(is_free=True, credits_remaining_usd=0.0)
        assert bal.has_budget is True
        bal2 = _make_balance(is_free=True, credits_remaining_usd=-100.0)
        assert bal2.has_budget is True

    def test_has_budget_true_subscription_when_pressure_below_90(self):
        bal = _make_balance(is_subscription=True, subscription_pressure=0.89)
        assert bal.has_budget is True

    def test_has_budget_false_subscription_when_pressure_at_90(self):
        bal = _make_balance(is_subscription=True, subscription_pressure=0.90)
        assert bal.has_budget is False

    def test_has_budget_false_subscription_when_pressure_above_90(self):
        bal = _make_balance(is_subscription=True, subscription_pressure=0.95)
        assert bal.has_budget is False

    def test_has_budget_false_when_5h_raw_pct_at_90(self):
        bal = _make_balance(is_subscription=True, subscription_pressure=0.5)
        bal.extra["five_hour_pct"] = 90.0
        assert bal.has_budget is False

    def test_has_budget_false_when_weekly_raw_pct_at_90(self):
        bal = _make_balance(is_subscription=True, subscription_pressure=0.5)
        bal.extra["weekly_pct"] = 92.0
        assert bal.has_budget is False

    def test_has_budget_true_when_both_windows_below_90(self):
        bal = _make_balance(is_subscription=True, subscription_pressure=0.5)
        bal.extra["five_hour_pct"] = 80.0
        bal.extra["weekly_pct"] = 70.0
        assert bal.has_budget is True

    def test_has_budget_fails_open_when_error_set(self):
        bal = _make_balance(error="something broke", credits_remaining_usd=0.0)
        assert bal.has_budget is True


class TestProviderBalancePriorityScore:
    """Tests for ProviderBalance.priority_score property."""

    def test_priority_score_zero_when_has_budget_false(self):
        bal = _make_balance(credits_remaining_usd=0.0)
        assert bal.priority_score == pytest.approx(0.0)

    def test_priority_score_half_for_local_free(self):
        bal = _make_balance(is_free=True)
        assert bal.priority_score == pytest.approx(0.5)

    def test_priority_score_decreases_as_subscription_pressure_increases(self):
        low = _make_balance(is_subscription=True, subscription_pressure=0.1)
        mid = _make_balance(is_subscription=True, subscription_pressure=0.5)
        high = _make_balance(is_subscription=True, subscription_pressure=0.89)
        # Ordering holds: low pressure → high score
        assert low.priority_score > mid.priority_score > high.priority_score
        # Values from formula: max(0.1, 1.0 - pressure**0.7)
        assert low.priority_score == pytest.approx(1.0 - 0.1**0.7, rel=1e-4)
        assert high.priority_score == pytest.approx(max(0.1, 1.0 - 0.89**0.7), rel=1e-4)

    def test_priority_score_unknown_balance_no_error(self):
        bal = _make_balance()  # api, no credits = unknown balance
        assert bal.priority_score == pytest.approx(0.65)

    def test_priority_score_credit_based(self):
        import math
        bal = _make_balance(credits_remaining_usd=100.0)
        # Formula: min(0.85, 0.5 + 0.35 * log1p(100) / log1p(200))
        expected = min(0.85, 0.5 + 0.35 * math.log1p(100) / math.log1p(200))
        assert bal.priority_score == pytest.approx(expected, rel=1e-4)
        bal2 = _make_balance(credits_remaining_usd=1.0)
        assert bal2.priority_score > 0.0
        bal3 = _make_balance(credits_remaining_usd=0.0)
        assert bal3.priority_score == pytest.approx(0.0)


class TestProviderBalanceIsLow:
    """Tests for ProviderBalance.is_low property."""

    def test_is_low_true_when_remaining_below_50_cents(self):
        bal = _make_balance(credits_remaining_usd=0.49)
        assert bal.is_low is True

    def test_is_low_false_when_remaining_at_50_cents(self):
        bal = _make_balance(credits_remaining_usd=0.50)
        assert bal.is_low is False

    def test_is_low_false_when_remaining_above_50_cents(self):
        bal = _make_balance(credits_remaining_usd=10.0)
        assert bal.is_low is False

    def test_is_low_false_when_remaining_none(self):
        bal = _make_balance(credits_remaining_usd=None)
        assert bal.is_low is False


class TestProviderBalanceSummaryLine:
    """Tests for ProviderBalance.summary_line() method."""

    def test_summary_line_contains_dollar_for_known_balance(self):
        bal = _make_balance(credits_remaining_usd=12.34)
        line = bal.summary_line()
        assert "$" in line
        assert "12.34" in line

    def test_summary_line_shows_subscription_for_subscription_providers(self):
        bal = _make_balance(is_subscription=True, subscription_pressure=0.5)
        line = bal.summary_line()
        assert "subscription" in line

    def test_summary_line_shows_free_for_local_providers(self):
        bal = _make_balance(is_free=True)
        line = bal.summary_line()
        assert "free" in line

    def test_summary_line_shows_error_when_error_set(self):
        bal = _make_balance(error="network timeout")
        line = bal.summary_line()
        assert "error" in line
        assert "network timeout" in line

    def test_summary_line_subscription_no_pressure(self):
        bal = _make_balance(is_subscription=True, subscription_pressure=None)
        line = bal.summary_line()
        assert "subscription active" in line

    def test_summary_line_unknown_balance(self):
        bal = _make_balance()
        line = bal.summary_line()
        assert "balance unknown" in line

    def test_summary_line_low_flag(self):
        bal = _make_balance(credits_remaining_usd=0.10)
        line = bal.summary_line()
        assert "LOW" in line


# ---------------------------------------------------------------------------
# 2. Probe functions with mocked HTTP
# ---------------------------------------------------------------------------


class TestProbeOpenRouter:
    def test_probe_openrouter_returns_error_when_api_key_not_set(self):
        with patch.dict("os.environ", {}, clear=True):
            result = asyncio.run(probe_openrouter())
        assert result.error is not None
        assert "OPENROUTER_API_KEY not set" in result.error
        assert result.key == "openrouter"

    def test_probe_openrouter_with_mocked_response(self):
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "usage": 5.0,
                "limit": 100.0,
                "is_free_tier": False,
                "rate_limit": {"requests": 60},
                "label": "test-key",
            }
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(probe_openrouter(api_key="sk-test"))

        assert result.error is None
        assert result.credits_remaining_usd == pytest.approx(95.0)
        assert result.credits_used_usd == pytest.approx(5.0)
        assert result.credits_total_usd == pytest.approx(100.0)
        assert result.rate_limit_rpm == 60


class TestProbeKimi:
    def test_probe_kimi_returns_error_when_api_key_not_set(self):
        with patch.dict("os.environ", {}, clear=True):
            result = asyncio.run(probe_kimi())
        assert result.error is not None
        assert "KIMI_API_KEY not set" in result.error
        assert result.key == "kimi"

    def test_probe_kimi_usd_balance(self):
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "available_balance": "170.00",
                "currency": "USD",
            }
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(probe_kimi(api_key="sk-test"))

        assert result.error is None
        assert result.credits_remaining_usd == pytest.approx(170.0)
        assert result.extra.get("currency") == "USD"

    def test_probe_kimi_cny_conversion(self):
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "balance": {
                "available_balance": "72.00",
                "currency": "CNY",
            }
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(probe_kimi(api_key="sk-test"))

        assert result.error is None
        # 72.00 CNY ÷ 7.25 = 9.931... USD
        assert result.credits_remaining_usd == pytest.approx(72.0 / 7.25, rel=1e-3)


class TestProbeAnthropic:
    def test_probe_anthropic_returns_subscription_balance(self):
        fake_result = mock.MagicMock()
        fake_result.pressure = 0.35
        fake_result.five_hour_pct = 30.0
        fake_result.weekly_pct = 25.0
        fake_result.status.value = "go"
        fake_result.reset_time = ""

        with patch(
            "mullet.hud.guard.check_budget",
            return_value=fake_result,
        ), patch(
            "mullet.hud.usage.fetch_usage",
            return_value=None,
        ):
            result = asyncio.run(probe_anthropic())

        assert result.is_subscription is True
        assert result.subscription_pressure == pytest.approx(0.35)
        assert result.error is None
        assert result.key == "anthropic"

    def test_name_contains_times_five_when_has_sonnet_limit_true(self):
        fake_result = mock.MagicMock()
        fake_result.pressure = 0.35
        fake_result.five_hour_pct = 30.0
        fake_result.weekly_pct = 25.0
        fake_result.status.value = "go"
        fake_result.reset_time = ""

        fake_limits = mock.MagicMock()
        fake_limits.has_sonnet_limit = True
        fake_limits.sonnet_weekly_pct = 15.0

        with patch(
            "mullet.hud.guard.check_budget",
            return_value=fake_result,
        ), patch(
            "mullet.hud.usage.fetch_usage",
            return_value=fake_limits,
        ):
            result = asyncio.run(probe_anthropic())

        assert "×5" in result.name
        assert result.is_subscription is True
        assert result.error is None

    def test_name_contains_times_one_when_has_sonnet_limit_false(self):
        fake_result = mock.MagicMock()
        fake_result.pressure = 0.35
        fake_result.five_hour_pct = 30.0
        fake_result.weekly_pct = 25.0
        fake_result.status.value = "go"
        fake_result.reset_time = ""

        fake_limits = mock.MagicMock()
        fake_limits.has_sonnet_limit = False
        fake_limits.sonnet_weekly_pct = 0.0

        with patch(
            "mullet.hud.guard.check_budget",
            return_value=fake_result,
        ), patch(
            "mullet.hud.usage.fetch_usage",
            return_value=fake_limits,
        ):
            result = asyncio.run(probe_anthropic())

        assert "×1" in result.name
        assert result.is_subscription is True
        assert result.error is None


class TestProbeLocal:
    def test_probe_local_returns_is_free_true_when_reachable(self):
        mock_response = mock.MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(probe_local())

        assert result.is_free is True
        assert result.error is None
        assert result.extra.get("online") is True

    def test_probe_local_sets_error_when_server_unreachable(self):
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(probe_local())

        assert result.is_free is True
        assert result.error is not None
        assert "server not reachable" in result.error
        assert result.extra.get("online") is False


# ---------------------------------------------------------------------------
# 3. Routing helpers
# ---------------------------------------------------------------------------


class TestGetAvailableProviders:
    def test_filters_out_providers_without_budget(self):
        balances = {
            "kimi": _make_balance(key="kimi", credits_remaining_usd=10.0),
            "openrouter": _make_balance(key="openrouter", credits_remaining_usd=0.0),
            "local": _make_balance(key="local", is_free=True),
        }
        with patch(
            "mullet.core.providers.budget_probe.get_all_balances",
            return_value=balances,
        ):
            result = get_available_providers(["kimi", "openrouter", "local"])

        assert "kimi" in result
        assert "local" in result
        assert "openrouter" not in result

    def test_keeps_unknown_providers(self):
        """Providers not in probe results (None) are kept as available."""
        balances = {
            "kimi": _make_balance(key="kimi", credits_remaining_usd=10.0),
        }
        with patch(
            "mullet.core.providers.budget_probe.get_all_balances",
            return_value=balances,
        ):
            result = get_available_providers(["kimi", "unknown"])

        assert "kimi" in result
        assert "unknown" in result


class TestParallelProviderPairs:
    def test_distributes_tasks_round_robin(self):
        balances = {
            "kimi": _make_balance(key="kimi", credits_remaining_usd=10.0),
            "openrouter": _make_balance(key="openrouter", credits_remaining_usd=10.0),
            "local": _make_balance(key="local", is_free=True),
        }
        with patch(
            "mullet.core.providers.budget_probe.get_all_balances",
            return_value=balances,
        ):
            tasks = ["t0", "t1", "t2", "t3"]
            result = parallel_provider_pairs(tasks, ["kimi", "openrouter", "local"])

        assert result == [
            ("t0", "kimi"),
            ("t1", "openrouter"),
            ("t2", "local"),
            ("t3", "kimi"),
        ]

    def test_fallback_to_all_providers_if_none_have_budget(self):
        balances = {
            "kimi": _make_balance(key="kimi", credits_remaining_usd=0.0),
            "openrouter": _make_balance(key="openrouter", credits_remaining_usd=0.0),
        }
        with patch(
            "mullet.core.providers.budget_probe.get_all_balances",
            return_value=balances,
        ):
            tasks = ["t0", "t1"]
            result = parallel_provider_pairs(tasks, ["kimi", "openrouter"])

        assert result == [("t0", "kimi"), ("t1", "openrouter")]


# ---------------------------------------------------------------------------
# 4. Cache
# ---------------------------------------------------------------------------


class TestBudgetCache:
    def test_get_returns_none_for_missing_key(self):
        cache = _BudgetCache()
        assert cache.get("nonexistent") is None

    def test_set_then_get_returns_balance(self):
        cache = _BudgetCache()
        bal = _make_balance(key="kimi", credits_remaining_usd=10.0)
        cache.set(bal)
        retrieved = cache.get("kimi")
        assert retrieved is not None
        assert retrieved.key == "kimi"
        assert retrieved.credits_remaining_usd == pytest.approx(10.0)

    def test_invalidate_clears_all_when_no_key_given(self):
        cache = _BudgetCache()
        cache.set(_make_balance(key="a"))
        cache.set(_make_balance(key="b"))
        cache.invalidate()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_invalidate_single_key(self):
        cache = _BudgetCache()
        cache.set(_make_balance(key="a"))
        cache.set(_make_balance(key="b"))
        cache.invalidate("a")
        assert cache.get("a") is None
        assert cache.get("b") is not None

    def test_get_returns_none_after_ttl_expires(self):
        cache = _BudgetCache()
        bal = _make_balance(key="kimi")
        # Force a very old timestamp
        cache.set(bal)
        cache._ts["kimi"] = time.time() - 400  # > 300s TTL
        assert cache.get("kimi") is None


# ---------------------------------------------------------------------------
# Module-level cache reset fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_cache():
    """Clear the module-level singleton cache before each test."""
    _cache.invalidate()
    yield
    _cache.invalidate()


# ---------------------------------------------------------------------------
# 5. Gemini tier detection
# ---------------------------------------------------------------------------


class TestGeminiTierDetection:
    def test_default_free(self, monkeypatch):
        from mullet.core.providers.budget_probe import _detect_gemini_tier
        for k in ['GEMINI_TIER','MULLET_GEMINI_TIER','GOOGLE_GENAI_USE_VERTEXAI','GOOGLE_CLOUD_PROJECT','VERTEX_PROJECT_ID']:
            monkeypatch.delenv(k, raising=False)
        assert _detect_gemini_tier() == 'free'

    def test_env_pro(self, monkeypatch):
        from mullet.core.providers.budget_probe import _detect_gemini_tier
        for k in ['MULLET_GEMINI_TIER','GOOGLE_GENAI_USE_VERTEXAI']:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv('GEMINI_TIER', 'pro')
        assert _detect_gemini_tier() == 'pro'

    def test_env_ultra(self, monkeypatch):
        from mullet.core.providers.budget_probe import _detect_gemini_tier
        for k in ['MULLET_GEMINI_TIER','GOOGLE_GENAI_USE_VERTEXAI']:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv('GEMINI_TIER', 'ultra')
        assert _detect_gemini_tier() == 'ultra'

    def test_env_case_insensitive(self, monkeypatch):
        from mullet.core.providers.budget_probe import _detect_gemini_tier
        for k in ['MULLET_GEMINI_TIER','GOOGLE_GENAI_USE_VERTEXAI']:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv('GEMINI_TIER', 'PRO')
        assert _detect_gemini_tier() == 'pro'

    def test_env_unknown_falls_back_to_free(self, monkeypatch):
        from mullet.core.providers.budget_probe import _detect_gemini_tier
        for k in ['MULLET_GEMINI_TIER','GOOGLE_GENAI_USE_VERTEXAI']:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv('GEMINI_TIER', 'garbage')
        assert _detect_gemini_tier() == 'free'

    def test_mullet_alias(self, monkeypatch):
        from mullet.core.providers.budget_probe import _detect_gemini_tier
        for k in ['GEMINI_TIER','GOOGLE_GENAI_USE_VERTEXAI']:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv('MULLET_GEMINI_TIER', 'pro')
        assert _detect_gemini_tier() == 'pro'

    def test_gemini_tier_wins_over_alias(self, monkeypatch):
        from mullet.core.providers.budget_probe import _detect_gemini_tier
        monkeypatch.delenv('GOOGLE_GENAI_USE_VERTEXAI', raising=False)
        monkeypatch.setenv('GEMINI_TIER', 'ultra')
        monkeypatch.setenv('MULLET_GEMINI_TIER', 'pro')
        assert _detect_gemini_tier() == 'ultra'

    def test_vertex_heuristic_pro(self, monkeypatch):
        from mullet.core.providers.budget_probe import _detect_gemini_tier
        for k in ['GEMINI_TIER','MULLET_GEMINI_TIER']:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', '1')
        monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'demo')
        assert _detect_gemini_tier() == 'pro'

    def test_vertex_requires_both_envs(self, monkeypatch):
        from mullet.core.providers.budget_probe import _detect_gemini_tier
        for k in ['GEMINI_TIER','MULLET_GEMINI_TIER','GOOGLE_CLOUD_PROJECT','VERTEX_PROJECT_ID']:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', '1')
        # No GOOGLE_CLOUD_PROJECT → no pro inference
        assert _detect_gemini_tier() == 'free'
