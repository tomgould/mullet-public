"""Coverage-gap tests for mullet/core/providers/budget_probe.py.

Targets lines not hit by tests/test_budget_probe.py.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mullet.core.providers.budget_probe import (
    ProviderBalance,
    _cache,
    get_all_balances,
    probe_all,
    probe_anthropic,
    probe_gemini,
    probe_kimi,
    probe_openrouter,
    sort_providers_by_budget,
)


# ---------------------------------------------------------------------------
# ProviderBalance.has_budget — line 52  ("not found" branch)
# ---------------------------------------------------------------------------

def test_has_budget_false_when_error_contains_not_found():
    bal = ProviderBalance(
        name="X", key="x",
        credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
        error="Key not found in env",
    )
    assert bal.has_budget is False


# ---------------------------------------------------------------------------
# ProviderBalance.priority_score — line 94  (credits_remaining_usd <= 0)
# ---------------------------------------------------------------------------

def test_priority_score_zero_when_credits_remaining_zero():
    bal = ProviderBalance(
        name="X", key="x",
        credits_remaining_usd=0.0, credits_used_usd=None, credits_total_usd=None,
        billing_model="api",
    )
    assert bal.priority_score == pytest.approx(0.0)


def test_priority_score_zero_when_credits_remaining_negative():
    bal = ProviderBalance(
        name="X", key="x",
        credits_remaining_usd=-1.0, credits_used_usd=None, credits_total_usd=None,
        billing_model="api",
    )
    assert bal.priority_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _load_dotenv ImportError — lines 176-177
# ---------------------------------------------------------------------------

def test_load_dotenv_import_error():
    """When python-dotenv is missing, _load_dotenv must swallow ImportError."""
    real_import = __builtins__["__import__"]

    def fake_import(name, *args, **kwargs):
        if name == "dotenv" or name.startswith("dotenv."):
            raise ImportError("No module named 'dotenv'")
        return real_import(name, *args, **kwargs)

    # Temporarily remove dotenv from sys.modules so the import is attempted
    dotenv_modules = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "dotenv" or k.startswith("dotenv.")}
    try:
        with patch("builtins.__import__", fake_import):
            from mullet.core.providers.budget_probe import _load_dotenv
            # Should not raise even though dotenv is unavailable
            _load_dotenv()
    finally:
        sys.modules.update(dotenv_modules)


# ---------------------------------------------------------------------------
# probe_kimi — lines 281-289 (unknown balance shape)
# ---------------------------------------------------------------------------

def test_probe_kimi_unknown_balance_shape():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": {"something_else": 123}}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(probe_kimi(api_key="sk-test"))

    assert result.error is None
    assert result.credits_remaining_usd is None
    assert "raw_response" in result.extra


# ---------------------------------------------------------------------------
# probe_anthropic — lines 303-304, 319-320, 331-332, 355-357
# ---------------------------------------------------------------------------

def test_probe_anthropic_billing_detection_import_error():
    """Lines 303-304: importing ClaudeCodeProvider raises → default billing."""
    fake_result = MagicMock()
    fake_result.pressure = 0.1
    fake_result.five_hour_pct = 5.0
    fake_result.weekly_pct = 5.0
    fake_result.status.value = "go"
    fake_result.reset_time = ""

    with patch(
        "mullet.hud.guard.check_budget", return_value=fake_result,
    ), patch(
        "mullet.hud.usage.fetch_usage", return_value=None,
    ):
        result = asyncio.run(probe_anthropic())
    assert result.billing_model == "subscription_cli"


def test_probe_anthropic_reset_time_format_exception():
    """Lines 319-320: datetime formatting exception → raw reset string truncated."""
    fake_result = MagicMock()
    fake_result.pressure = 0.1
    fake_result.five_hour_pct = 5.0
    fake_result.weekly_pct = 5.0
    fake_result.status.value = "go"
    fake_result.reset_time = "not-a-valid-iso"

    with patch(
        "mullet.hud.guard.check_budget", return_value=fake_result,
    ), patch(
        "mullet.hud.usage.fetch_usage", return_value=None,
    ):
        result = asyncio.run(probe_anthropic())
    assert result.window_label.startswith("5h:")
    assert "not-a-valid-iso"[:16] in result.window_label


def test_probe_anthropic_fetch_usage_exception():
    """Lines 331-332: fetch_usage raises → tier defaults to ×1."""
    fake_result = MagicMock()
    fake_result.pressure = 0.1
    fake_result.five_hour_pct = 5.0
    fake_result.weekly_pct = 5.0
    fake_result.status.value = "go"
    fake_result.reset_time = ""

    with patch(
        "mullet.hud.guard.check_budget", return_value=fake_result,
    ), patch(
        "mullet.hud.usage.fetch_usage", side_effect=RuntimeError("boom"),
    ):
        result = asyncio.run(probe_anthropic())
    assert "×1" in result.name


def test_probe_anthropic_main_body_exception():
    """Lines 355-357: check_budget raises → error balance with default tier."""
    with patch(
        "mullet.hud.guard.check_budget", side_effect=RuntimeError("guard boom"),
    ):
        result = asyncio.run(probe_anthropic())
    assert result.error is not None
    assert "guard boom" in result.error
    assert "×5" in result.name


# ---------------------------------------------------------------------------
# probe_gemini — lines 406-407, 422, 424
# ---------------------------------------------------------------------------

def test_probe_gemini_import_exception():
    """Lines 406-407: is_gemini_cli_available import fails → billing='api'."""
    with patch(
        "mullet.core.providers.gemini_cli.is_gemini_cli_available",
        side_effect=ImportError("no module"),
    ), patch.dict("os.environ", {"GOOGLE_API_KEY": "key"}, clear=False):
        result = asyncio.run(probe_gemini())
    assert result.billing_model == "api"


def test_probe_gemini_subscription_cli_window():
    """Line 422: billing == subscription_cli → subscription window string."""
    with patch(
        "mullet.core.providers.gemini_cli.is_gemini_cli_available", return_value=True,
    ), patch.dict("os.environ", {"GOOGLE_API_KEY": "key"}, clear=False):
        result = asyncio.run(probe_gemini())
    assert result.billing_model == "subscription_cli"
    assert "resets midnight Pacific" in result.window_label


def test_probe_gemini_no_key_and_not_subscription():
    """Line 424: no key + not subscription_cli → error balance."""
    with patch(
        "mullet.core.providers.gemini_cli.is_gemini_cli_available", return_value=False,
    ), patch.dict("os.environ", {}, clear=True):
        result = asyncio.run(probe_gemini())
    assert result.billing_model == "api"
    assert result.error is not None
    assert "not set" in result.error


# ---------------------------------------------------------------------------
# probe_all — lines 514-520 (exception from gather)
# ---------------------------------------------------------------------------

def test_probe_all_exception_from_gather():
    """When a coroutine raises an exception, probe_all wraps it in a ProviderBalance."""
    with patch(
        "mullet.core.providers.budget_probe.probe_openrouter",
        side_effect=RuntimeError("openrouter boom"),
    ):
        result = asyncio.run(probe_all(include=["openrouter"], force_refresh=True))
    assert "openrouter" in result
    assert result["openrouter"].error is not None
    assert "openrouter boom" in result["openrouter"].error


# ---------------------------------------------------------------------------
# get_all_balances — lines 529-536 (running event loop)
# ---------------------------------------------------------------------------

def test_get_all_balances_when_loop_is_running():
    """When an event loop is already running, return cached values only."""
    _cache.invalidate()
    bal = ProviderBalance(
        name="Local", key="local",
        credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
        is_free=True, billing_model="free",
    )
    _cache.set(bal)

    fake_loop = MagicMock()
    fake_loop.is_running.return_value = True
    with patch("asyncio.get_event_loop", return_value=fake_loop):
        result = get_all_balances(force_refresh=False)
    assert "local" in result
    assert result["local"].key == "local"
    _cache.invalidate()


# ---------------------------------------------------------------------------
# sort_providers_by_budget — lines 552-559
# ---------------------------------------------------------------------------

def test_sort_providers_by_budget():
    """Providers with higher priority scores should come first."""
    balances = {
        "kimi": ProviderBalance(
            name="Kimi", key="kimi",
            credits_remaining_usd=10.0, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
        ),
        "openrouter": ProviderBalance(
            name="OR", key="openrouter",
            credits_remaining_usd=0.0, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
        ),
        "local": ProviderBalance(
            name="Local", key="local",
            credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
            is_free=True, billing_model="free",
        ),
    }
    with patch(
        "mullet.core.providers.budget_probe.get_all_balances",
        return_value=balances,
    ):
        result = sort_providers_by_budget(["openrouter", "kimi", "local"])
    # openrouter has 0 budget → should be last
    assert result[-1] == "openrouter"
    # local has 0.5 score, kimi has ~0.75 → kimi first, then local
    assert result == ["kimi", "local", "openrouter"]


def test_sort_providers_by_budget_unknown_middle():
    """Providers not in balances get 0.5 default score → middle."""
    balances = {
        "kimi": ProviderBalance(
            name="Kimi", key="kimi",
            credits_remaining_usd=0.0, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
        ),
    }
    with patch(
        "mullet.core.providers.budget_probe.get_all_balances",
        return_value=balances,
    ):
        result = sort_providers_by_budget(["kimi", "unknown", "other"])
    # kimi has 0 budget, unknown/other get 0.5
    assert result[0] == "unknown"
    assert result[1] == "other"
    assert result[2] == "kimi"


# ---------------------------------------------------------------------------
# ProviderBalance.summary_line — line 115 (subscription with window_label)
# ---------------------------------------------------------------------------

def test_summary_line_subscription_with_window_label():
    bal = ProviderBalance(
        name="Anthropic", key="anthropic",
        credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
        is_subscription=True, billing_model="subscription_cli",
        window_label="5h: 10% | weekly: 20%",
    )
    line = bal.summary_line()
    assert "5h: 10% | weekly: 20%" in line


# ---------------------------------------------------------------------------
# probe_openrouter exception — lines 230-232
# ---------------------------------------------------------------------------

def test_probe_openrouter_http_exception():
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("connection timeout")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(probe_openrouter(api_key="sk-test"))

    assert result.error is not None
    assert "connection timeout" in result.error


# ---------------------------------------------------------------------------
# probe_kimi exception — lines 287-289
# ---------------------------------------------------------------------------

def test_probe_kimi_http_exception():
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("connection timeout")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(probe_kimi(api_key="sk-test"))

    assert result.error is not None
    assert "connection timeout" in result.error


# ---------------------------------------------------------------------------
# probe_anthropic — fix lines 303-304, add line 318
# ---------------------------------------------------------------------------

def test_probe_anthropic_billing_detection_import_error():
    """Lines 303-304: default billing when check_budget works."""
    fake_result = MagicMock()
    fake_result.pressure = 0.1
    fake_result.five_hour_pct = 5.0
    fake_result.weekly_pct = 5.0
    fake_result.status.value = "go"
    fake_result.reset_time = ""

    with patch(
        "mullet.hud.guard.check_budget", return_value=fake_result,
    ), patch(
        "mullet.hud.usage.fetch_usage", return_value=None,
    ):
        result = asyncio.run(probe_anthropic())
    assert result.billing_model == "subscription_cli"


def test_probe_anthropic_valid_reset_time():
    """Line 318: valid ISO reset time is formatted as HH:MM UTC."""
    fake_result = MagicMock()
    fake_result.pressure = 0.1
    fake_result.five_hour_pct = 5.0
    fake_result.weekly_pct = 5.0
    fake_result.status.value = "go"
    fake_result.reset_time = "2026-06-18T14:30:00+00:00"

    with patch(
        "mullet.hud.guard.check_budget", return_value=fake_result,
    ), patch(
        "mullet.hud.usage.fetch_usage", return_value=None,
    ):
        result = asyncio.run(probe_anthropic())
    assert "resets 14:30 UTC" in result.window_label


# ---------------------------------------------------------------------------
# probe_all cache and coroutine building — lines 484-487, 491, 498-505, 510-511
# ---------------------------------------------------------------------------

def test_probe_all_uses_cache():
    """Lines 484-487: cached providers are returned without probing."""
    _cache.invalidate()
    cached_bal = ProviderBalance(
        name="OpenRouter", key="openrouter",
        credits_remaining_usd=10.0, credits_used_usd=None, credits_total_usd=None,
        billing_model="api",
    )
    _cache.set(cached_bal)

    with patch("mullet.core.providers.budget_probe.probe_openrouter") as mock_probe:
        result = asyncio.run(probe_all(include=["openrouter"], force_refresh=False))
    assert result["openrouter"].credits_remaining_usd == pytest.approx(10.0)
    assert not mock_probe.called
    _cache.invalidate()


def test_probe_all_all_cached_returns_early():
    """Line 491: when every provider is cached, probe_all returns early."""
    _cache.invalidate()
    for key in ["openrouter", "kimi", "anthropic", "gemini", "local"]:
        _cache.set(ProviderBalance(
            name=key.title(), key=key,
            credits_remaining_usd=1.0, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
        ))

    with patch("mullet.core.providers.budget_probe.probe_openrouter") as mock_probe:
        result = asyncio.run(probe_all(force_refresh=False))
    assert len(result) == 5
    assert not mock_probe.called
    _cache.invalidate()


def test_probe_all_builds_coros_for_all_providers():
    """Lines 498-505: probe_all builds coroutines for all provider types."""
    _cache.invalidate()

    async def fake_probe():
        return ProviderBalance(
            name="Fake", key="fake",
            credits_remaining_usd=1.0, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
        )

    with patch("mullet.core.providers.budget_probe.probe_openrouter", side_effect=fake_probe) as m_or, \
         patch("mullet.core.providers.budget_probe.probe_kimi", side_effect=fake_probe) as m_kimi, \
         patch("mullet.core.providers.budget_probe.probe_anthropic", side_effect=fake_probe) as m_anthro, \
         patch("mullet.core.providers.budget_probe.probe_gemini", side_effect=fake_probe) as m_gemini, \
         patch("mullet.core.providers.budget_probe.probe_local", side_effect=fake_probe) as m_local:
        result = asyncio.run(probe_all(force_refresh=True))

    assert m_or.called
    assert m_kimi.called
    assert m_anthro.called
    assert m_gemini.called
    assert m_local.called
    assert len(result) == 5
    _cache.invalidate()


def test_probe_all_success_caches_result():
    """Lines 510-511: successful ProviderBalance results are cached."""
    _cache.invalidate()

    async def fake_probe():
        return ProviderBalance(
            name="OpenRouter", key="openrouter",
            credits_remaining_usd=5.0, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
        )

    with patch("mullet.core.providers.budget_probe.probe_openrouter", side_effect=fake_probe):
        result = asyncio.run(probe_all(include=["openrouter"], force_refresh=True))

    assert result["openrouter"].credits_remaining_usd == pytest.approx(5.0)
    cached = _cache.get("openrouter")
    assert cached is not None
    assert cached.credits_remaining_usd == pytest.approx(5.0)
    _cache.invalidate()


# ---------------------------------------------------------------------------
# get_all_balances RuntimeError — lines 537-539
# ---------------------------------------------------------------------------

def test_get_all_balances_runtime_error_from_get_event_loop():
    """Lines 537-539: asyncio.get_event_loop raises RuntimeError → run probe_all."""
    with patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")):
        with patch("mullet.core.providers.budget_probe.probe_all") as mock_probe_all:
            mock_probe_all.return_value = {}
            get_all_balances(force_refresh=True)
    assert mock_probe_all.called


def test_provider_balance_api_depleted_credits_returns_zero():
    """Line 94: billing_model='api' with credits_remaining_usd=0.0 → 0.0 (depleted)."""
    bal = ProviderBalance(
        name="Test", key="test",
        credits_remaining_usd=0.0, credits_used_usd=10.0, credits_total_usd=10.0,
        billing_model="api",
    )
    assert bal.priority_score == pytest.approx(0.0)
