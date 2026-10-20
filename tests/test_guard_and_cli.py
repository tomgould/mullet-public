"""Tests for Token Guard and Claude CLI provider."""
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

import pytest

from mullet.hud.guard import check_budget, BudgetStatus, _paced_pressure, compute_pressure, recommend_provider
from mullet.hud.usage import RateLimits
from mullet.core.providers.claude_cli import ClaudeCLIProvider, is_claude_cli_available


@patch('mullet.hud.guard.fetch_usage')
def test_check_budget_GO_when_usage_low(mock_fetch):
    mock_fetch.return_value = RateLimits(
        five_hour_pct=10.0, weekly_pct=20.0,
        five_hour_resets=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        weekly_resets=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
    )
    result = check_budget()
    assert result.status == BudgetStatus.GO


@patch('mullet.hud.guard.fetch_usage')
def test_check_budget_THROTTLE_when_5h_over_65(mock_fetch):
    mock_fetch.return_value = RateLimits(
        five_hour_pct=70.0, weekly_pct=20.0,
        five_hour_resets=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        weekly_resets=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
    )
    result = check_budget()
    assert result.status == BudgetStatus.THROTTLE


@patch('mullet.hud.guard.fetch_usage')
def test_check_budget_THROTTLE_when_5h_over_85(mock_fetch):
    """High 5h usage alone → THROTTLE, not PAUSE (weekly is low)."""
    mock_fetch.return_value = RateLimits(
        five_hour_pct=90.0, weekly_pct=20.0,
        five_hour_resets=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        weekly_resets=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
    )
    result = check_budget()
    assert result.status == BudgetStatus.THROTTLE


@patch('mullet.hud.guard.fetch_usage')
def test_check_budget_THROTTLE_when_weekly_over_90(mock_fetch):
    """High weekly usage alone → THROTTLE with local provider (5h is low, no PAUSE)."""
    mock_fetch.return_value = RateLimits(
        five_hour_pct=10.0, weekly_pct=95.0,
        five_hour_resets="", weekly_resets="",
    )
    result = check_budget()
    assert result.status == BudgetStatus.THROTTLE
    assert result.recommended_provider == "kimi"


@patch('mullet.hud.guard.fetch_usage')
def test_check_budget_GO_when_api_unavailable(mock_fetch):
    mock_fetch.return_value = None
    result = check_budget()
    assert result.status == BudgetStatus.GO


def test_messages_to_prompt():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": [{"type": "text", "text": "More."}]},
    ]
    result = ClaudeCLIProvider._messages_to_prompt(messages)
    assert "System: You are helpful." in result
    assert "Hello." in result
    assert "Previous response: Hi." in result
    assert "More." in result


@patch('subprocess.run')
def test_is_claude_cli_available_true(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    assert is_claude_cli_available() is True


@patch('subprocess.run', side_effect=FileNotFoundError)
def test_is_claude_cli_available_false(mock_run):
    assert is_claude_cli_available() is False


class TestPacedPressure:
    def test_on_linear_track(self):
        assert _paced_pressure(0.5, 0.5) == 0.5

    def test_banking_headroom(self):
        assert _paced_pressure(0.2, 0.5) == pytest.approx(0.08)

    def test_overspending(self):
        assert _paced_pressure(0.5, 0.25) == 1.0

    def test_gentle_overspend(self):
        assert _paced_pressure(0.3, 0.2) == pytest.approx(0.6)

    def test_zero_elapsed_clamped(self):
        assert _paced_pressure(0.5, 0.0) == 1.0


class TestComputePressure:
    def test_returns_float_between_zero_and_one(self):
        five_h_reset = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        weekly_reset = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        limits = RateLimits(
            five_hour_pct=50.0,
            weekly_pct=30.0,
            five_hour_resets=five_h_reset,
            weekly_resets=weekly_reset,
        )
        result = compute_pressure(limits)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_zero_usage_returns_low_pressure(self):
        five_h_reset = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        weekly_reset = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        limits = RateLimits(
            five_hour_pct=0.0,
            weekly_pct=0.0,
            five_hour_resets=five_h_reset,
            weekly_resets=weekly_reset,
        )
        result = compute_pressure(limits)
        assert result <= 0.5

    def test_high_five_hour_usage_returns_high_pressure(self):
        five_h_reset = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        weekly_reset = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        limits = RateLimits(
            five_hour_pct=90.0,
            weekly_pct=20.0,
            five_hour_resets=five_h_reset,
            weekly_resets=weekly_reset,
        )
        result = compute_pressure(limits)
        assert result >= 0.5


class TestRecommendProvider:
    @pytest.mark.parametrize("pressure,tier,expected", [
        (0.10, "STANDARD", "haiku"),
        (0.50, "STANDARD", "haiku"),
        (0.60, "STANDARD", "openrouter"),
        (0.79, "STANDARD", "openrouter"),
        (0.90, "STANDARD", "kimi"),
        (0.20, "CRITICAL", "opus"),
        (0.60, "CRITICAL", "sonnet"),
        (0.85, "CRITICAL", "openrouter"),
        (0.20, "ADVANCED", "sonnet"),
        (0.50, "ADVANCED", "haiku"),
        (0.70, "ADVANCED", "openrouter"),
        (0.90, "ADVANCED", "kimi"),
        (0.10, "LOW", "local"),
        (0.50, "LOW", "local"),
        (0.90, "LOW", "local"),
        (0.10, "PREMIUM", "opus"),
        (0.40, "PREMIUM", "sonnet"),
        (0.60, "PREMIUM", "haiku"),
        (0.85, "PREMIUM", "openrouter"),
    ])
    def test_recommend_provider(self, pressure, tier, expected):
        assert recommend_provider(pressure, tier) == expected


@patch('mullet.hud.guard.fetch_usage')
def test_check_budget_PAUSE_when_both_critical(mock_fetch):
    """PAUSE fires when 5h ≥95% AND weekly ≥90% simultaneously (runaway safeguard)."""
    from mullet.hud.guard import BudgetStatus
    mock_fetch.return_value = RateLimits(
        five_hour_pct=96.0, weekly_pct=92.0,
        five_hour_resets=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        weekly_resets=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
    )
    result = check_budget()
    assert result.status == BudgetStatus.PAUSE
    assert "PAUSE" in result.message


@patch('mullet.hud.guard.fetch_usage')
def test_check_budget_only_5h_critical_is_throttle_not_pause(mock_fetch):
    """5h ≥95% but weekly <90% → THROTTLE not PAUSE."""
    mock_fetch.return_value = RateLimits(
        five_hour_pct=97.0, weekly_pct=50.0,
        five_hour_resets=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        weekly_resets=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
    )
    result = check_budget()
    assert result.status == BudgetStatus.THROTTLE


@patch('mullet.hud.guard.fetch_usage')
def test_check_budget_only_weekly_critical_is_throttle_not_pause(mock_fetch):
    """weekly ≥90% but 5h <95% → THROTTLE not PAUSE."""
    mock_fetch.return_value = RateLimits(
        five_hour_pct=60.0, weekly_pct=93.0,
        five_hour_resets=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        weekly_resets=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
    )
    result = check_budget()
    assert result.status == BudgetStatus.THROTTLE
