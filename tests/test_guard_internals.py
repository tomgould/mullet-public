"""Test the internal functions of the Token Guard module."""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch


from mullet.hud.guard import (
    compute_pressure,
    recommend_provider,
    _days_until_reset,
    _paced_pressure,
    pressure_components,
    wait_for_reset,
    BudgetCheck,
    BudgetStatus
)
from mullet.hud.usage import RateLimits


class TestComputePressure:
    """Test the compute_pressure function."""
    
    def test_compute_pressure_low_usage(self):
        """Low usage should result in low pressure."""
        limits = RateLimits(
            five_hour_pct=5,
            weekly_pct=10,
            five_hour_resets=None,
            weekly_resets=None
        )
        pressure = compute_pressure(limits)
        assert isinstance(pressure, float)
        assert pressure < 0.15
    
    def test_compute_pressure_high_session(self):
        """High session usage should result in high pressure."""
        limits = RateLimits(
            five_hour_pct=90,
            weekly_pct=10,
            five_hour_resets=None,
            weekly_resets=None
        )
        pressure = compute_pressure(limits)
        assert isinstance(pressure, float)
        assert pressure > 0.8
    
    def test_compute_pressure_high_weekly(self):
        """High weekly usage should result in high pressure."""
        limits = RateLimits(
            five_hour_pct=5,
            weekly_pct=90,
            five_hour_resets=None,
            weekly_resets=None
        )
        pressure = compute_pressure(limits)
        assert isinstance(pressure, float)
        assert pressure > 0.8
    
    def test_compute_pressure_clamped_to_1(self):
        """Pressure should be clamped to 1.0 maximum."""
        limits = RateLimits(
            five_hour_pct=100,
            weekly_pct=100,
            five_hour_resets=None,
            weekly_resets=None
        )
        pressure = compute_pressure(limits)
        assert pressure == 1.0
    
    def test_compute_pressure_invalid_reset_string(self):
        """Invalid reset strings should not cause exceptions."""
        limits = RateLimits(
            five_hour_pct=50,
            weekly_pct=50,
            five_hour_resets="not-a-date",
            weekly_resets="bad"
        )
        pressure = compute_pressure(limits)
        assert isinstance(pressure, float)
        assert 0.0 <= pressure <= 1.0
    
    def test_compute_pressure_past_reset(self):
        """Past reset times should not cause exceptions."""
        past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        limits = RateLimits(
            five_hour_pct=50,
            weekly_pct=50,
            five_hour_resets=past_time,
            weekly_resets=past_time
        )
        pressure = compute_pressure(limits)
        assert isinstance(pressure, float)
        assert 0.0 <= pressure <= 1.0


class TestRecommendProvider:
    """Test the recommend_provider function (headroom-first ladder)."""

    def test_recommend_provider_low_tier_low_pressure(self):
        """LOW tier always uses local (free) — never burn Anthropic quota on simple tasks."""
        assert recommend_provider(0.1, "LOW") == "local"

    def test_recommend_provider_low_tier_high_pressure(self):
        """LOW tier with high pressure falls through to local."""
        assert recommend_provider(0.9, "LOW") == "local"

    def test_recommend_provider_standard_low_pressure(self):
        """STANDARD low pressure → haiku (Anthropic prepaid substitutes cheap cloud when headroom)."""
        assert recommend_provider(0.1, "STANDARD") == "haiku"

    def test_recommend_provider_standard_mid_pressure(self):
        """STANDARD mid pressure → haiku (still below 60% Anthropic threshold)."""
        assert recommend_provider(0.4, "STANDARD") == "haiku"

    def test_recommend_provider_standard_high_pressure(self):
        """STANDARD high pressure → kimi (>80% pressure → fall back off Anthropic)."""
        assert recommend_provider(0.9, "STANDARD") == "kimi"

    def test_recommend_provider_advanced_low(self):
        """ADVANCED low pressure → sonnet (complex reasoning, headroom available)."""
        assert recommend_provider(0.1, "ADVANCED") == "sonnet"

    def test_recommend_provider_advanced_mid(self):
        """ADVANCED mid pressure → haiku (0.5 is between 0.30 and 0.55)."""
        assert recommend_provider(0.5, "ADVANCED") == "haiku"

    def test_recommend_provider_premium_low(self):
        """PREMIUM low pressure → opus (override mode, Anthropic budget allows)."""
        assert recommend_provider(0.1, "PREMIUM") == "opus"

    def test_recommend_provider_premium_mid(self):
        """PREMIUM mid pressure → sonnet (0.4 is between 0.30 and 0.55)."""
        assert recommend_provider(0.4, "PREMIUM") == "sonnet"

    def test_recommend_provider_premium_high(self):
        """PREMIUM high pressure → haiku (0.7 is between 0.55 and 0.80)."""
        assert recommend_provider(0.7, "PREMIUM") == "haiku"

    def test_recommend_provider_premium_very_high(self):
        """PREMIUM very high pressure → openrouter (0.9 >= 0.80 — auto-downgrade off Anthropic)."""
        assert recommend_provider(0.9, "PREMIUM") == "openrouter"


class TestDaysUntilReset:
    """Test the _days_until_reset function."""
    
    def test_days_until_reset_future(self):
        """Future ISO string should return positive days."""
        future_time = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        result = _days_until_reset(future_time)
        assert isinstance(result, float)
        assert result > 0
    
    def test_days_until_reset_empty(self):
        """Empty string should return 3.5 days."""
        assert _days_until_reset("") == 3.5
    
    def test_days_until_reset_invalid(self):
        """Invalid string should return 3.5 days."""
        assert _days_until_reset("garbage") == 3.5


class TestWaitForReset:
    """Test the wait_for_reset function."""
    
    @patch('time.sleep')
    def test_wait_for_reset_no_reset_time(self, mock_sleep):
        """BudgetCheck with empty reset_time should sleep for 30 minutes."""
        check = BudgetCheck(
            status=BudgetStatus.GO,
            reset_time="",
            message="test"
        )
        wait_for_reset(check)
        mock_sleep.assert_called_once_with(1800)
    
    @patch('time.sleep')
    def test_wait_for_reset_past_time(self, mock_sleep):
        """BudgetCheck with past reset_time should not sleep."""
        past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        check = BudgetCheck(
            status=BudgetStatus.GO,
            reset_time=past_time,
            message="test"
        )
        wait_for_reset(check)
        mock_sleep.assert_not_called()


class TestPacedPressure:
    """Direct unit tests for _paced_pressure."""

    def test_on_pace_returns_pct(self):
        # pace == 1.0 (50% pct, 50% elapsed) → pct * pace = 0.5 * 1.0 = 0.5
        result = _paced_pressure(0.5, 0.5)
        assert abs(result - 0.5) < 1e-9

    def test_under_pace_drops_below_raw(self):
        # 10% used in 50% of window → under-spending, pressure below raw 0.10
        result = _paced_pressure(0.10, 0.50)
        assert result < 0.10

    def test_over_pace_exceeds_raw(self):
        # 50% used in 10% of window → over-spending, pressure > 0.50
        result = _paced_pressure(0.50, 0.10)
        assert result > 0.50

    def test_clamped_to_one(self):
        # Extreme over-spending should be capped at 1.0
        result = _paced_pressure(0.99, 0.01)
        assert result <= 1.0

    def test_near_zero_pct_stays_near_zero(self):
        result = _paced_pressure(0.0, 0.5)
        assert result == 0.0

    def test_elapsed_zero_defaults_to_0_01(self):
        # elapsed_fraction=0 is replaced by 0.01 internally
        result = _paced_pressure(0.5, 0.0)
        assert result >= 0.0  # must not raise or return negative


class TestPressureComponents:
    def test_returns_tuple_of_two_floats(self):
        limits = RateLimits(five_hour_pct=30, weekly_pct=20)
        sp, wp = pressure_components(limits)
        assert isinstance(sp, float) and isinstance(wp, float)

    def test_without_resets_uses_raw_pct(self):
        limits = RateLimits(five_hour_pct=40, weekly_pct=60,
                            five_hour_resets=None, weekly_resets=None)
        sp, wp = pressure_components(limits)
        # Without resets info, pressure == raw pct fraction
        assert abs(sp - 0.40) < 1e-9
        assert abs(wp - 0.60) < 1e-9

    def test_invalid_resets_iso_falls_back_to_raw(self):
        limits = RateLimits(five_hour_pct=50, weekly_pct=50,
                            five_hour_resets="not-a-date", weekly_resets="bad")
        sp, wp = pressure_components(limits)
        assert 0.0 <= sp <= 1.0 and 0.0 <= wp <= 1.0


class TestWaitForResetMissingLines:
    """Cover lines 209-215 in guard.py."""

    @patch('time.sleep')
    @patch('builtins.print')
    def test_future_reset_time_sleeps_with_buffer(self, mock_print, mock_sleep):
        """Future reset_time should sleep until reset + 60s buffer."""
        future_time = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        check = BudgetCheck(
            status=BudgetStatus.GO,
            reset_time=future_time,
            message="test"
        )
        wait_for_reset(check)
        mock_sleep.assert_called_once()
        args = mock_sleep.call_args[0][0]
        assert 300 <= args <= 370  # 5 minutes + 60s buffer
        printed = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
        assert "Sleeping" in printed

    @patch('time.sleep')
    @patch('builtins.print')
    def test_invalid_reset_time_string_sleeps_default(self, mock_print, mock_sleep):
        """Invalid reset_time format should fall back to 30-minute sleep."""
        check = BudgetCheck(
            status=BudgetStatus.GO,
            reset_time="not-a-valid-iso-string",
            message="test"
        )
        wait_for_reset(check)
        mock_sleep.assert_called_once_with(1800)
        printed = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
        assert "Could not parse reset time" in printed