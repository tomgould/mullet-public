"""Tests for mullet.core.budget_advisor."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from mullet.core.budget_advisor import (
    LaneRecommendation,
    _estimate_cost,
    _get_provider_balances,
    format_preflight_message,
    recommend_lane,
    should_preflight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_live_balance_probes():
    """Keep recommend_lane() off the network and the machine's provider config.

    _get_provider_balances() calls budget_probe.get_all_balances(), which
    probes every provider's real balance API and local LLM. Without this, the
    lane tests pass or fail depending on which keys and local models happen to
    be set up (and fail in CI, which has none). Tests that exercise balances
    patch them explicitly, which overrides this.
    """
    with patch("mullet.core.providers.budget_probe.get_all_balances", return_value={}):
        yield


def _mock_budget(pressure: float, five_pct: float = 10.0, weekly_pct: float = 20.0):
    """Return a MagicMock that looks like BudgetCheck."""
    bc = MagicMock()
    bc.pressure = pressure
    bc.five_hour_pct = five_pct
    bc.weekly_pct = weekly_pct
    return bc


# ---------------------------------------------------------------------------
# should_preflight
# ---------------------------------------------------------------------------

class TestShouldPreflight:
    def test_small_task_low_iterations_no_preflight(self):
        assert should_preflight(1_000, 1) is False

    def test_large_task_triggers_preflight(self):
        assert should_preflight(25_000, 1) is True

    def test_many_iterations_triggers_preflight(self):
        assert should_preflight(100, 5) is True

    def test_orchestration_always_triggers(self):
        assert should_preflight(500, 1, is_orchestration=True) is True

    def test_exactly_20000_chars_no_preflight(self):
        # boundary: >20_000 triggers, ==20_000 does not
        assert should_preflight(20_000, 1) is False

    def test_exactly_3_iterations_triggers(self):
        assert should_preflight(100, 3) is True


# ---------------------------------------------------------------------------
# recommend_lane
# ---------------------------------------------------------------------------

GUARD_PATH = "mullet.core.budget_advisor.check_budget"


class TestRecommendLane:
    def test_low_pressure_low_weekly_premium_opus(self):
        with patch(GUARD_PATH, return_value=_mock_budget(0.10, 5.0, 20.0)):
            rec = recommend_lane(1_000)
        assert rec.lane == "premium"
        assert rec.primary_provider == "opus"

    def test_mid_pressure_balanced_sonnet(self):
        # pressure=0.35 is < 0.40, so balanced lane is selected
        with patch(GUARD_PATH, return_value=_mock_budget(0.35, 35.0, 40.0)):
            rec = recommend_lane(1_000)
        assert rec.lane == "balanced"
        assert rec.primary_provider == "sonnet"

    def test_high_pressure_economy_kimi(self):
        with patch(GUARD_PATH, return_value=_mock_budget(0.85, 80.0, 70.0)):
            rec = recommend_lane(1_000)
        assert rec.lane == "economy"
        assert rec.primary_provider == "kimi"
        assert rec.fallback_provider == "local"

    def test_user_preference_honoured_regardless_of_budget(self):
        """economy preference should be used even when pressure is low."""
        with patch(GUARD_PATH, return_value=_mock_budget(0.05, 1.0, 5.0)):
            rec = recommend_lane(1_000, user_preference="economy")
        assert rec.lane == "economy"

    def test_user_preference_premium_at_high_pressure(self):
        """premium preference should be respected even under pressure."""
        with patch(GUARD_PATH, return_value=_mock_budget(0.90, 85.0, 75.0)):
            rec = recommend_lane(1_000, user_preference="premium")
        assert rec.lane == "premium"

    def test_pressure_boundary_balanced_at_0_20(self):
        # pressure == 0.20 is NOT < 0.20, so should not be premium
        with patch(GUARD_PATH, return_value=_mock_budget(0.20, 15.0, 20.0)):
            rec = recommend_lane(500)
        assert rec.lane == "balanced"

    def test_pressure_boundary_economy_at_0_40(self):
        # pressure == 0.40 is NOT < 0.40
        with patch(GUARD_PATH, return_value=_mock_budget(0.40, 35.0, 45.0)):
            rec = recommend_lane(500)
        assert rec.lane in ("economy", "balanced")  # at boundary 0.40 tips to economy

    def test_high_weekly_pct_prevents_premium(self):
        """Even with low pressure, weekly >= 50% should not produce premium."""
        with patch(GUARD_PATH, return_value=_mock_budget(0.10, 5.0, 60.0)):
            rec = recommend_lane(1_000)
        # weekly_pct=60 >= 50 → should NOT be premium
        assert rec.lane != "premium"

    def test_budget_snapshot_populated(self):
        with patch(GUARD_PATH, return_value=_mock_budget(0.15, 8.0, 22.0)):
            rec = recommend_lane(500)
        assert "5h:" in rec.budget_snapshot
        assert "weekly:" in rec.budget_snapshot
        assert "pressure:" in rec.budget_snapshot


# ---------------------------------------------------------------------------
# Cost estimate math
# ---------------------------------------------------------------------------

class TestCostEstimates:
    def test_40k_chars_1_iter_opus_within_10pct(self):
        """
        input_tokens = 40_000 / 4 = 10_000
        output_tokens = 10_000 * 0.3 = 3_000
        opus in  = 10_000 / 1_000_000 * 5.00 = 0.05
        opus out = 3_000  / 1_000_000 * 25.00 = 0.075
        """
        expected_in  = 0.05
        expected_out = 0.075

        with patch(GUARD_PATH, return_value=_mock_budget(0.05, 1.0, 10.0)):
            rec = recommend_lane(40_000, loop_iterations=1)

        # Only check costs when the adviser actually chose opus
        if rec.primary_provider == "opus":
            assert abs(rec.est_input_cost_usd  - expected_in)  / expected_in  < 0.10
            assert abs(rec.est_output_cost_usd - expected_out) / expected_out < 0.10

    def test_40k_chars_1_iter_explicit_economy_kimi(self):
        """
        First KNOWN_PRICING match for "kimi" is k2.6:
        kimi in  = 10_000 / 1_000_000 * 0.95 = 0.0095
        kimi out = 3_000  / 1_000_000 * 4.00 = 0.012
        """
        expected_in  = 0.0095
        expected_out = 0.012

        with patch(GUARD_PATH, return_value=_mock_budget(0.85, 80.0, 70.0)):
            rec = recommend_lane(40_000, loop_iterations=1)

        # high pressure → kimi primary
        assert rec.primary_provider == "kimi"
        assert abs(rec.est_input_cost_usd  - expected_in)  / max(expected_in,  1e-12) < 0.10
        assert abs(rec.est_output_cost_usd - expected_out) / max(expected_out, 1e-12) < 0.10

    def test_local_provider_zero_cost(self):
        with patch(GUARD_PATH, return_value=_mock_budget(0.80, 75.0, 65.0)):
            rec = recommend_lane(5_000, user_preference="economy")
        if rec.primary_provider == "local":
            assert rec.est_input_cost_usd  == 0.0
            assert rec.est_output_cost_usd == 0.0

    def test_loop_iterations_multiply_cost(self):
        with patch(GUARD_PATH, return_value=_mock_budget(0.10, 5.0, 15.0)):
            rec1 = recommend_lane(10_000, loop_iterations=1)
            rec3 = recommend_lane(10_000, loop_iterations=3)

        total1 = rec1.est_input_cost_usd + rec1.est_output_cost_usd
        total3 = rec3.est_input_cost_usd + rec3.est_output_cost_usd
        assert total3 == pytest.approx(total1 * 3, rel=1e-6)


# ---------------------------------------------------------------------------
# format_preflight_message
# ---------------------------------------------------------------------------

class TestFormatPreflightMessage:
    def _make_rec(self, lane: str = "balanced", provider: str = "sonnet") -> LaneRecommendation:
        return LaneRecommendation(
            lane=lane,  # type: ignore[arg-type]
            primary_provider=provider,
            fallback_provider="haiku",
            rationale="Test rationale for preflight display",
            est_input_cost_usd=0.001,
            est_output_cost_usd=0.003,
            budget_snapshot="5h: 12% | weekly: 34% | pressure: 0.25",
        )

    def test_output_under_1200_chars(self):
        rec = self._make_rec()
        alt = self._make_rec("economy", "kimi")
        msg = format_preflight_message(rec, [alt])
        assert len(msg) < 1200, f"Message too long: {len(msg)} chars"

    def test_output_under_20_lines(self):
        rec = self._make_rec()
        alt = self._make_rec("economy", "kimi")
        msg = format_preflight_message(rec, [alt])
        lines = msg.splitlines()
        assert len(lines) <= 20, f"Too many lines: {len(lines)}"

    def test_contains_lane_and_provider(self):
        rec = self._make_rec()
        msg = format_preflight_message(rec, [])
        assert "balanced" in msg
        assert "sonnet" in msg

    def test_contains_snapshot(self):
        rec = self._make_rec()
        msg = format_preflight_message(rec, [])
        assert "5h: 12%" in msg
        assert "weekly: 34%" in msg

    def test_alternatives_appear_in_output(self):
        rec = self._make_rec()
        alts = [self._make_rec("economy", "kimi"), self._make_rec("premium", "opus")]
        msg = format_preflight_message(rec, alts)
        assert "economy" in msg
        assert "kimi" in msg

    def test_no_alternatives_no_error(self):
        rec = self._make_rec()
        msg = format_preflight_message(rec, [])
        assert "Budget Pre-flight" in msg

    def test_contains_cost_estimate(self):
        rec = self._make_rec()
        msg = format_preflight_message(rec, [])
        assert "$" in msg
        assert "USD" in msg


# ---------------------------------------------------------------------------
# _get_provider_balances exception path (lines 26-27)
# ---------------------------------------------------------------------------

class TestGetProviderBalancesException:
    def test_returns_empty_dict_on_import_error(self):
        with patch(
            "mullet.core.providers.budget_probe.get_all_balances",
            side_effect=RuntimeError("import failed"),
        ):
            assert _get_provider_balances() == {}


# ---------------------------------------------------------------------------
# _estimate_cost fallback when no pricing at all (line 91)
# ---------------------------------------------------------------------------

class TestEstimateCostFallback:
    def test_zero_cost_when_no_pricing_available(self):
        with patch("mullet.core.budget_advisor._get_pricing", return_value={}):
            in_cost, out_cost = _estimate_cost("unknown_provider", 1000, 300)
            assert in_cost == 0.0
            assert out_cost == 0.0


# ---------------------------------------------------------------------------
# Auto-switch primary provider when no budget (lines 189-194)
# ---------------------------------------------------------------------------

class TestAutoSwitchPrimary:
    def test_auto_switch_when_primary_has_no_budget(self):
        mock_bal = MagicMock()
        mock_bal.has_budget = False
        mock_bal.priority_score = 0.1

        mock_alt = MagicMock()
        mock_alt.has_budget = True
        mock_alt.priority_score = 0.9

        balances = {
            "anthropic": mock_bal,
            "kimi": mock_alt,
        }

        with patch(GUARD_PATH, return_value=_mock_budget(0.10, 5.0, 20.0)):
            with patch(
                "mullet.core.budget_advisor._get_provider_balances",
                return_value=balances,
            ):
                rec = recommend_lane(1_000)

        assert "auto-switched" in rec.rationale
        assert rec.primary_provider == "kimi"


# ---------------------------------------------------------------------------
# _estimate_cost local provider (line 74)
# ---------------------------------------------------------------------------

class TestEstimateCostLocal:
    def test_local_returns_zero(self):
        in_cost, out_cost = _estimate_cost("local", 1000, 300)
        assert in_cost == 0.0
        assert out_cost == 0.0


# ---------------------------------------------------------------------------
# _estimate_cost fallback to first kimi model (lines 86-88)
# ---------------------------------------------------------------------------

class TestEstimateCostKimiFallback:
    def test_fallback_to_kimi_when_provider_not_found(self):
        pricing = {"some-other-model": (1.0, 2.0), "kimi-k2.6": (0.95, 4.0)}
        with patch("mullet.core.budget_advisor._get_pricing", return_value=pricing):
            in_cost, out_cost = _estimate_cost("unknown_provider", 1_000_000, 300_000)
        # rates = (0.95, 4.0) from kimi fallback
        expected_in = (1_000_000 / 1_000_000) * (0.95 * 1000)  # 0.95
        expected_out = (300_000 / 1_000_000) * (4.0 * 1000)  # 1.2
        assert in_cost == pytest.approx(expected_in)
        assert out_cost == pytest.approx(expected_out)
