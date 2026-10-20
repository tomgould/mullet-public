"""Tests for mullet.core.budget_advisor pure helpers."""


from mullet.core.budget_advisor import (
    _estimate_tokens,
    _estimate_cost,
    should_preflight,
    LaneRecommendation,
    format_preflight_message,
    _LANE_PROVIDERS,
)
from mullet.core.providers.model_registry import KNOWN_PRICING as _PRICING


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_basic_estimation(self):
        input_t, output_t = _estimate_tokens(4000, 1)
        assert input_t == 1000
        assert output_t == 300

    def test_scales_with_iterations(self):
        input_t, output_t = _estimate_tokens(4000, 5)
        assert input_t == 5000
        assert output_t == 1500

    def test_output_is_30_percent_of_input(self):
        input_t, output_t = _estimate_tokens(8000, 2)
        assert output_t == int(input_t * 0.3)

    def test_zero_chars_returns_zeros(self):
        input_t, output_t = _estimate_tokens(0, 10)
        assert input_t == 0
        assert output_t == 0

    def test_returns_tuple_of_ints(self):
        result = _estimate_tokens(4000, 1)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_local_is_free(self):
        in_cost, out_cost = _estimate_cost("local", 1_000_000, 1_000_000)
        assert in_cost == 0.0
        assert out_cost == 0.0

    def test_kimi_pricing(self):
        in_cost, out_cost = _estimate_cost("kimi", 1_000_000, 300_000)
        assert in_cost > 0
        assert out_cost > 0

    def test_unknown_provider_falls_back_to_kimi(self):
        in_c, out_c = _estimate_cost("unknown_provider", 1_000_000, 300_000)
        kimi_in, kimi_out = _estimate_cost("kimi", 1_000_000, 300_000)
        assert in_c == kimi_in
        assert out_c == kimi_out

    def test_returns_tuple_of_floats(self):
        result = _estimate_cost("kimi", 1000, 300)
        assert isinstance(result, tuple)
        assert all(isinstance(v, float) for v in result)

    def test_opus_more_expensive_than_kimi(self):
        opus_in, _ = _estimate_cost("opus", 1_000_000, 0)
        kimi_in, _ = _estimate_cost("kimi", 1_000_000, 0)
        assert opus_in > kimi_in


# ---------------------------------------------------------------------------
# should_preflight
# ---------------------------------------------------------------------------

class TestShouldPreflight:
    def test_small_single_iteration_false(self):
        assert should_preflight(5000, 1) is False

    def test_large_task_triggers_preflight(self):
        assert should_preflight(25_000, 1) is True

    def test_exactly_20000_is_false(self):
        assert should_preflight(20_000, 1) is False

    def test_20001_is_true(self):
        assert should_preflight(20_001, 1) is True

    def test_three_iterations_triggers(self):
        assert should_preflight(100, 3) is True

    def test_two_iterations_does_not_trigger(self):
        assert should_preflight(100, 2) is False

    def test_orchestration_always_triggers(self):
        assert should_preflight(0, 0, is_orchestration=True) is True

    def test_returns_bool(self):
        assert isinstance(should_preflight(100, 1), bool)


# ---------------------------------------------------------------------------
# LaneRecommendation dataclass
# ---------------------------------------------------------------------------

class TestLaneRecommendation:
    def test_fields_stored(self):
        rec = LaneRecommendation(
            lane="economy",
            primary_provider="kimi",
            fallback_provider="local",
            rationale="saving tokens",
            est_input_cost_usd=0.001,
            est_output_cost_usd=0.0005,
            budget_snapshot="5h: 10% | weekly: 30% | pressure: 0.10",
        )
        assert rec.lane == "economy"
        assert rec.primary_provider == "kimi"
        assert rec.fallback_provider == "local"
        assert "saving" in rec.rationale


# ---------------------------------------------------------------------------
# _PRICING and _LANE_PROVIDERS constants
# ---------------------------------------------------------------------------

class TestPricingTable:
    def test_has_expected_providers(self):
        keys = " ".join(_PRICING.keys()).lower()
        assert "opus" in keys
        assert "sonnet" in keys
        assert "haiku" in keys

    def test_local_is_free(self):
        assert _PRICING.get("local", (0.0, 0.0))[0] == 0.0

    def test_opus_more_expensive_than_sonnet(self):
        opus_key = next((k for k in _PRICING if "opus" in k.lower()), None)
        sonnet_key = next((k for k in _PRICING if "sonnet" in k.lower()), None)
        if opus_key and sonnet_key:
            assert _PRICING[opus_key][0] > _PRICING[sonnet_key][0]


class TestLaneProviders:
    def test_has_expected_lanes(self):
        for lane in ("premium", "max", "balanced", "fast", "economy"):
            assert lane in _LANE_PROVIDERS

    def test_each_lane_has_primary_and_fallback(self):
        for lane, (primary, fallback) in _LANE_PROVIDERS.items():
            assert isinstance(primary, str)
            assert isinstance(fallback, str)
            assert primary != ""


# ---------------------------------------------------------------------------
# format_preflight_message
# ---------------------------------------------------------------------------

class TestFormatPreflightMessage:
    def _make_rec(self, lane="economy", provider="kimi") -> LaneRecommendation:
        return LaneRecommendation(
            lane=lane,
            primary_provider=provider,
            fallback_provider="local",
            rationale="test rationale",
            est_input_cost_usd=0.001,
            est_output_cost_usd=0.0005,
            budget_snapshot="5h: 10% | weekly: 30% | pressure: 0.10",
        )

    def test_returns_string(self):
        rec = self._make_rec()
        assert isinstance(format_preflight_message(rec, []), str)

    def test_contains_lane(self):
        rec = self._make_rec(lane="economy")
        result = format_preflight_message(rec, [])
        assert "economy" in result

    def test_contains_provider(self):
        rec = self._make_rec(provider="kimi")
        result = format_preflight_message(rec, [])
        assert "kimi" in result

    def test_contains_budget_snapshot(self):
        rec = self._make_rec()
        result = format_preflight_message(rec, [])
        assert "pressure" in result

    def test_no_alternatives_still_works(self):
        rec = self._make_rec()
        result = format_preflight_message(rec, [])
        assert len(result) > 0

    def test_with_alternatives_included(self):
        rec = self._make_rec()
        alt = self._make_rec(lane="fast", provider="haiku")
        result = format_preflight_message(rec, [alt])
        assert "fast" in result or "haiku" in result or "Alternatives" in result
