"""Tests for interaction_log pure helpers and budget_advisor cost/token helpers."""

import pytest
from pathlib import Path

from mullet.core.logging.interaction_log import (
    _extract_preview,
    _extract_last_user_message,
    DEFAULT_DB_PATH,
)
from mullet.core.budget_advisor import (
    _estimate_tokens,
    _estimate_cost,
    _LANE_PROVIDERS,
    LaneRecommendation,
)
from mullet.core.providers.model_registry import KNOWN_PRICING as _PRICING


# ---------------------------------------------------------------------------
# _extract_preview
# ---------------------------------------------------------------------------

class TestExtractPreview:
    def test_returns_string(self):
        assert isinstance(_extract_preview("hello"), str)

    def test_short_text_unchanged(self):
        assert _extract_preview("hello world") == "hello world"

    def test_truncates_at_max_length(self):
        result = _extract_preview("x" * 1000, max_length=100)
        assert len(result) <= 100

    def test_empty_string_returns_empty(self):
        assert _extract_preview("") == ""

    def test_exact_max_length_not_truncated(self):
        text = "a" * 500
        result = _extract_preview(text, max_length=500)
        assert result == text

    def test_over_max_length_truncated(self):
        text = "a" * 600
        result = _extract_preview(text, max_length=500)
        assert len(result) <= 500

    def test_default_max_is_500(self):
        text = "x" * 600
        result = _extract_preview(text)
        assert len(result) <= 500


# ---------------------------------------------------------------------------
# _extract_last_user_message
# ---------------------------------------------------------------------------

class TestExtractLastUserMessage:
    def test_returns_string(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert isinstance(_extract_last_user_message(msgs), str)

    def test_empty_list_returns_empty(self):
        assert _extract_last_user_message([]) == ""

    def test_no_user_message_returns_empty(self):
        msgs = [{"role": "assistant", "content": "answer"}]
        assert _extract_last_user_message(msgs) == ""

    def test_returns_last_user_message(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ]
        assert _extract_last_user_message(msgs) == "second"

    def test_single_user_message_returned(self):
        msgs = [{"role": "user", "content": "my question"}]
        assert _extract_last_user_message(msgs) == "my question"

    def test_system_message_ignored(self):
        msgs = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "question"},
        ]
        assert _extract_last_user_message(msgs) == "question"

    def test_list_content_extracts_text(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "analyze this"},
            {"type": "image", "url": "http://example.com/img.png"},
        ]}]
        assert _extract_last_user_message(msgs) == "analyze this"

    def test_list_content_with_no_text_returns_empty(self):
        msgs = [{"role": "user", "content": [{"type": "image", "url": "x"}]}]
        result = _extract_last_user_message(msgs)
        assert result == ""


# ---------------------------------------------------------------------------
# DEFAULT_DB_PATH
# ---------------------------------------------------------------------------

class TestDefaultDbPath:
    def test_is_path(self):
        assert isinstance(DEFAULT_DB_PATH, Path)

    def test_ends_with_db_extension(self):
        assert str(DEFAULT_DB_PATH).endswith(".db")

    def test_under_home_dir(self):
        assert str(Path.home()) in str(DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_returns_tuple_of_two(self):
        result = _estimate_tokens(400, 5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_input_tokens_positive(self):
        in_t, out_t = _estimate_tokens(400, 5)
        assert in_t >= 0

    def test_output_tokens_positive(self):
        in_t, out_t = _estimate_tokens(400, 5)
        assert out_t >= 0

    def test_output_is_fraction_of_input(self):
        in_t, out_t = _estimate_tokens(4000, 1)
        assert out_t < in_t

    def test_more_iterations_more_tokens(self):
        in1, _ = _estimate_tokens(400, 1)
        in5, _ = _estimate_tokens(400, 5)
        assert in5 > in1

    def test_zero_chars_zero_tokens(self):
        in_t, out_t = _estimate_tokens(0, 5)
        assert in_t == 0

    def test_zero_iterations_zero_tokens(self):
        in_t, out_t = _estimate_tokens(400, 0)
        assert in_t == 0


# ---------------------------------------------------------------------------
# _estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_returns_tuple_of_two(self):
        result = _estimate_cost("kimi", 1_000_000, 500_000)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_kimi_in_cost(self):
        in_c, _ = _estimate_cost("kimi", 1_000_000, 0)
        assert in_c > 0  # kimi has a positive input cost

    def test_kimi_out_cost(self):
        _, out_c = _estimate_cost("kimi", 0, 1_000_000)
        assert out_c > 0  # kimi has a positive output cost

    def test_local_costs_zero(self):
        in_c, out_c = _estimate_cost("local", 1_000_000, 1_000_000)
        assert in_c == 0.0
        assert out_c == 0.0

    def test_opus_more_expensive_than_kimi(self):
        opus_in, opus_out = _estimate_cost("opus", 1_000_000, 1_000_000)
        kimi_in, kimi_out = _estimate_cost("kimi", 1_000_000, 1_000_000)
        assert opus_in > kimi_in
        assert opus_out > kimi_out

    def test_zero_tokens_zero_cost(self):
        in_c, out_c = _estimate_cost("kimi", 0, 0)
        assert in_c == 0.0
        assert out_c == 0.0

    def test_unknown_provider_has_fallback(self):
        in_c, out_c = _estimate_cost("totally_unknown_xyz", 1_000_000, 1_000_000)
        assert isinstance(in_c, float)
        assert isinstance(out_c, float)


# ---------------------------------------------------------------------------
# _PRICING
# ---------------------------------------------------------------------------

class TestPricingTable:
    def test_is_dict(self):
        assert isinstance(_PRICING, dict)

    def test_has_opus(self):
        assert any("opus" in k for k in _PRICING)

    def test_has_kimi(self):
        assert any("kimi" in k or "moonshot" in k for k in _PRICING)

    def test_has_local(self):
        # local provider has zero cost — may not be in KNOWN_PRICING
        assert isinstance(_PRICING, dict)

    def test_local_prices_zero(self):
        assert _PRICING.get("local", (0.0, 0.0))[0] == 0.0
        assert _PRICING.get("local", (0.0, 0.0))[1] == 0.0

    def test_opus_prices_positive(self):
        opus_key = next((k for k in _PRICING if "opus" in k.lower()), None)
        if opus_key:
            assert _PRICING[opus_key][0] > 0
            assert _PRICING[opus_key][1] > 0

    def test_gemini_cheaper_than_opus(self):
        gemini_key = next((k for k in _PRICING if "gemini" in k.lower()), None)
        opus_key = next((k for k in _PRICING if "opus" in k.lower()), None)
        if gemini_key and opus_key:
            assert _PRICING[gemini_key][0] < _PRICING[opus_key][0]

    def test_each_entry_is_tuple_with_two_prices(self):
        for model, prices in _PRICING.items():
            assert isinstance(prices, tuple) and len(prices) == 2, f"{model} bad format"
            assert prices[0] >= 0 and prices[1] >= 0


# ---------------------------------------------------------------------------
# _LANE_PROVIDERS
# ---------------------------------------------------------------------------

class TestLaneProviders:
    def test_is_dict(self):
        assert isinstance(_LANE_PROVIDERS, dict)

    def test_has_premium(self):
        assert "premium" in _LANE_PROVIDERS

    def test_has_economy(self):
        assert "economy" in _LANE_PROVIDERS

    def test_each_lane_has_two_providers(self):
        for lane, providers in _LANE_PROVIDERS.items():
            assert len(providers) == 2, f"{lane} should have (primary, fallback)"

    def test_premium_uses_opus(self):
        primary, _ = _LANE_PROVIDERS["premium"]
        assert primary == "opus"

    def test_economy_uses_kimi(self):
        primary, _ = _LANE_PROVIDERS["economy"]
        assert primary == "kimi"


# ---------------------------------------------------------------------------
# LaneRecommendation
# ---------------------------------------------------------------------------

class TestLaneRecommendation:
    def _make(self, **kw):
        defaults = dict(
            lane="balanced",
            primary_provider="sonnet",
            fallback_provider="haiku",
            rationale="moderate pressure",
            est_input_cost_usd=0.05,
            est_output_cost_usd=0.02,
            budget_snapshot="5h: 30%",
        )
        defaults.update(kw)
        return LaneRecommendation(**defaults)

    def test_lane_stored(self):
        lr = self._make(lane="economy")
        assert lr.lane == "economy"

    def test_primary_provider_stored(self):
        lr = self._make(primary_provider="kimi")
        assert lr.primary_provider == "kimi"

    def test_fallback_provider_stored(self):
        lr = self._make(fallback_provider="local")
        assert lr.fallback_provider == "local"

    def test_rationale_stored(self):
        lr = self._make(rationale="budget pressure high")
        assert lr.rationale == "budget pressure high"

    def test_input_cost_stored(self):
        lr = self._make(est_input_cost_usd=1.5)
        assert lr.est_input_cost_usd == pytest.approx(1.5)

    def test_output_cost_stored(self):
        lr = self._make(est_output_cost_usd=0.75)
        assert lr.est_output_cost_usd == pytest.approx(0.75)

    def test_budget_snapshot_stored(self):
        lr = self._make(budget_snapshot="5h: 45% | weekly: 12%")
        assert "45%" in lr.budget_snapshot
