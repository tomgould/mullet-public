"""Tests for mullet.core.providers.value_picker (routing-v3 Phase 2)."""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from mullet.core.providers.value_picker import pick_by_value


class TestPickByValue:
    def test_basic_cheapest_wins(self):
        candidates = [
            ("claude-opus-4-6", 1_000_000),
            ("claude-sonnet-4-6", 1_000_000),
            ("claude-haiku-4-5-20251001", 200_000),
        ]
        result = pick_by_value(candidates, "LOW", 4096, "coding")
        assert result == "claude-haiku-4-5-20251001"

    def test_context_filtering_excludes_too_small(self):
        candidates = [
            ("claude-haiku-4-5-20251001", 200_000),
            ("claude-sonnet-4-6", 1_000_000),
        ]
        result = pick_by_value(candidates, "LOW", 500_000, "coding")
        assert result == "claude-sonnet-4-6"

    def test_reasoning_gate_excludes_non_reasoning(self):
        candidates = [
            ("deepseek/deepseek-v4-flash", 1_000_000),
            ("deepseek/deepseek-v4-pro", 1_000_000),
        ]
        result = pick_by_value(
            candidates, "LOW", 4096, "coding", prefer_reasoning=True
        )
        assert result == "deepseek/deepseek-v4-pro"

    def test_quality_floor_excludes_cheap_model(self):
        candidates = [
            ("gemini-2.5-flash-lite", 1_000_000),
            ("claude-opus-4-8", 1_000_000),
        ]
        result = pick_by_value(candidates, "ADVANCED", 4096, "coding")
        assert result == "claude-opus-4-8"

    def test_one_tier_escalation(self, caplog):
        candidates = [("special-model", 1_000_000)]

        def fake_floor(model_id, tier, domain):
            return model_id != "special-model" or tier == "CRITICAL"

        with patch(
            "mullet.core.providers.value_picker.meets_quality_floor",
            side_effect=fake_floor,
        ):
            with caplog.at_level(logging.WARNING):
                result = pick_by_value(candidates, "ADVANCED", 4096, "coding")

        assert result == "special-model"
        assert "escalated one tier to CRITICAL" in caplog.text

    def test_escalation_cap_returns_none(self):
        candidates = [("fails-everything", 1_000_000)]

        with patch(
            "mullet.core.providers.value_picker.meets_quality_floor",
            return_value=False,
        ):
            result = pick_by_value(candidates, "ADVANCED", 4096, "coding")

        assert result is None

    def test_zero_cost_sorts_last(self):
        candidates = [
            ("unpriced-model", 1_000_000),
            ("claude-haiku-4-5-20251001", 200_000),
        ]

        with patch("mullet.core.providers.value_picker._pricing_for") as mock_pricing:

            def side_effect(model_id):
                if model_id == "unpriced-model":
                    return (0.0, 0.0)
                return (0.001, 0.005)

            mock_pricing.side_effect = side_effect
            result = pick_by_value(candidates, "LOW", 4096, "coding")

        assert result == "claude-haiku-4-5-20251001"

    def test_zero_cost_returned_if_only_survivor(self):
        candidates = [("unpriced-model", 1_000_000)]

        with patch(
            "mullet.core.providers.value_picker._pricing_for",
            return_value=(0.0, 0.0),
        ):
            result = pick_by_value(candidates, "LOW", 4096, "coding")

        assert result == "unpriced-model"

    def test_determinism_same_cost(self):
        candidates = [
            ("model-b", 1_000_000),
            ("model-a", 1_000_000),
        ]

        with patch(
            "mullet.core.providers.value_picker._pricing_for",
            return_value=(1.0, 1.0),
        ):
            result1 = pick_by_value(candidates, "LOW", 4096, "coding")
            result2 = pick_by_value(candidates, "LOW", 4096, "coding")

        assert result1 == "model-a"
        assert result2 == "model-a"

    def test_domain_required(self):
        with pytest.raises(TypeError):
            pick_by_value([], "LOW", 4096)
