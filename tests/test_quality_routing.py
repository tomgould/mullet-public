"""Tests for quality-feedback routing — providers with poor success rates are skipped.

Red phase: all tests should fail until decision.py wires in routing_history.success_rate().
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mullet.core.router.decision import RoutingDecision, Router, ProviderEntry
from mullet.core.router.complexity import Tier


def _make_router(providers: dict[str, ProviderEntry] | None = None) -> Router:
    """Build a minimal router with local + kimi + openrouter."""
    default_providers = {
        "local": ProviderEntry(name="local", tier="free", models={"default": "local-model"}, healthy=True),
        "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek-v3"}, healthy=True),
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.7-code"}, healthy=True),
    }
    return Router(
        providers=providers or default_providers,
        tier_map={
            "LOW": ["local"],
            "STANDARD": ["openrouter", "kimi", "local"],
            "ADVANCED": ["kimi", "openrouter"],
            "CRITICAL": ["kimi"],
        },
    )


def _write_history(path: Path, records: list[dict]) -> None:
    """Write routing history JSONL to path."""
    from datetime import datetime, timezone
    with open(path, "w") as f:
        for rec in records:
            rec.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            rec.setdefault("task_preview", "test task")
            rec.setdefault("task_signature", "aabbccdd11223344")
            rec.setdefault("model", "some-model")
            rec.setdefault("tier", "STANDARD")
            rec.setdefault("tokens_in", 100)
            rec.setdefault("tokens_out", 50)
            rec.setdefault("wall_time_ms", 1000)
            rec.setdefault("cost_usd", 0.001)
            rec.setdefault("error_class", None)
            rec.setdefault("tags", [])
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQualityFeedbackRouting:
    """Quality-feedback routing — providers below success threshold are skipped."""

    def test_poor_provider_skipped_for_next_candidate(self, tmp_path):
        """openrouter has 1/5 success rate — should be skipped in favour of kimi."""
        history_path = tmp_path / "routing_history.jsonl"
        records = [
            {"provider": "openrouter", "success": False},
            {"provider": "openrouter", "success": False},
            {"provider": "openrouter", "success": False},
            {"provider": "openrouter", "success": False},
            {"provider": "openrouter", "success": True},  # 1/5 = 20%
        ]
        _write_history(history_path, records)

        router = _make_router()
        with patch("mullet.core.router.decision._quality_history_path", return_value=history_path):
            result = router.route("ok", force_tier=Tier.STANDARD)

        assert result.provider != "openrouter", (
            f"Expected openrouter to be skipped (20% success), got: {result.provider}"
        )
        assert "quality" in " ".join(result.reasons).lower(), (
            f"Expected quality skip reason, got: {result.reasons}"
        )

    def test_insufficient_data_does_not_penalise(self, tmp_path):
        """Provider with only 2 records is not penalised (below min_samples=3)."""
        history_path = tmp_path / "routing_history.jsonl"
        records = [
            {"provider": "openrouter", "success": False},
            {"provider": "openrouter", "success": False},
        ]
        _write_history(history_path, records)

        router = _make_router()
        with patch("mullet.core.router.decision._quality_history_path", return_value=history_path):
            result = router.route("ok", force_tier=Tier.STANDARD)

        # With 2 records (below min_samples=3), provider is not penalised
        # openrouter is first in STANDARD tier_map so it should be picked
        assert result.provider == "openrouter", (
            f"Expected openrouter (insufficient data — not penalised), got: {result.provider}"
        )

    def test_good_provider_not_skipped(self, tmp_path):
        """Provider with 5/5 success rate is NOT skipped."""
        history_path = tmp_path / "routing_history.jsonl"
        records = [{"provider": "openrouter", "success": True} for _ in range(5)]
        _write_history(history_path, records)

        router = _make_router()
        with patch("mullet.core.router.decision._quality_history_path", return_value=history_path):
            result = router.route("ok", force_tier=Tier.STANDARD)

        assert result.provider == "openrouter", (
            f"Expected openrouter (100% success rate — should not be skipped), got: {result.provider}"
        )

    def test_fallback_when_all_providers_poor(self, tmp_path):
        """When ALL candidates have poor success rates, pick the least-bad one (don't fail)."""
        history_path = tmp_path / "routing_history.jsonl"
        # Both openrouter and kimi have terrible rates; local has no history
        records = (
            [{"provider": "openrouter", "success": False}] * 4 +
            [{"provider": "openrouter", "success": True}] +
            [{"provider": "kimi", "success": False}] * 4 +
            [{"provider": "kimi", "success": True}]
        )
        _write_history(history_path, records)

        router = _make_router()
        with patch("mullet.core.router.decision._quality_history_path", return_value=history_path):
            result = router.route("ok", force_tier=Tier.STANDARD)

        # Must still return something — never a dead end
        assert result.provider not in ("none", None), (
            f"Expected a fallback provider, got: {result.provider}"
        )

    def test_absent_history_file_routes_normally(self):
        """When routing_history.jsonl doesn't exist, routing proceeds without quality filter."""
        absent = Path("/tmp/__nonexistent_routing_history_test__.jsonl")
        assert not absent.exists()

        router = _make_router()
        with patch("mullet.core.router.decision._quality_history_path", return_value=absent):
            result = router.route("ok", force_tier=Tier.STANDARD)

        assert result.provider in ("openrouter", "kimi", "local"), (
            f"Expected a valid provider when history absent, got: {result.provider}"
        )
        assert result.provider != "none"

    def test_quality_reason_appended(self, tmp_path):
        """When a provider is skipped for quality, the reason is appended to reasons list."""
        history_path = tmp_path / "routing_history.jsonl"
        records = [{"provider": "openrouter", "success": False}] * 5
        _write_history(history_path, records)

        router = _make_router()
        with patch("mullet.core.router.decision._quality_history_path", return_value=history_path):
            result = router.route("ok", force_tier=Tier.STANDARD)

        reasons_str = " ".join(result.reasons).lower()
        assert "quality" in reasons_str or "success" in reasons_str, (
            f"Expected quality skip in reasons, got: {result.reasons}"
        )


class TestUserRatingRouting:
    """User thumbs-up/down ratings from QualityTracker deprioritize bad providers."""

    def _write_user_quality(self, path: Path, records: dict) -> None:
        """Write provider_quality.json in QualityTracker format."""
        import json
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def test_low_user_rating_skipped(self, tmp_path):
        """Provider with <35% user approval is skipped in favour of next candidate."""
        quality_path = tmp_path / "provider_quality.json"
        self._write_user_quality(quality_path, {
            "openrouter": {"up": 1, "down": 4, "total_tokens": 0, "response_count": 5},  # 20%
        })
        absent_history = tmp_path / "routing_history.jsonl"
        router = _make_router()
        with patch("mullet.core.router.decision._quality_tracker_path", return_value=quality_path), \
             patch("mullet.core.router.decision._quality_history_path", return_value=absent_history):
            result = router.route("ok", force_tier=Tier.STANDARD)
        assert result.provider != "openrouter", (
            f"Expected openrouter to be skipped (20% user approval), got: {result.provider}"
        )
        reasons_str = " ".join(result.reasons).lower()
        assert "user-rating" in reasons_str or "approval" in reasons_str

    def test_high_user_rating_not_skipped(self, tmp_path):
        """Provider with >35% user approval is NOT skipped."""
        quality_path = tmp_path / "provider_quality.json"
        self._write_user_quality(quality_path, {
            "openrouter": {"up": 4, "down": 1, "total_tokens": 0, "response_count": 5},  # 80%
        })
        absent_history = tmp_path / "routing_history.jsonl"
        router = _make_router()
        with patch("mullet.core.router.decision._quality_tracker_path", return_value=quality_path), \
             patch("mullet.core.router.decision._quality_history_path", return_value=absent_history):
            result = router.route("ok", force_tier=Tier.STANDARD)
        assert result.provider == "openrouter"

    def test_insufficient_ratings_not_penalised(self, tmp_path):
        """Provider with < min_samples ratings is not penalised."""
        quality_path = tmp_path / "provider_quality.json"
        self._write_user_quality(quality_path, {
            "openrouter": {"up": 0, "down": 2, "total_tokens": 0, "response_count": 2},  # 0% but only 2 ratings
        })
        absent_history = tmp_path / "routing_history.jsonl"
        router = _make_router()
        with patch("mullet.core.router.decision._quality_tracker_path", return_value=quality_path), \
             patch("mullet.core.router.decision._quality_history_path", return_value=absent_history):
            result = router.route("ok", force_tier=Tier.STANDARD)
        assert result.provider == "openrouter"

    def test_absent_user_quality_file_routes_normally(self):
        """When provider_quality.json doesn't exist, routing is unaffected."""
        absent = Path("/tmp/__nonexistent_user_quality_test__.json")
        assert not absent.exists()
        router = _make_router()
        with patch("mullet.core.router.decision._quality_tracker_path", return_value=absent):
            result = router.route("ok", force_tier=Tier.STANDARD)
        assert result.provider in ("openrouter", "kimi", "local")

    def test_user_rating_and_history_both_skip(self, tmp_path):
        """Both signals can independently skip a provider (union, not intersection)."""
        # openrouter: bad user rating; kimi: bad history success rate
        quality_path = tmp_path / "provider_quality.json"
        history_path = tmp_path / "routing_history.jsonl"
        self._write_user_quality(quality_path, {
            "openrouter": {"up": 0, "down": 5, "total_tokens": 0, "response_count": 5},  # 0% user
        })
        records = [{"provider": "kimi", "success": False}] * 4 + [{"provider": "kimi", "success": True}]
        _write_history(history_path, records)
        router = _make_router()
        with (
            patch("mullet.core.router.decision._quality_tracker_path", return_value=quality_path),
            patch("mullet.core.router.decision._quality_history_path", return_value=history_path),
        ):
            result = router.route("ok", force_tier=Tier.STANDARD)
        # Both openrouter (user) and kimi (history) are skipped -> falls back to local
        assert result.provider in ("local", "openrouter", "kimi"), (
            f"Expected a fallback provider, got: {result.provider}"
        )
        # At minimum a provider is returned (no dead end)
        assert result.provider not in ("none", None)


class TestRateLimitRouting:
    """Rate-limit down-weighting — providers with high recent 429 rates are skipped."""

    def test_high_rate_limit_rate_skips_provider(self, tmp_path):
        """openrouter has 10/15 rate-limit errors in last 20 calls → skipped."""
        history_path = tmp_path / "routing_history.jsonl"
        records = (
            [{"provider": "openrouter", "success": False, "error_class": "rate_limit"}] * 10
            + [{"provider": "openrouter", "success": True, "error_class": None}] * 5
        )
        _write_history(history_path, records)

        router = _make_router()
        with patch("mullet.core.router.decision._quality_history_path", return_value=history_path):
            result = router.route("ok", force_tier=Tier.STANDARD)

        # openrouter should be skipped — either by the rate-limit or quality check
        assert result.provider != "openrouter", (
            f"Expected openrouter to be skipped (67% rate-limit), got: {result.provider}"
        )

    def test_low_rate_limit_rate_does_not_skip(self, tmp_path):
        """openrouter has only 1/20 rate-limit errors → not skipped."""
        history_path = tmp_path / "routing_history.jsonl"
        records = (
            [{"provider": "openrouter", "success": False, "error_class": "rate_limit"}] * 1
            + [{"provider": "openrouter", "success": True, "error_class": None}] * 19
        )
        _write_history(history_path, records)

        router = _make_router()
        with patch("mullet.core.router.decision._quality_history_path", return_value=history_path):
            result = router.route("ok", force_tier=Tier.STANDARD)

        assert result.provider == "openrouter"

    def test_insufficient_samples_not_penalised(self, tmp_path):
        """Provider with only 2 recent records (below min_samples for both quality and rate-limit) is not skipped."""
        history_path = tmp_path / "routing_history.jsonl"
        records = [
            {"provider": "openrouter", "success": False, "error_class": "rate_limit"},
            {"provider": "openrouter", "success": False, "error_class": "rate_limit"},
        ]
        _write_history(history_path, records)

        router = _make_router()
        with patch("mullet.core.router.decision._quality_history_path", return_value=history_path):
            result = router.route("ok", force_tier=Tier.STANDARD)

        assert result.provider == "openrouter"

    def test_unknown_provider_entries_ignored(self, tmp_path):
        """Entries with provider='unknown' are ignored in rate-limit tracking."""
        from mullet.core.router.decision import _load_recent_rate_limit_rates
        history_path = tmp_path / "routing_history.jsonl"
        records = [
            {"provider": "unknown", "success": False, "error_class": "rate_limit"},
        ] * 20
        _write_history(history_path, records)

        rates = _load_recent_rate_limit_rates(history_path)
        assert "unknown" not in rates

    def test_absent_file_returns_empty(self):
        """When the history file doesn't exist, returns empty dict (fail-open)."""
        from mullet.core.router.decision import _load_recent_rate_limit_rates
        absent = Path("/tmp/__nonexistent_rl_test__.jsonl")
        assert not absent.exists()
        rates = _load_recent_rate_limit_rates(absent)
        assert rates == {}

    def test_window_only_counts_last_n_calls(self, tmp_path):
        """Window=5 means only last 5 calls matter, not the full history."""
        from mullet.core.router.decision import _load_recent_rate_limit_rates
        history_path = tmp_path / "routing_history.jsonl"
        # First 20 calls all rate-limited, last 5 all successful
        records = (
            [{"provider": "openrouter", "success": False, "error_class": "rate_limit"}] * 20
            + [{"provider": "openrouter", "success": True, "error_class": None}] * 5
        )
        _write_history(history_path, records)

        rates = _load_recent_rate_limit_rates(history_path, window=5, min_samples=3)
        assert rates.get("openrouter", 1.0) == 0.0, (
            "With window=5 and last 5 all successful, rate should be 0.0"
        )
