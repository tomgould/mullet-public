"""Unit tests for mullet.harness.fallback.retry_signals."""

from __future__ import annotations

from datetime import timedelta

import pytest

from mullet.harness.fallback.retry_signals import extract_auto_retry_signal

pytestmark = pytest.mark.harness


class TestExtractAutoRetrySignal:
    def test_seconds(self):
        assert extract_auto_retry_signal("retry in 30 seconds") == timedelta(
            seconds=30
        )
        assert extract_auto_retry_signal("retrying in 5 sec") == timedelta(
            seconds=5
        )
        assert extract_auto_retry_signal("retry after 1 s") == timedelta(
            seconds=1
        )

    def test_minutes(self):
        assert extract_auto_retry_signal("retry in 2 minutes") == timedelta(
            minutes=2
        )
        assert extract_auto_retry_signal("retry after 5 min") == timedelta(
            minutes=5
        )
        assert extract_auto_retry_signal("retrying in 1 m") == timedelta(
            minutes=1
        )

    def test_hours(self):
        assert extract_auto_retry_signal("retry in 3 hours") == timedelta(
            hours=3
        )
        assert extract_auto_retry_signal("retry after 1 hr") == timedelta(
            hours=1
        )
        assert extract_auto_retry_signal("retrying in 2 h") == timedelta(
            hours=2
        )

    def test_days(self):
        assert extract_auto_retry_signal("retry in 4 days") == timedelta(
            days=4
        )
        assert extract_auto_retry_signal("retrying in 1 d") == timedelta(
            days=1
        )

    def test_weeks(self):
        assert extract_auto_retry_signal("retry in 2 weeks") == timedelta(
            weeks=2
        )
        assert extract_auto_retry_signal("retrying in 1 w") == timedelta(
            weeks=1
        )

    def test_with_tilde_prefix(self):
        assert extract_auto_retry_signal(
            "retrying in ~2 weeks"
        ) == timedelta(weeks=2)
        assert extract_auto_retry_signal(
            "retry in ~30 seconds"
        ) == timedelta(seconds=30)
        assert extract_auto_retry_signal(
            "retry after ~5 minutes"
        ) == timedelta(minutes=5)

    def test_returns_none_for_no_match(self):
        assert extract_auto_retry_signal("no signal here") is None
        assert extract_auto_retry_signal("") is None
        assert extract_auto_retry_signal("retry sometime soon") is None

    def test_case_insensitive(self):
        assert extract_auto_retry_signal(
            "RETRY IN 30 SECONDS"
        ) == timedelta(seconds=30)
        assert extract_auto_retry_signal(
            "Retry After 2 Minutes"
        ) == timedelta(minutes=2)
