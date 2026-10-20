"""Tests for mullet.harness.coordinator.concurrency."""

from __future__ import annotations

import pytest

from mullet.harness.coordinator.concurrency import ConcurrencyLimits, ConcurrencyManager

pytestmark = pytest.mark.harness


class TestConcurrencyManager:
    def test_acquire_release(self):
        mgr = ConcurrencyManager()
        assert mgr.acquire("kimi") is True
        assert mgr.usage()["total"] == 1
        mgr.release("kimi")
        assert mgr.usage()["total"] == 0

    def test_provider_limit_enforced(self):
        limits = ConcurrencyLimits(
            default_concurrency=2,
            max_total_agents=10,
        )
        mgr = ConcurrencyManager(limits)
        assert mgr.acquire("kimi") is True
        assert mgr.acquire("kimi") is True
        assert mgr.acquire("kimi") is False  # at limit
        mgr.release("kimi")
        assert mgr.acquire("kimi") is True

    def test_model_limit_overrides_provider(self):
        limits = ConcurrencyLimits(
            default_concurrency=5,
            max_total_agents=10,
            model_concurrency={"anthropic/opus-4-7": 1},
        )
        mgr = ConcurrencyManager(limits)
        assert mgr.acquire("anthropic", "opus-4-7") is True
        assert mgr.acquire("anthropic", "opus-4-7") is False  # model limit
        # Different model on same provider should still work (provider limit=5)
        assert mgr.acquire("anthropic", "sonnet") is True

    def test_release_below_zero_safe(self):
        mgr = ConcurrencyManager()
        # Release without prior acquire should not go negative
        mgr.release("kimi")
        usage = mgr.usage()
        assert usage["total"] == 0
        assert usage["provider"].get("kimi", 0) == 0

    def test_max_total_agents_enforced(self):
        limits = ConcurrencyLimits(max_total_agents=2)
        mgr = ConcurrencyManager(limits)
        assert mgr.acquire("kimi") is True
        assert mgr.acquire("gemini") is True
        assert mgr.acquire("local") is False  # total limit

    def test_usage_returns_snapshot(self):
        limits = ConcurrencyLimits()
        mgr = ConcurrencyManager(limits)
        mgr.acquire("kimi", "k2")
        mgr.acquire("anthropic", "sonnet")
        usage = mgr.usage()
        assert usage["total"] == 2
        assert usage["provider"]["kimi"] == 1
        assert usage["provider"]["anthropic"] == 1
        assert usage["model"]["kimi/k2"] == 1
        assert usage["model"]["anthropic/sonnet"] == 1
