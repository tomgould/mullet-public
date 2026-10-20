"""Unit tests for mullet.harness.fallback.state."""

from __future__ import annotations

import time

import pytest

from mullet.harness.fallback.state import (
    FallbackState,
    FallbackStateRegistry,
)

pytestmark = pytest.mark.harness


class TestFallbackState:
    def test_state_initialised(self):
        state = FallbackState(
            session_id="sess-1",
            original_model="kimi-k2.6",
            current_model="kimi-k2.6",
        )
        assert state.session_id == "sess-1"
        assert state.original_model == "kimi-k2.6"
        assert state.current_model == "kimi-k2.6"
        assert state.fallback_index == 0
        assert state.attempt_count == 0
        assert state.failed_models == {}
        assert state.last_error is None
        assert state.last_error_classification is None

    def test_mark_failed_sets_cooldown(self):
        state = FallbackState(
            session_id="s",
            original_model="m",
            current_model="m",
        )
        before = time.time()
        state.mark_failed("m", cooldown_seconds=60)
        after = time.time()
        assert "m" in state.failed_models
        # Cooldown horizon must lie in [now+60, now+60] approx.
        assert before + 60 <= state.failed_models["m"] <= after + 60 + 0.5

    def test_is_in_cooldown_returns_true_during_window(self):
        state = FallbackState(
            session_id="s",
            original_model="m",
            current_model="m",
        )
        state.mark_failed("m", cooldown_seconds=120)
        assert state.is_in_cooldown("m") is True
        # An unmarked model is not in cooldown.
        assert state.is_in_cooldown("other") is False

    def test_is_in_cooldown_returns_false_after_window(self, monkeypatch):
        state = FallbackState(
            session_id="s",
            original_model="m",
            current_model="m",
        )
        # Pin "now" so mark_failed sets a known horizon.
        monkeypatch.setattr(
            "mullet.harness.fallback.state.time.time", lambda: 1_000.0
        )
        state.mark_failed("m", cooldown_seconds=60)
        # Travel past the cooldown horizon.
        monkeypatch.setattr(
            "mullet.harness.fallback.state.time.time", lambda: 1_100.0
        )
        assert state.is_in_cooldown("m") is False

    def test_clear_cooldown_drops_record(self):
        state = FallbackState(
            session_id="s",
            original_model="m",
            current_model="m",
        )
        state.mark_failed("m")
        state.clear_cooldown("m")
        assert state.is_in_cooldown("m") is False
        assert "m" not in state.failed_models
        # Idempotent on absent keys.
        state.clear_cooldown("missing")

    def test_find_next_available_skips_cooldown_models(self):
        state = FallbackState(
            session_id="s",
            original_model="a",
            current_model="a",
        )
        state.mark_failed("a")
        state.mark_failed("b")
        chain = ["a", "b", "c", "d"]
        assert state.find_next_available(chain) == "c"

    def test_find_next_available_returns_none_when_all_in_cooldown(self):
        state = FallbackState(
            session_id="s",
            original_model="a",
            current_model="a",
        )
        chain = ["a", "b"]
        for m in chain:
            state.mark_failed(m, cooldown_seconds=300)
        assert state.find_next_available(chain) is None

    def test_advance_sets_current_increments_index(self):
        state = FallbackState(
            session_id="s",
            original_model="a",
            current_model="a",
        )
        state.advance("b")
        assert state.current_model == "b"
        assert state.fallback_index == 1
        assert state.attempt_count == 1

        state.advance("c")
        assert state.current_model == "c"
        assert state.fallback_index == 2
        assert state.attempt_count == 2


class TestFallbackStateRegistry:
    def test_state_registry_get_or_create(self):
        reg = FallbackStateRegistry()
        s1 = reg.get_or_create("sess-1", original_model="kimi-k2.6")
        assert s1.session_id == "sess-1"
        assert s1.original_model == "kimi-k2.6"
        assert s1.current_model == "kimi-k2.6"
        # Idempotent: second call returns same instance.
        s2 = reg.get_or_create("sess-1", original_model="other")
        assert s2 is s1
        # original_model isn't overwritten.
        assert s2.original_model == "kimi-k2.6"

    def test_state_registry_isolates_sessions(self):
        reg = FallbackStateRegistry()
        a = reg.get_or_create("sess-a", original_model="kimi-k2.6")
        b = reg.get_or_create("sess-b", original_model="gemini-2.5-flash")
        a.mark_failed("kimi-k2.6")
        assert b.failed_models == {}
        assert "kimi-k2.6" in a.failed_models
        # __contains__ + len.
        assert "sess-a" in reg
        assert "sess-b" in reg
        assert len(reg) == 2

    def test_state_registry_pop_and_clear(self):
        reg = FallbackStateRegistry()
        reg.get_or_create("sess-1", original_model="m")
        reg.get_or_create("sess-2", original_model="m")
        popped = reg.pop("sess-1")
        assert popped is not None and popped.session_id == "sess-1"
        assert "sess-1" not in reg
        reg.clear()
        assert len(reg) == 0
