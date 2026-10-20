"""Unit tests for mullet.harness.fallback.hook."""

from __future__ import annotations

import time

import pytest

from mullet.harness.fallback.hook import (
    default_fallback_chain_resolver,
    make_fallback_event_hook,
)
from mullet.harness.fallback.state import FallbackStateRegistry
from mullet.harness.hooks.types import EventPayload

pytestmark = pytest.mark.harness


def _make_payload(
    *,
    event: str = "session.error",
    error: str | None = "Rate limit exceeded",
    session_id: str = "sess-1",
    model: str = "kimi-k2.6",
) -> EventPayload:
    return EventPayload(
        event=event,
        session_id=session_id,
        error=error,
        payload={"model": model},
    )


def _static_chain(*models: str):
    def _resolver(_session_id: str, current_model: str) -> list[str]:
        return [m for m in models if m != current_model]

    return _resolver


class TestFallbackEventHook:
    def test_hook_ignores_non_error_events(self):
        registry = FallbackStateRegistry()
        hook = make_fallback_event_hook(
            registry, _static_chain("kimi-k2.6", "gemini-2.5-flash")
        )
        result = hook(
            _make_payload(event="session.message_complete", error=None)
        )
        assert result is None
        assert len(registry) == 0

    def test_hook_ignores_non_retryable_errors(self):
        registry = FallbackStateRegistry()
        hook = make_fallback_event_hook(
            registry, _static_chain("kimi-k2.6", "gemini-2.5-flash")
        )
        # Auth error → not retryable.
        result = hook(_make_payload(error="Invalid API key supplied"))
        assert result is None
        # Plain unknown error → not retryable.
        result = hook(_make_payload(error="Something completely unrelated"))
        assert result is None
        assert len(registry) == 0

    def test_hook_marks_current_model_failed_on_retryable(self):
        registry = FallbackStateRegistry()
        hook = make_fallback_event_hook(
            registry, _static_chain("kimi-k2.6", "gemini-2.5-flash")
        )
        hook(_make_payload(error="Rate limit exceeded"))
        state = registry.get("sess-1")
        assert state is not None
        assert state.is_in_cooldown("kimi-k2.6") is True
        assert state.last_error == "Rate limit exceeded"
        assert state.last_error_classification == "rate_limit"

    def test_hook_returns_payload_with_fallback_model_set(self):
        registry = FallbackStateRegistry()
        hook = make_fallback_event_hook(
            registry, _static_chain("kimi-k2.6", "gemini-2.5-flash")
        )
        result = hook(_make_payload(error="429 too many requests"))
        assert isinstance(result, EventPayload)
        assert result.payload["fallback_model"] == "gemini-2.5-flash"
        assert result.payload["fallback_attempt"] == 1
        # Registry advanced.
        state = registry.get("sess-1")
        assert state is not None
        assert state.current_model == "gemini-2.5-flash"
        assert state.fallback_index == 1

    def test_hook_returns_none_when_no_fallback_available(self):
        registry = FallbackStateRegistry()
        # Resolver yields a one-item chain that matches current_model →
        # filtered out, leaving an empty list.
        hook = make_fallback_event_hook(
            registry, _static_chain("kimi-k2.6")
        )
        result = hook(_make_payload(error="Rate limit exceeded"))
        assert result is None
        # State still records the failure for observability.
        state = registry.get("sess-1")
        assert state is not None
        assert state.is_in_cooldown("kimi-k2.6") is True

    def test_hook_uses_retry_signal_for_cooldown(self):
        registry = FallbackStateRegistry()
        hook = make_fallback_event_hook(
            registry, _static_chain("kimi-k2.6", "gemini-2.5-flash")
        )
        # Provider says "retry in 10 minutes" — must clamp to >= 600s.
        before = time.time()
        hook(
            _make_payload(
                error="Rate limit exceeded, retry in 10 minutes"
            )
        )
        state = registry.get("sess-1")
        assert state is not None
        cooldown_until = state.failed_models["kimi-k2.6"]
        assert cooldown_until - before >= 600 - 1  # allow tiny clock skew
        assert cooldown_until - before < 600 + 5

    def test_hook_uses_default_60s_cooldown_when_no_signal(self):
        registry = FallbackStateRegistry()
        hook = make_fallback_event_hook(
            registry, _static_chain("kimi-k2.6", "gemini-2.5-flash")
        )
        before = time.time()
        hook(_make_payload(error="Rate limit exceeded"))
        state = registry.get("sess-1")
        assert state is not None
        cooldown_until = state.failed_models["kimi-k2.6"]
        assert 59 <= cooldown_until - before <= 61


class TestDefaultFallbackChainResolver:
    def test_skips_current_model(self):
        chain = default_fallback_chain_resolver("sess", "kimi-k2.6")
        assert "kimi-k2.6" not in chain
        assert "kimi-k2.5" in chain
        assert "gemini-2.5-flash" in chain

    def test_returns_full_chain_when_current_not_in_list(self):
        chain = default_fallback_chain_resolver("sess", "unknown-model")
        assert chain[0] == "kimi-k2.6"
        assert "claude-haiku-4-5" in chain
