"""Integration tests for fallback wiring in HarnessEngine."""

from __future__ import annotations

from collections import deque

import pytest

from mullet.core.providers.base import LLMResponse, Provider
from mullet.harness.engine.loop import HarnessEngine
from mullet.harness.fallback.state import (
    default_registry as fallback_registry,
)
from mullet.harness.sessions.store import SessionStore

pytestmark = pytest.mark.harness


class _ScriptedProvider(Provider):
    """Provider whose ``create`` runs a queued script of behaviours.

    Each script entry is either an ``LLMResponse`` (return) or an
    ``Exception`` (raise).  We track the *model* the engine asked for so
    integration tests can assert which call hit which model.
    """

    def __init__(
        self, script_for_model: dict[str, deque], default_model: str
    ) -> None:
        self.script_for_model = script_for_model
        self.default_model = default_model
        self.calls: list[str] = []

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        # The model comes from `self._asked_model` set by the resolver
        # patch right before each call.
        model = getattr(self, "_asked_model", self.default_model) or self.default_model
        self.calls.append(model)
        script = self.script_for_model.get(model)
        if not script:
            return LLMResponse(content=f"OK from {model}", finish_reason="stop")
        action = script.popleft()
        if isinstance(action, BaseException):
            raise action
        return action


@pytest.fixture(autouse=True)
def _clear_fallback_registry():
    """Ensure each test starts with a clean cross-test global registry."""
    fallback_registry.clear()
    yield
    fallback_registry.clear()


def _install_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: _ScriptedProvider,
) -> None:
    def _resolver(name: str, model: str | None = None):
        provider._asked_model = model  # type: ignore[attr-defined]
        return provider

    monkeypatch.setattr(
        "mullet.harness.engine.loop._resolve_provider", _resolver
    )


def _make_engine(tmp_path) -> HarnessEngine:
    store = SessionStore(base_dir=tmp_path / "sessions")
    # Use the engine's default-built HookRegistry so the fallback hook is
    # auto-registered for the integration paths under test.
    return HarnessEngine(store=store)


class TestEngineFallbackIntegration:
    def test_engine_swaps_model_on_retryable_error_then_succeeds(
        self, tmp_path, monkeypatch
    ):
        scripts = {
            "kimi-k2.6": deque([RuntimeError("Rate limit exceeded")]),
            "kimi-k2.5": deque(
                [LLMResponse(content="ok-from-kimi-k2.5", finish_reason="stop")]
            ),
        }
        provider = _ScriptedProvider(scripts, default_model="kimi-k2.6")
        _install_provider(monkeypatch, provider)

        engine = _make_engine(tmp_path)
        result = engine.run("hello", model="kimi-k2.6")

        assert result["session_id"]
        assert result["response"] == "ok-from-kimi-k2.5"
        assert result["stop_reason"] == "stop"
        # Two calls were made: first to kimi-k2.6 (failed), then to the
        # fallback selected by the hook (kimi-k2.5).
        assert provider.calls == ["kimi-k2.6", "kimi-k2.5"]

        # Per-session state was registered.
        state = fallback_registry.get(result["session_id"])
        assert state is not None
        assert state.original_model == "kimi-k2.6"
        assert state.current_model == "kimi-k2.5"
        assert state.fallback_index == 1
        assert state.is_in_cooldown("kimi-k2.6") is True

    def test_engine_does_not_retry_on_non_retryable(
        self, tmp_path, monkeypatch
    ):
        scripts = {
            "kimi-k2.6": deque(
                [
                    RuntimeError(
                        "Authentication failed (invalid API key)"
                    )
                ]
            ),
        }
        provider = _ScriptedProvider(scripts, default_model="kimi-k2.6")
        _install_provider(monkeypatch, provider)

        engine = _make_engine(tmp_path)
        with pytest.raises(RuntimeError, match="invalid API key"):
            engine.run("hello", model="kimi-k2.6")

        # Only one call attempted.
        assert provider.calls == ["kimi-k2.6"]
        # No fallback state registered (the hook bailed early).
        assert len(fallback_registry) == 0

    def test_engine_caps_at_one_fallback_retry_per_run(
        self, tmp_path, monkeypatch
    ):
        scripts = {
            "kimi-k2.6": deque([RuntimeError("Rate limit exceeded")]),
            "kimi-k2.5": deque(
                [RuntimeError("503 service unavailable on fallback")]
            ),
        }
        provider = _ScriptedProvider(scripts, default_model="kimi-k2.6")
        _install_provider(monkeypatch, provider)

        engine = _make_engine(tmp_path)
        with pytest.raises(RuntimeError, match="service unavailable"):
            engine.run("hello", model="kimi-k2.6")

        # Engine attempted exactly two calls: original + one fallback.
        assert provider.calls == ["kimi-k2.6", "kimi-k2.5"]
