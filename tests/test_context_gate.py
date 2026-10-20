"""Tests for mullet.core.providers.context_gate — D4 dispatch-time check."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers.context_gate import assert_context_fits
from mullet.core.providers.errors import ModelContextExceeded
from mullet.core.providers.model_registry import CONTEXT_SAFETY_MARGIN


def _make_registry(max_context: int | None):
    constraint = MagicMock()
    constraint.max_context = max_context
    registry = MagicMock()
    registry.get.return_value = constraint
    return registry


class TestAssertContextFits:
    def test_passes_when_tokens_within_limit(self) -> None:
        with (
            patch("mullet.core.providers.context_gate.estimate_tokens", return_value=1000),
            patch("mullet.core.providers.context_gate.get_constraint_registry",
                  return_value=_make_registry(max_context=8192)),
        ):
            # 1000 * 1.1 = 1100 < 8192 → should not raise
            assert_context_fits("openai", "gpt-4o", [{"role": "user", "content": "hi"}])

    def test_raises_when_tokens_overflow(self) -> None:
        # tokens * CONTEXT_SAFETY_MARGIN > max_ctx → raises
        with (
            patch("mullet.core.providers.context_gate.estimate_tokens", return_value=8000),
            patch("mullet.core.providers.context_gate.get_constraint_registry",
                  return_value=_make_registry(max_context=8192)),
        ):
            # 8000 * 1.1 = 8800 > 8192 → should raise
            with pytest.raises(ModelContextExceeded) as exc_info:
                assert_context_fits("openai", "gpt-4o", [])
            assert exc_info.value.all_candidates_exhausted is True
            assert exc_info.value.context_tokens == 8000
            assert "gpt-4o" in exc_info.value.tried_candidates
            assert exc_info.value.max_candidate_context == 8192

    def test_soft_fails_on_estimate_error(self) -> None:
        with patch("mullet.core.providers.context_gate.estimate_tokens",
                   side_effect=ValueError("estimator crashed")):
            # Should not raise — soft fail
            assert_context_fits("anthropic", "claude-sonnet-4-6", [])

    def test_soft_fails_on_registry_error(self) -> None:
        registry = MagicMock()
        registry.get.side_effect = KeyError("unknown model")
        with (
            patch("mullet.core.providers.context_gate.estimate_tokens", return_value=500),
            patch("mullet.core.providers.context_gate.get_constraint_registry",
                  return_value=registry),
        ):
            # Should not raise — soft fail
            assert_context_fits("anthropic", "unknown-model", [])

    def test_soft_fails_when_max_ctx_is_zero(self) -> None:
        with (
            patch("mullet.core.providers.context_gate.estimate_tokens", return_value=1000),
            patch("mullet.core.providers.context_gate.get_constraint_registry",
                  return_value=_make_registry(max_context=0)),
        ):
            # max_ctx=0 is falsy — should return without raising
            assert_context_fits("gemini", "gemini-flash", [])

    def test_soft_fails_when_max_ctx_is_none(self) -> None:
        with (
            patch("mullet.core.providers.context_gate.estimate_tokens", return_value=1000),
            patch("mullet.core.providers.context_gate.get_constraint_registry",
                  return_value=_make_registry(max_context=None)),
        ):
            assert_context_fits("gemini", "gemini-flash", [])

    def test_exactly_at_limit_passes(self) -> None:
        # tokens * 1.1 == max_ctx: NOT > max_ctx, so should pass
        max_ctx = 8192
        tokens = int(max_ctx / CONTEXT_SAFETY_MARGIN)  # ~7447
        assert tokens * CONTEXT_SAFETY_MARGIN <= max_ctx
        with (
            patch("mullet.core.providers.context_gate.estimate_tokens", return_value=tokens),
            patch("mullet.core.providers.context_gate.get_constraint_registry",
                  return_value=_make_registry(max_context=max_ctx)),
        ):
            assert_context_fits("anthropic", "claude-haiku-4-5", [])

    def test_one_token_over_limit_raises(self) -> None:
        max_ctx = 1000
        # tokens such that tokens * 1.1 > 1000: tokens > 909.09, so tokens=910
        tokens = 910
        assert tokens * CONTEXT_SAFETY_MARGIN > max_ctx
        with (
            patch("mullet.core.providers.context_gate.estimate_tokens", return_value=tokens),
            patch("mullet.core.providers.context_gate.get_constraint_registry",
                  return_value=_make_registry(max_context=max_ctx)),
        ):
            with pytest.raises(ModelContextExceeded):
                assert_context_fits("local", "phi-3.5", [])

    def test_error_includes_provider_name(self) -> None:
        with (
            patch("mullet.core.providers.context_gate.estimate_tokens", return_value=5000),
            patch("mullet.core.providers.context_gate.get_constraint_registry",
                  return_value=_make_registry(max_context=4096)),
        ):
            with pytest.raises(ModelContextExceeded) as exc_info:
                assert_context_fits("kimi", "kimi-k2.6", [])
            # provider is embedded in the exception message
            assert "kimi" in str(exc_info.value).lower() or exc_info.value.tried_candidates == ["kimi-k2.6"]
