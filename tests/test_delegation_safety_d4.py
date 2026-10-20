"""Delegation safety D4 — dispatch-time context-fit assertion at the HTTP wire.

D4 is the last-chance safety net: after the router (D1), candidate filter
(D2) and structured error (D3) have run, each provider's ``_call_with_retry``
re-verifies that the composed payload fits the picked model's ``max_context``
BEFORE issuing the real HTTP call.

Invariants:
  1. The shared helper ``assert_context_fits`` raises
     ``ModelContextExceeded(all_candidates_exhausted=True)`` when a prompt
     would exceed the model's ``max_context * (1 / CONTEXT_SAFETY_MARGIN)``.
     Exhausted-True is correct because by the time D4 runs the caller has
     already committed to this provider+model — there is no fallback left.
  2. The helper SOFT-FAILS on estimator or registry errors — D4 is a net,
     not a gate. An internal crash must not block an otherwise-valid call.
  3. Each of ``openai_compat.OpenAICompatibleProvider``,
     ``anthropic.AnthropicProvider`` and ``gemini.GeminiProvider`` calls
     the helper as the FIRST thing their ``_call_with_retry`` does, so the
     block happens BEFORE the HTTP client is touched.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers.errors import ModelContextExceeded


# ---------------------------------------------------------------------------
# 1. Shared helper — unit tests for context_gate.assert_context_fits
# ---------------------------------------------------------------------------


def test_assert_context_fits_passes_for_small_prompt():
    """Red: a prompt that fits comfortably must not raise.

    Failure mode: over-zealous gate blocks every dispatch; all real calls
    start failing with MCE even though the picker did its job.
    """
    from mullet.core.providers.context_gate import assert_context_fits

    # kimi-k2.5 has max_context=262144. A one-line message is ~a few tokens.
    messages = [{"role": "user", "content": "ping"}]
    # Must not raise.
    assert_context_fits("kimi", "kimi-k2.5", messages)


def test_assert_context_fits_raises_for_overflow():
    """Red: a prompt that overflows max_context must raise MCE with exhausted=True.

    Failure mode: a 14K prompt ends up at moonshot-v1-8k (8192 ctx) and
    the HTTP 400 from the provider is the first sign of the mismatch —
    exactly the failure mode D4 exists to catch.
    """
    from mullet.core.providers.context_gate import assert_context_fits

    # moonshot-v1-8k has max_context=8_192. Build a message whose JSON
    # encoding clearly overflows even with the 6 chars/token fallback.
    big_content = "A" * 80_000  # ~13K tokens with cl100k_base, ~13K with fallback
    messages = [{"role": "user", "content": big_content}]

    with pytest.raises(ModelContextExceeded) as exc_info:
        assert_context_fits("kimi", "moonshot-v1-8k", messages)

    err = exc_info.value
    assert err.all_candidates_exhausted is True, (
        "At dispatch time there is no fallback left — exhausted must be True"
    )
    assert err.provider == "kimi"
    assert err.tried_candidates == ["moonshot-v1-8k"]
    assert err.max_candidate_context == 8_192
    assert err.context_tokens > 8_192


def test_assert_context_fits_soft_fails_on_estimator_error():
    """Red: if ``estimate_tokens`` raises, the gate must SWALLOW the error
    and let the call proceed — D4 is a safety net, not a hard gate.

    Failure mode: a tiktoken install hiccup or a weird message shape
    causes estimate_tokens to throw and D4 starts blocking every call.
    """
    from mullet.core.providers import context_gate

    def _boom(_obj):
        raise RuntimeError("tiktoken is sad")

    messages = [{"role": "user", "content": "hello"}]
    with patch.object(context_gate, "estimate_tokens", side_effect=_boom):
        # Must not raise — soft-fail path.
        context_gate.assert_context_fits("kimi", "kimi-k2.5", messages)


def test_assert_context_fits_soft_fails_on_unknown_model():
    """Red: if the constraint registry throws on an unknown model, proceed.

    Failure mode: a new model id slips through that the registry doesn't
    know about, and D4 starts blocking it instead of letting the real
    provider decide.
    """
    from mullet.core.providers import context_gate

    fake_registry = MagicMock()
    fake_registry.get.side_effect = KeyError("who-is-this-model-even")

    messages = [{"role": "user", "content": "hello"}]
    with patch.object(
        context_gate, "get_constraint_registry", return_value=fake_registry
    ):
        # Must not raise — soft-fail path.
        context_gate.assert_context_fits(
            "openrouter", "some-unknown-exotic-model-v9000", messages
        )


def test_assert_context_fits_soft_fails_when_max_context_is_zero():
    """Red: a constraint entry with max_context=0/None must skip the check.

    Failure mode: a misconfigured entry where max_context is falsy must
    NOT cause every dispatch to raise (the arithmetic would be nonsense).
    """
    from mullet.core.providers import context_gate
    from mullet.core.providers.constraints import ModelConstraints

    fake_registry = MagicMock()
    # max_context=0 — gate must silently skip rather than raise.
    fake_registry.get.return_value = ModelConstraints(max_context=0)

    messages = [{"role": "user", "content": "A" * 100_000}]
    with patch.object(
        context_gate, "get_constraint_registry", return_value=fake_registry
    ):
        context_gate.assert_context_fits("kimi", "weird-model", messages)


# ---------------------------------------------------------------------------
# 2. OpenAICompatibleProvider — gate blocks oversize prompt BEFORE HTTP
# ---------------------------------------------------------------------------


def _make_openai_compat_provider(model: str):
    """Construct an OpenAICompatibleProvider without touching the real SDK.

    We bypass __init__ (which would try to reach LM Studio / OpenAI)
    and set just the fields _call_with_retry needs.
    """
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider

    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.client = MagicMock()
    # chat.completions.create is what D4 must prevent from being called.
    p.client.chat.completions.create = MagicMock(
        side_effect=AssertionError(
            "HTTP call was reached — D4 gate failed to block oversize prompt"
        )
    )
    p.model = model
    p.name = "kimi"
    p._openrouter_config = {}
    return p


def test_openai_compat_blocks_oversize_prompt():
    """Red: _call_with_retry on OpenAI-compat provider must raise MCE
    BEFORE touching ``client.chat.completions.create`` when the prompt
    exceeds the picked model's max_context.

    Failure mode: the wire call goes out and returns 400 "request
    exceeded model token limit" — exactly the Cycle 1 regression.
    """
    provider = _make_openai_compat_provider("moonshot-v1-8k")

    # Build a kwargs payload with messages totalling ~14K tokens.
    big_content = "A" * 90_000
    kwargs = {
        "model": "moonshot-v1-8k",
        "messages": [{"role": "user", "content": big_content}],
        "max_tokens": 4096,
    }

    with pytest.raises(ModelContextExceeded) as exc_info:
        provider._call_with_retry(kwargs)

    err = exc_info.value
    assert err.all_candidates_exhausted is True
    assert err.provider == "kimi"
    assert err.tried_candidates == ["moonshot-v1-8k"]
    assert err.max_candidate_context == 8_192
    # Confirm the HTTP client was NEVER called.
    provider.client.chat.completions.create.assert_not_called()


def test_openai_compat_allows_within_context_prompt():
    """Red: a prompt that fits the model's max_context must sail past D4
    and reach the HTTP client unmodified (sanity check — no over-blocking).
    """
    provider = _make_openai_compat_provider("kimi-k2.5")

    # Replace the always-raising side_effect with a fake successful response.
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "ok"
    fake_response.choices[0].message.tool_calls = None
    fake_response.choices[0].finish_reason = "stop"
    fake_response.usage.prompt_tokens = 3
    fake_response.usage.completion_tokens = 1
    fake_response.model = "kimi-k2.5"
    provider.client.chat.completions.create = MagicMock(return_value=fake_response)

    kwargs = {
        "model": "kimi-k2.5",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 16,
    }

    provider._call_with_retry(kwargs)
    provider.client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# 3. AnthropicProvider — gate blocks oversize prompt BEFORE HTTP
# ---------------------------------------------------------------------------


def _make_anthropic_provider(model: str):
    """Construct an AnthropicProvider without hitting the real SDK."""
    from mullet.core.providers.anthropic import AnthropicProvider

    p = AnthropicProvider.__new__(AnthropicProvider)
    p.client = MagicMock()
    p.client.messages.create = MagicMock(
        side_effect=AssertionError(
            "HTTP call was reached — D4 gate failed to block oversize prompt"
        )
    )
    p.model = model
    p.name = "anthropic"
    return p


def test_anthropic_blocks_oversize_prompt():
    """Red: Anthropic _call_with_retry must raise MCE BEFORE touching
    ``client.messages.create`` when the prompt exceeds max_context.

    Failure mode: the request reaches Anthropic and returns a 400
    invalid_request_error about input length.
    """
    # claude-haiku-4-5-20251001 has max_context=200_000 per constraints table.
    provider = _make_anthropic_provider("claude-haiku-4-5-20251001")

    # A 400K-char payload tokenises to ~100K with cl100k_base and ~66K with
    # the 6-char fallback — neither overflows 200K alone. Patch the registry
    # to a smaller max_context so the gate trips without feeding a monster
    # payload into tiktoken (which would slow the suite to a crawl).
    from mullet.core.providers import context_gate
    from mullet.core.providers.constraints import ModelConstraints

    big_content = "A" * 80_000  # ~13K tokens
    kwargs = {
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": big_content}],
        "max_tokens": 1024,
    }

    fake_registry = MagicMock()
    fake_registry.get.return_value = ModelConstraints(max_context=8_192)

    with patch.object(
        context_gate, "get_constraint_registry", return_value=fake_registry
    ):
        with pytest.raises(ModelContextExceeded) as exc_info:
            provider._call_with_retry(kwargs)

    err = exc_info.value
    assert err.all_candidates_exhausted is True
    assert err.provider == "anthropic"
    assert err.tried_candidates == ["claude-haiku-4-5-20251001"]
    assert err.max_candidate_context == 8_192
    provider.client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# 4. GeminiProvider — gate blocks oversize prompt BEFORE HTTP
# ---------------------------------------------------------------------------


def _make_gemini_provider(model: str):
    """Construct a GeminiProvider without hitting the real SDK."""
    try:
        from mullet.core.providers.gemini import GeminiProvider
    except ImportError:
        pytest.skip("gemini provider not importable in this env")

    p = GeminiProvider.__new__(GeminiProvider)
    p.client = MagicMock()
    p.client.models.generate_content = MagicMock(
        side_effect=AssertionError(
            "HTTP call was reached — D4 gate failed to block oversize prompt"
        )
    )
    p.model = model
    p.name = "gemini"
    return p


def test_gemini_blocks_oversize_prompt():
    """Red: Gemini _call_with_retry must raise MCE BEFORE touching
    ``client.models.generate_content`` when contents exceed max_context.

    Failure mode: Gemini SDK sends a 2M-token request and gets
    400 INVALID_ARGUMENT — exactly the kind of late failure D4 catches.

    Uses constraint-injection (rather than a literal 13M-char payload,
    which would freeze tiktoken). We patch the registry to report a
    1024-token model so a small overflow trips the gate.
    """
    from mullet.core.providers import context_gate
    from mullet.core.providers.constraints import ModelConstraints

    provider = _make_gemini_provider("gemini-tiny-test")

    # ~3K tokens of contents — overflows a 1024-ctx model by 3x.
    big_content = "A" * 18_000
    contents = [{"role": "user", "parts": [{"text": big_content}]}]
    fake_config = MagicMock()

    fake_registry = MagicMock()
    fake_registry.get.return_value = ModelConstraints(max_context=1024)

    with patch.object(
        context_gate, "get_constraint_registry", return_value=fake_registry
    ):
        with pytest.raises(ModelContextExceeded) as exc_info:
            provider._call_with_retry(contents, fake_config)

    err = exc_info.value
    assert err.all_candidates_exhausted is True
    assert err.provider == "gemini"
    assert err.tried_candidates == ["gemini-tiny-test"]
    assert err.max_candidate_context == 1024
    provider.client.models.generate_content.assert_not_called()
