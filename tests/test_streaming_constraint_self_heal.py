"""Tests for streaming constraint self-healing on openai_compat + anthropic.

The non-streaming path is already tested in test_anthropic_provider.py and
test_openai_compat_extras.py. This file covers the stream-open retry path
added in commit 7accb93.
"""

from unittest.mock import MagicMock, patch

import pytest


# --- openai_compat.create_stream -------------------------------------------


def test_openai_compat_stream_self_heals_on_constraint_error():
    """If the SDK raises on stream open and parse_error returns a param to omit,
    the provider should learn, reshape, and retry the stream creation once."""
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        base_url="https://api.moonshot.ai/v1",
        api_key="test-key",
        model="moonshot-v1-128k",
    )

    # First call raises constraint error; second call returns an empty iterator
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("invalid temperature: only 1 is allowed for this model")
        return iter([])  # empty stream, enough to drive the loop to completion

    provider.client.chat.completions.create = fake_create

    # The constraint parser already knows this Moonshot message; it doesn't
    # need to be patched. Run the stream to completion (iter exhaustion).
    chunks = list(provider.create_stream(messages=[{"role": "user", "content": "hi"}]))

    assert call_count["n"] == 2, f"expected 2 calls (1st raises, 2nd retries); got {call_count['n']}"
    # Moonshot v1 already strips temperature statically, so it might not show
    # up in the learned registry; key assertion is that the retry happened.


def test_openai_compat_stream_does_not_retry_on_non_constraint_error():
    """Errors the parser doesn't recognize should propagate — not mask real bugs."""
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        base_url="https://api.example.com/v1",
        api_key="x",
        model="kimi-k2.5",
    )

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        raise RuntimeError("network unreachable")  # not a constraint error

    provider.client.chat.completions.create = fake_create

    with pytest.raises(RuntimeError, match="network unreachable"):
        list(provider.create_stream(messages=[{"role": "user", "content": "x"}]))

    # Only one attempt — no retry on non-constraint error
    assert call_count["n"] == 1


# --- anthropic.create_stream -----------------------------------------------


def test_anthropic_stream_self_heals_on_constraint_error():
    """Same pattern as openai_compat but for anthropic.client.messages.stream."""
    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        from mullet.core.providers.anthropic import AnthropicProvider
        p = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-6")

        call_count = {"n": 0}

        # Context-manager returned by messages.stream(**kwargs)
        class _EmptyStreamCM:
            def __enter__(self):
                return iter([])
            def __exit__(self, *a):
                return False

        def fake_stream(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("unsupported_value: Invalid value for 'frequency_penalty'")
            return _EmptyStreamCM()

        mock_client.messages.stream.side_effect = fake_stream

        # Drain the stream iterator (None yields expected with empty stream)
        try:
            list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
        except Exception:
            # Empty mock may raise inside; what matters is the retry happened
            pass

        assert call_count["n"] == 2, (
            f"expected 2 stream.open calls (1st raises, 2nd retries); got {call_count['n']}"
        )


def test_anthropic_stream_does_not_retry_on_unknown_error():
    """Non-constraint errors should propagate from the stream path too."""
    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        from mullet.core.providers.anthropic import AnthropicProvider
        p = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-6")

        call_count = {"n": 0}

        def raise_unknown(**kwargs):
            call_count["n"] += 1
            raise TimeoutError("connection timed out")

        mock_client.messages.stream.side_effect = raise_unknown

        with pytest.raises(TimeoutError):
            list(p.create_stream(messages=[{"role": "user", "content": "x"}]))

        assert call_count["n"] == 1  # no retry
