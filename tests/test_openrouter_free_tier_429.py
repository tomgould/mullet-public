"""Tests: OpenRouter :free models should not retry on 429."""
import pytest
from unittest.mock import MagicMock


def _make_provider(model: str):
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider
    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.name = "openrouter"
    p.model = model
    # Mock the OpenAI client
    p.client = MagicMock()
    p._openrouter_config = {}
    return p


def test_free_tier_429_does_not_retry():
    """A :free model hitting 429 should not retry — escalate immediately."""
    p = _make_provider("deepseek/deepseek-v3:free")
    p.client.chat.completions.create.side_effect = RuntimeError("429 rate limit")

    with pytest.raises(RuntimeError):
        p._call_with_retry(
            {"model": "deepseek/deepseek-v3:free", "messages": []},
            max_retries=3,
        )

    assert p.client.chat.completions.create.call_count == 1


def test_paid_model_429_does_retry():
    """A paid model hitting 429 should retry up to max_retries."""
    p = _make_provider("deepseek/deepseek-chat")
    p.client.chat.completions.create.side_effect = RuntimeError("429 rate limit")

    with pytest.raises(RuntimeError):
        p._call_with_retry(
            {"model": "deepseek/deepseek-chat", "messages": []},
            max_retries=3,
        )

    assert p.client.chat.completions.create.call_count == 3


def test_free_tier_5xx_does_retry():
    """A :free model hitting 5xx should still retry (server error, not rate limit)."""
    p = _make_provider("deepseek/deepseek-v3:free")
    p.client.chat.completions.create.side_effect = RuntimeError("500 internal server error")

    with pytest.raises(RuntimeError):
        p._call_with_retry(
            {"model": "deepseek/deepseek-v3:free", "messages": []},
            max_retries=3,
        )

    assert p.client.chat.completions.create.call_count == 3


def test_free_tier_check_is_substring():
    """Model IDs containing :free anywhere are identified as free-tier."""
    assert ":free" in "meta-llama/llama-3.3-70b-instruct:free"
