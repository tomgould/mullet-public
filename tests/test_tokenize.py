"""Tests for mullet.core.providers.tokenize.estimate_tokens."""
from __future__ import annotations

import json

import pytest

from mullet.core.providers import tokenize


@pytest.fixture(autouse=True)
def _reset_encoder_cache():
    """Ensure each test starts with a fresh lazy-encoder state."""
    tokenize._ENCODER = None
    yield
    tokenize._ENCODER = None


def test_estimate_tokens_without_tiktoken_fallback(monkeypatch):
    """When tiktoken is unavailable, fallback uses 6 chars/token heuristic."""
    # Force the lazy loader to signal "no tiktoken available"
    monkeypatch.setattr(tokenize, "_lazy_encoder", lambda: False)

    text = "a" * 60  # 60 chars → 60 // 6 = 10 tokens
    assert tokenize.estimate_tokens(text) == 10

    # Very short input still returns at least 1
    assert tokenize.estimate_tokens("x") == 1


def test_estimate_tokens_with_tiktoken():
    """When tiktoken is installed, returns encoder-derived count (approximate)."""
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        pytest.skip("tiktoken not installed")

    # A known-ish English sentence — cl100k_base encodes "Hello, world!" as ~4 tokens
    sentence = "Hello, world!"
    tokens = tokenize.estimate_tokens(sentence)
    assert 2 <= tokens <= 8, f"expected ~4 tokens, got {tokens}"

    # Encoder should be cached after first call (not False sentinel)
    assert tokenize._ENCODER is not None and tokenize._ENCODER is not False


def test_estimate_tokens_on_dict_and_list():
    """estimate_tokens accepts non-string inputs by JSON-serialising them."""
    payload = {"role": "user", "content": "hello"}
    result = tokenize.estimate_tokens(payload)
    assert isinstance(result, int)
    assert result >= 1

    # Must be ~equal to tokenising the same JSON manually
    manual = tokenize.estimate_tokens(json.dumps(payload))
    assert result == manual

    # Lists work too
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert tokenize.estimate_tokens(messages) >= 1


def test_estimate_tokens_empty_string(monkeypatch):
    """Empty input via the fallback path clamps to 1."""
    monkeypatch.setattr(tokenize, "_lazy_encoder", lambda: False)
    assert tokenize.estimate_tokens("") == 1


def test_estimate_tokens_handles_non_serialisable(monkeypatch):
    """Objects that json.dumps can't handle fall back to str()."""
    monkeypatch.setattr(tokenize, "_lazy_encoder", lambda: False)

    class Weird:
        def __repr__(self):
            return "weird-object-repr-xxxx"

    # Would normally raise TypeError in json.dumps without default=str — but
    # estimate_tokens swallows that and uses str() as the last resort.
    result = tokenize.estimate_tokens(Weird())
    assert result >= 1


def test_lazy_encoder_importerror_sets_false_sentinel():
    """_lazy_encoder hits the except ImportError branch when tiktoken is hidden."""
    import sys
    from unittest.mock import patch

    tokenize._ENCODER = None
    with patch.dict(sys.modules, {"tiktoken": None}):
        enc = tokenize._lazy_encoder()
    assert enc is False
    assert tokenize._ENCODER is False


def test_estimate_tokens_circular_reference_fallback(monkeypatch):
    """Circular dicts trigger ValueError in json.dumps — fallback to str()."""
    monkeypatch.setattr(tokenize, "_lazy_encoder", lambda: False)

    circular: dict = {}
    circular["self"] = circular

    result = tokenize.estimate_tokens(circular)
    assert result >= 1
