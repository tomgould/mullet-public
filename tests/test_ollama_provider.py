"""Tests for the refactored sync OllamaProvider."""
import json
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers.ollama import OllamaConfig, OllamaProvider
from mullet.core.providers.base import LLMResponse


@pytest.fixture
def provider():
    return OllamaProvider(OllamaConfig(base_url="http://localhost:11434", model="test-model"))


def _make_urlopen_response(body: bytes):
    """Return a mock context-manager response for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_streaming_response(lines: list[bytes]):
    """Return a mock response that iterates NDJSON lines."""
    mock_resp = MagicMock()
    mock_resp.__iter__ = MagicMock(return_value=iter(lines))
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_provider_init(provider):
    assert provider.config.model == "test-model"
    assert provider.config.base_url == "http://localhost:11434"


def test_create_success(provider):
    body = json.dumps({"message": {"content": "ready"}, "done": True}).encode()
    with patch("urllib.request.urlopen", return_value=_make_urlopen_response(body)):
        resp = provider.create([{"role": "user", "content": "say ready"}])
    assert isinstance(resp, LLMResponse)
    assert resp.content == "ready"
    assert resp.finish_reason == "stop"
    assert resp.provider == "ollama"


def test_create_not_done(provider):
    body = json.dumps({"message": {"content": "partial"}, "done": False}).encode()
    with patch("urllib.request.urlopen", return_value=_make_urlopen_response(body)):
        resp = provider.create([{"role": "user", "content": "hi"}])
    assert resp.finish_reason == "length"


def test_create_connection_error(provider):
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(ConnectionError, match="Ollama unreachable"):
            provider.create([{"role": "user", "content": "hi"}])


def test_create_stream_success(provider):
    lines = [
        json.dumps({"message": {"content": "chunk1"}, "done": False}).encode() + b"\n",
        json.dumps({"message": {"content": "chunk2"}, "done": False}).encode() + b"\n",
        json.dumps({"message": {"content": "chunk3"}, "done": True}).encode() + b"\n",
    ]
    with patch("urllib.request.urlopen", return_value=_make_streaming_response(lines)):
        chunks = list(provider.create_stream([{"role": "user", "content": "stream"}]))
    texts = [c.content for c in chunks if c.content]
    assert texts == ["chunk1", "chunk2", "chunk3"]
    assert chunks[-1].finish_reason == "stop"


def test_create_stream_skips_invalid_json(provider):
    lines = [
        json.dumps({"message": {"content": "ok"}, "done": False}).encode() + b"\n",
        b"not-json\n",
        json.dumps({"message": {"content": "done"}, "done": True}).encode() + b"\n",
    ]
    with patch("urllib.request.urlopen", return_value=_make_streaming_response(lines)):
        chunks = list(provider.create_stream([{"role": "user", "content": "stream"}]))
    texts = [c.content for c in chunks if c.content]
    assert texts == ["ok", "done"]


def test_supports_tool_calling(provider):
    assert provider.supports_tool_calling is False


def test_create_stream_skips_empty_lines(provider):
    lines = [
        b"",
        json.dumps({"message": {"content": "hi"}, "done": True}).encode() + b"\n",
    ]
    with patch("urllib.request.urlopen", return_value=_make_streaming_response(lines)):
        chunks = list(provider.create_stream([{"role": "user", "content": "hi"}]))
    assert len(chunks) == 1
    assert chunks[0].content == "hi"


def test_create_stream_urlerror_raises_connection_error(provider):
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
        with pytest.raises(ConnectionError, match="Ollama unreachable"):
            list(provider.create_stream([{"role": "user", "content": "hi"}]))
