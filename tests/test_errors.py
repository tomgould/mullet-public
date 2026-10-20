"""Tests for error formatting."""

from mullet.core.agent.errors import format_error, ERROR_SUGGESTIONS


def test_connection_refused():
    msg = format_error(ConnectionError("Connection refused"))
    assert "connect" in msg.lower() or "LLM" in msg
    assert "LM Studio" in msg or "Start" in msg


def test_rate_limit():
    msg = format_error(Exception("429 rate limit exceeded"))
    assert "rate limit" in msg.lower()
    assert "Suggestions" in msg


def test_api_key_error():
    msg = format_error(Exception("Invalid API key"))
    assert "key" in msg.lower()


def test_timeout():
    msg = format_error(TimeoutError("Request timed out"))
    assert "timeout" in msg.lower() or "timed out" in msg.lower()


def test_unknown_error():
    msg = format_error(Exception("Something completely unknown"))
    assert "doctor" in msg.lower()


def test_error_suggestions_categories():
    assert len(ERROR_SUGGESTIONS) >= 6
    for key, info in ERROR_SUGGESTIONS.items():
        assert "message" in info
        assert "suggestions" in info
        assert len(info["suggestions"]) >= 2
