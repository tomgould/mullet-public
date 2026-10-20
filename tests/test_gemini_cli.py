"""Tests for Gemini CLI adapter provider."""

from unittest.mock import patch, MagicMock
import subprocess

from mullet.core.providers.gemini_cli import GeminiCLIProvider, is_gemini_cli_available


def test_init():
    p = GeminiCLIProvider()
    assert p.name == "gemini-cli"
    assert p.model == "gemini-2.5-flash"
    assert p.supports_tool_calling is False


def test_messages_to_prompt():
    p = GeminiCLIProvider()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    prompt = p._messages_to_prompt(messages)
    assert "Instructions: You are helpful." in prompt
    assert "Hello" in prompt


def test_create_success():
    p = GeminiCLIProvider()
    mock_result = MagicMock()
    mock_result.stdout = "The answer is 42."
    mock_result.stderr = ""
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        response = p.create(messages=[{"role": "user", "content": "test"}])

    assert response.content == "The answer is 42."
    assert response.finish_reason == "stop"
    assert response.provider == "gemini-cli"
    assert response.tool_calls is None


def test_create_error():
    p = GeminiCLIProvider()
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "Some API error"  # Non-permission error
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        response = p.create(messages=[{"role": "user", "content": "test"}])

    assert "Error" in response.content or "API error" in response.content


def test_create_timeout():
    p = GeminiCLIProvider()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("gemini", 120)):
        response = p.create(messages=[{"role": "user", "content": "test"}])

    assert "timed out" in response.content.lower()
    assert response.finish_reason == "error"


def test_create_not_found():
    p = GeminiCLIProvider()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        response = p.create(messages=[{"role": "user", "content": "test"}])

    assert "not found" in response.content.lower()


def test_health_check_available():
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        assert is_gemini_cli_available() is True


def test_health_check_not_available():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert is_gemini_cli_available() is False


def test_health_check_delegates():
    provider = GeminiCLIProvider.__new__(GeminiCLIProvider)
    with patch("mullet.core.providers.gemini_cli.is_gemini_cli_available", return_value=True):
        assert provider.health_check() is True


def test_create_permission_denied_fallback():
    p = GeminiCLIProvider()

    fail_result = MagicMock()
    fail_result.stdout = ""
    fail_result.stderr = "Permission denied"
    fail_result.returncode = 1

    success_result = MagicMock()
    success_result.stdout = "response text"
    success_result.stderr = ""
    success_result.returncode = 0

    with patch("subprocess.run", side_effect=[fail_result, success_result]):
        response = p.create(messages=[{"role": "user", "content": "hi"}])

    assert "response text" in response.content
