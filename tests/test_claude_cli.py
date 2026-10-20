"""Tests for Claude CLI provider."""
import subprocess
from unittest.mock import MagicMock, patch

from mullet.core.providers.claude_cli import ClaudeCLIProvider, is_claude_cli_available


@patch("subprocess.run")
def test_is_available(mock_run):
    mock_run.return_value.returncode = 0
    assert is_claude_cli_available() is True


@patch("subprocess.run", side_effect=FileNotFoundError)
def test_not_available(mock_run):
    assert is_claude_cli_available() is False


@patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5))
def test_is_available_timeout(mock_run):
    assert is_claude_cli_available() is False


def test_provider_init():
    p = ClaudeCLIProvider(model="sonnet")
    assert p.name == "claude-cli"
    assert p.model == "sonnet"


def test_provider_default_model():
    p = ClaudeCLIProvider()
    assert p.model in ("sonnet", "claude-sonnet-4-6")


def _make_proc(returncode=0, stdout="hello", stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def test_create_success():
    provider = ClaudeCLIProvider(model="haiku")
    messages = [{"role": "user", "content": "Say hi"}]
    with patch("subprocess.run", return_value=_make_proc(0, "Hi there", "")):
        response = provider.create(messages)
    assert response.content == "Hi there"
    assert response.usage.total_tokens > 0


def test_create_nonzero_returncode_returns_error():
    provider = ClaudeCLIProvider()
    messages = [{"role": "user", "content": "fail"}]
    with patch("subprocess.run", return_value=_make_proc(1, "", "auth error")):
        response = provider.create(messages)
    assert "Error" in response.content
    assert response.usage.total_tokens == 0


def test_create_nonzero_with_no_stderr():
    provider = ClaudeCLIProvider()
    messages = [{"role": "user", "content": "fail"}]
    with patch("subprocess.run", return_value=_make_proc(2, "", "")):
        response = provider.create(messages)
    assert "code 2" in response.content


def test_create_timeout():
    provider = ClaudeCLIProvider()
    messages = [{"role": "user", "content": "slow"}]
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300)):
        response = provider.create(messages)
    assert "timed out" in response.content.lower()
    assert response.usage.total_tokens == 0


def test_create_success_log_interaction_failure_is_swallowed():
    provider = ClaudeCLIProvider()
    messages = [{"role": "user", "content": "x"}]
    with patch("subprocess.run", return_value=_make_proc(0, "ok")):
        with patch(
            "mullet.core.logging.interaction_log.log_interaction",
            side_effect=RuntimeError("log failed"),
        ):
            response = provider.create(messages)
    assert response.content == "ok"


def test_create_error_log_interaction_failure_is_swallowed():
    provider = ClaudeCLIProvider()
    messages = [{"role": "user", "content": "x"}]
    with patch("subprocess.run", return_value=_make_proc(1, "", "err")):
        with patch(
            "mullet.core.logging.interaction_log.log_interaction",
            side_effect=RuntimeError("log failed"),
        ):
            response = provider.create(messages)
    assert "Error" in response.content


def test_create_timeout_log_interaction_failure_is_swallowed():
    provider = ClaudeCLIProvider()
    messages = [{"role": "user", "content": "x"}]
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300)):
        with patch(
            "mullet.core.logging.interaction_log.log_interaction",
            side_effect=RuntimeError("log failed"),
        ):
            response = provider.create(messages)
    assert "timed out" in response.content.lower()


def test_stream_yields_single_chunk():
    provider = ClaudeCLIProvider()
    messages = [{"role": "user", "content": "hi"}]
    with patch("subprocess.run", return_value=_make_proc(0, "stream result")):
        chunks = list(provider.stream(messages))
    assert len(chunks) == 1
    assert chunks[0].content == "stream result"
    assert chunks[0].finish_reason == "stop"


def test_messages_to_prompt_system():
    msgs = [{"role": "system", "content": "Be helpful."}]
    prompt = ClaudeCLIProvider._messages_to_prompt(msgs)
    assert prompt == "System: Be helpful."


def test_messages_to_prompt_user():
    msgs = [{"role": "user", "content": "hello"}]
    assert ClaudeCLIProvider._messages_to_prompt(msgs) == "hello"


def test_messages_to_prompt_assistant():
    msgs = [{"role": "assistant", "content": "I can help."}]
    prompt = ClaudeCLIProvider._messages_to_prompt(msgs)
    assert "Previous response:" in prompt


def test_messages_to_prompt_list_content():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "part one"},
        {"type": "image"},
        {"type": "text", "text": "part two"},
    ]}]
    prompt = ClaudeCLIProvider._messages_to_prompt(msgs)
    assert "part one" in prompt
    assert "part two" in prompt


def test_messages_to_prompt_multi_turn():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "answer"},
    ]
    prompt = ClaudeCLIProvider._messages_to_prompt(msgs)
    assert "System: sys" in prompt
    assert "ask" in prompt
    assert "Previous response: answer" in prompt
