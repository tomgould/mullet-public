"""Tests for the error recovery system."""

from mullet.core.agent.recovery import (
    detect_context_error,
    detect_edit_error,
    detect_session_error,
    detect_and_recover,
    truncate_for_context,
    RecoveryAction,
)


# --- Context Window Recovery ---

def test_context_exceeded():
    r = detect_context_error("Error: Context length has been exceeded.")
    assert r is not None
    assert r.action == RecoveryAction.TRUNCATE
    assert r.severity == 3


def test_token_limit():
    r = detect_context_error("token limit exceeded: 128000 max")
    assert r is not None
    assert r.action == RecoveryAction.TRUNCATE
    assert r.context_limit == 128000


def test_too_many_tokens():
    r = detect_context_error("Error: too many tokens in request (150,000 > 128,000)")
    assert r is not None
    assert r.context_limit == 150000


def test_context_not_error():
    assert detect_context_error("Everything worked fine") is None


# --- Edit Error Recovery ---

def test_string_not_found():
    r = detect_edit_error("old_string not found in file")
    assert r is not None
    assert r.action == RecoveryAction.RETRY
    assert "read the file" in r.message.lower()


def test_file_not_found():
    r = detect_edit_error("FileNotFoundError: No such file or directory: 'x.py'")
    assert r is not None
    assert "file_not_found" in r.error_type


def test_permission_denied():
    r = detect_edit_error("PermissionError: Permission denied")
    assert r is not None
    assert "permission" in r.error_type


def test_syntax_error():
    r = detect_edit_error("SyntaxError: invalid syntax (line 42)")
    assert r is not None
    assert "syntax" in r.message.lower()


def test_indent_error():
    r = detect_edit_error("IndentationError: unexpected indent")
    assert r is not None
    assert "indent" in r.error_type


def test_edit_not_error():
    assert detect_edit_error("File written successfully") is None


# --- Session Error Recovery ---

def test_rate_limit():
    r = detect_session_error("429: Rate limit exceeded")
    assert r is not None
    assert r.action == RecoveryAction.RETRY
    assert r.severity == 1


def test_connection_refused():
    r = detect_session_error("ConnectionRefusedError: connection refused")
    assert r is not None
    assert r.action == RecoveryAction.RETRY


def test_auth_error():
    r = detect_session_error("Authentication failed: invalid token")
    assert r is not None
    assert r.action == RecoveryAction.NONE  # Can't auto-recover
    assert r.severity == 3


def test_api_key_invalid():
    r = detect_session_error("Error: API key is invalid or expired")
    assert r is not None
    assert r.action == RecoveryAction.NONE


def test_session_not_error():
    assert detect_session_error("Task completed successfully") is None


# --- detect_and_recover (priority ordering) ---

def test_recover_context_priority():
    r = detect_and_recover("Context size has been exceeded. Also file not found.")
    assert r.error_type.startswith("context_")  # Context wins


def test_recover_edit_priority():
    r = detect_and_recover("old_string not found in file. Rate limit warning.")
    assert r.error_type.startswith("edit_")  # Edit wins over session


def test_recover_unknown():
    r = detect_and_recover("Something weird happened")
    assert r.action == RecoveryAction.NONE
    assert r.error_type == "unknown"


def test_recover_empty():
    r = detect_and_recover("")
    assert r.action == RecoveryAction.NONE


# --- truncate_for_context ---

def test_truncate_keeps_system():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "resp1"},
        {"role": "user", "content": "msg2"},
        {"role": "assistant", "content": "resp2"},
        {"role": "user", "content": "msg3"},
    ]
    result = truncate_for_context(messages, keep_recent=2)
    assert result[0]["role"] == "system"
    assert len(result) == 3  # system + 2 recent
    assert result[-1]["content"] == "msg3"


def test_truncate_short_list():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    result = truncate_for_context(messages, keep_recent=10)
    assert len(result) == 2  # No truncation needed


def test_edit_guidance_unknown_error_type():
    """Line 118: unknown error_type falls through to generic message."""
    from mullet.core.agent.recovery import _edit_guidance
    result = _edit_guidance("disk_full", "no space left")
    assert "disk_full" in result
    assert "Edit failed" in result


def test_detect_and_recover_unknown_error():
    """Lines 191-196: error not matching any known patterns → RecoveryAction.NONE."""
    result = detect_and_recover("completely unknown blah blah xyz999")
    assert result.action == RecoveryAction.NONE


def test_detect_and_recover_session_error_branch():
    """Line 188: detect_session_error match → return that result."""
    result = detect_and_recover("connection refused by server")
    assert result.action == RecoveryAction.RETRY
    assert "session" in result.error_type
