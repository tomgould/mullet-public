"""Error recovery system — detect and recover from common failures.

Three recovery strategies with priority ordering:
1. Context window (highest) — token limit exceeded
2. Edit error — file edit/patch failures
3. Session error — general agent failures

Each strategy detects specific error patterns and returns a recovery
action (truncate, retry with fix, reset).

Ported from OMC hooks/recovery/.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """What to do after detecting a recoverable error."""
    NONE = "none"
    TRUNCATE = "truncate"       # Remove old messages to free context
    RETRY = "retry"             # Retry with modified approach
    COMPRESS = "compress"       # Force context compression
    RESET = "reset"             # Reset conversation (last resort)


@dataclass
class RecoveryResult:
    """Result of error detection and recovery analysis."""
    action: RecoveryAction
    error_type: str
    message: str  # Guidance for the agent or user
    severity: int  # 1=low, 2=medium, 3=high
    context_limit: int | None = None  # Detected context limit (tokens)


# --- Context Window Recovery ---

_CONTEXT_PATTERNS = [
    (re.compile(r"context.?(?:length|size|window).*?exceeded", re.IGNORECASE), "context_exceeded"),
    (re.compile(r"maximum.?context.?length.*?(\d[\d,]+)", re.IGNORECASE), "max_context"),
    (re.compile(r"token.?limit.*?exceeded", re.IGNORECASE), "token_limit"),
    (re.compile(r"too.?many.?tokens", re.IGNORECASE), "too_many_tokens"),
    (re.compile(r"input.*?too.?long", re.IGNORECASE), "input_too_long"),
    (re.compile(r"(\d[\d,]+)\s*tokens?.*?maximum.*?(\d[\d,]+)", re.IGNORECASE), "token_count"),
    (re.compile(r"max_tokens.*?(\d+)", re.IGNORECASE), "max_tokens"),
]


def detect_context_error(error_msg: str) -> RecoveryResult | None:
    """Detect context window/token limit errors."""
    for pattern, error_type in _CONTEXT_PATTERNS:
        match = pattern.search(error_msg)
        if match:
            # Try to extract the context limit
            limit = None
            numbers = re.findall(r'(\d[\d,]+)', error_msg)
            if numbers:
                limit = max(int(n.replace(",", "")) for n in numbers)

            return RecoveryResult(
                action=RecoveryAction.TRUNCATE,
                error_type=f"context_{error_type}",
                message="Context window exceeded. Truncating old messages to free space.",
                severity=3,
                context_limit=limit,
            )
    return None


# --- Edit Error Recovery ---

_EDIT_PATTERNS = [
    (re.compile(r"old_string.*?not found", re.IGNORECASE), "string_not_found"),
    (re.compile(r"no.?match.*?for.?replacement", re.IGNORECASE), "no_match"),
    (re.compile(r"file.?not.?found|no.?such.?file", re.IGNORECASE), "file_not_found"),
    (re.compile(r"permission.?denied", re.IGNORECASE), "permission_denied"),
    (re.compile(r"is.?a.?directory", re.IGNORECASE), "is_directory"),
    (re.compile(r"syntax.?error.*?line\s*(\d+)", re.IGNORECASE), "syntax_error"),
    (re.compile(r"indentation.?error", re.IGNORECASE), "indent_error"),
]


def detect_edit_error(error_msg: str) -> RecoveryResult | None:
    """Detect file edit/patch errors."""
    for pattern, error_type in _EDIT_PATTERNS:
        match = pattern.search(error_msg)
        if match:
            guidance = _edit_guidance(error_type, error_msg)
            return RecoveryResult(
                action=RecoveryAction.RETRY,
                error_type=f"edit_{error_type}",
                message=guidance,
                severity=2,
            )
    return None


def _edit_guidance(error_type: str, error_msg: str) -> str:
    """Generate specific guidance for edit errors."""
    if error_type == "string_not_found":
        return (
            "The text you tried to replace was not found in the file. "
            "Read the file first to get the exact current content, then retry the edit."
        )
    if error_type == "file_not_found":
        return "File not found. Check the path and create it if needed."
    if error_type == "permission_denied":
        return "Permission denied. Check file permissions."
    if error_type == "syntax_error":
        return "Your edit introduced a syntax error. Read the file, find the error, and fix it."
    if error_type == "indent_error":
        return "Indentation error after edit. Check whitespace consistency."
    return f"Edit failed ({error_type}). Read the file and retry with correct content."


# --- Session Error Recovery ---

_SESSION_PATTERNS = [
    (re.compile(r"connection.?(?:refused|reset|timeout)", re.IGNORECASE), "connection"),
    (re.compile(r"rate.?limit", re.IGNORECASE), "rate_limit"),
    (re.compile(r"server.?error|500\s+internal", re.IGNORECASE), "server_error"),
    (re.compile(r"authentication.?(?:failed|error|invalid)", re.IGNORECASE), "auth_error"),
    (re.compile(r"api.?key.*?(?:invalid|expired|missing)", re.IGNORECASE), "api_key"),
]


def detect_session_error(error_msg: str) -> RecoveryResult | None:
    """Detect session/connection errors."""
    for pattern, error_type in _SESSION_PATTERNS:
        if pattern.search(error_msg):
            if error_type == "rate_limit":
                return RecoveryResult(
                    action=RecoveryAction.RETRY,
                    error_type="session_rate_limit",
                    message="Rate limited. Wait and retry.",
                    severity=1,
                )
            if error_type in ("auth_error", "api_key"):
                return RecoveryResult(
                    action=RecoveryAction.NONE,
                    error_type=f"session_{error_type}",
                    message="Authentication error. Check API keys.",
                    severity=3,
                )
            return RecoveryResult(
                action=RecoveryAction.RETRY,
                error_type=f"session_{error_type}",
                message=f"Session error ({error_type}). Retrying.",
                severity=2,
            )
    return None


# --- Main Recovery ---

def detect_and_recover(error_msg: str) -> RecoveryResult:
    """Detect error type and return recovery action.

    Priority: context window > edit error > session error.
    Returns RecoveryAction.NONE if error is not recoverable.
    """
    if not error_msg:
        return RecoveryResult(
            action=RecoveryAction.NONE,
            error_type="unknown",
            message="",
            severity=0,
        )

    # Priority 1: Context window (most critical)
    result = detect_context_error(error_msg)
    if result:
        return result

    # Priority 2: Edit errors (actionable)
    result = detect_edit_error(error_msg)
    if result:
        return result

    # Priority 3: Session errors (retryable)
    result = detect_session_error(error_msg)
    if result:
        return result

    # Unknown error
    return RecoveryResult(
        action=RecoveryAction.NONE,
        error_type="unknown",
        message="Unrecognized error. Check logs for details.",
        severity=1,
    )


def truncate_for_context(messages: list[dict], keep_recent: int = 10) -> list[dict]:
    """Truncate messages to recover from context window overflow.

    Keeps system prompt + last N messages.
    """
    if len(messages) <= keep_recent + 1:
        return messages

    system = [m for m in messages[:2] if m.get("role") == "system"]
    recent = messages[-keep_recent:]
    return system + recent
