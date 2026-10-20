"""Tests for the consolidated error classification (C3).

`retry.is_retryable` now delegates to `health.classify_error`. These tests
pin both the delegation contract and the behavioural parity with the old
keyword/status-code-based implementation.
"""

from unittest.mock import patch

from mullet.core.providers import retry
from mullet.core.providers.health import classify_error
from mullet.core.providers.retry import RETRYABLE_CATEGORIES, is_retryable


# ── Delegation contract ─────────────────────────────────────────────────────


def test_is_retryable_delegates_to_classify_error():
    """is_retryable must call classify_error — not duplicate its logic.

    Failure mode if broken: retry.py re-implements classification and drifts
    from health.py's rules.
    """
    err = Exception("arbitrary")
    with patch("mullet.core.providers.retry.classify_error") as mock_cls:
        mock_cls.return_value = "rate_limit"
        assert is_retryable(err) is True
        mock_cls.assert_called_once_with(err)

    with patch("mullet.core.providers.retry.classify_error") as mock_cls:
        mock_cls.return_value = "auth"
        assert is_retryable(err) is False
        mock_cls.assert_called_once_with(err)


def test_retryable_categories_are_subset_of_known_categories():
    """RETRYABLE_CATEGORIES must only contain values classify_error actually emits.

    Failure mode if broken: a typo in the set silently makes nothing retryable.
    """
    known = {"rate_limit", "overloaded", "server_error", "connection", "auth", "unknown"}
    assert known >= RETRYABLE_CATEGORIES
    assert "auth" not in RETRYABLE_CATEGORIES
    assert "unknown" not in RETRYABLE_CATEGORIES


# ── Behavioural parity with old is_retryable ───────────────────────────────


def test_is_retryable_behaviour_preserved_for_timeout():
    """Built-in TimeoutError must remain retryable.

    Failure mode if broken: transient timeouts fail fast and don't retry.
    """
    assert is_retryable(TimeoutError("timed out")) is True


def test_is_retryable_behaviour_preserved_for_auth_error():
    """401-style errors must NOT retry (auth is non-transient).

    Failure mode if broken: we spam a bad API key through exponential backoff.
    """
    assert is_retryable(Exception("401 Unauthorized")) is False
    assert is_retryable(Exception("403 Forbidden")) is False


def test_is_retryable_behaviour_preserved_for_rate_limit():
    """429 and 'rate limit' messages must remain retryable.

    Failure mode if broken: we give up on transient rate limits instead of backing off.
    """
    assert is_retryable(Exception("429 Too Many Requests")) is True
    assert is_retryable(Exception("rate limit exceeded")) is True
    assert is_retryable(Exception("quota exceeded")) is True
    assert is_retryable(Exception("resource exhausted")) is True


def test_is_retryable_behaviour_preserved_for_server_errors():
    """5xx errors must remain retryable (including 502/504).

    Failure mode if broken: bad gateway / gateway timeout stop retrying.
    """
    assert is_retryable(Exception("500 Internal Server Error")) is True
    assert is_retryable(Exception("502 Bad Gateway")) is True
    assert is_retryable(Exception("504 Gateway Timeout")) is True
    assert is_retryable(Exception("internal error")) is True


def test_is_retryable_behaviour_preserved_for_overloaded():
    """503 / overloaded / try-again must remain retryable.

    Failure mode if broken: transient capacity errors surface instead of retrying.
    """
    assert is_retryable(Exception("Server overloaded, try again")) is True
    assert is_retryable(Exception("503 Service Unavailable")) is True
    assert is_retryable(Exception("over capacity")) is True


def test_is_retryable_behaviour_preserved_for_connection():
    """Network-ish errors must remain retryable.

    Failure mode if broken: flaky networks blow up on first glitch.
    """
    assert is_retryable(Exception("Connection timeout")) is True
    assert is_retryable(ConnectionError("connection refused")) is True
    assert is_retryable(Exception("connection reset by peer")) is True

    class NetworkError(Exception):
        pass

    assert is_retryable(NetworkError("some network glitch")) is True


def test_is_retryable_returns_false_for_unknown_error():
    """Plain ValueError / bad-request must NOT be retried.

    Failure mode if broken: client bugs masquerade as transient errors.
    """
    assert is_retryable(ValueError("bad input")) is False
    assert is_retryable(Exception("400 Bad Request: invalid JSON")) is False


# ── Regression: classify_error still works standalone ──────────────────────


def test_classify_error_still_works_standalone():
    """classify_error must keep emitting the original category strings.

    Failure mode if broken: health.record_failure gets garbage category and
    BACKOFF_DURATIONS lookup falls through to 'unknown' for every error.
    """
    assert classify_error(Exception("HTTP 429 Too Many Requests")) == "rate_limit"
    assert classify_error(Exception("HTTP 503 Service Unavailable")) == "overloaded"
    assert classify_error(Exception("500 Internal Server Error")) == "server_error"
    assert classify_error(Exception("401 Unauthorized")) == "auth"
    assert classify_error(ConnectionError("refused")) == "connection"
    assert classify_error(TimeoutError("timed out")) == "connection"
    assert classify_error(Exception("something random")) == "unknown"


def test_retry_module_exposes_classify_error_binding():
    """retry.py must import classify_error from health (so patching works).

    Failure mode if broken: downstream patches target the wrong module and
    silently no-op.
    """
    assert retry.classify_error is classify_error
