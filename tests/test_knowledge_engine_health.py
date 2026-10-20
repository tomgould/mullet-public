"""
Tests for mullet.knowledge.engine.health — CircuitBreaker, check_lm_studio_model,
and wait_for_lm_studio.  Achieves 100 % branch coverage.
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from mullet.knowledge.engine.health import (
    CircuitBreaker,
    check_lm_studio_model,
    wait_for_lm_studio,
)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_is_closed_and_is_open_when_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.is_closed() is True
        assert cb.is_open() is False

    def test_is_closed_and_is_open_when_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_closed() is False
        assert cb.is_open() is True

    def test_allow_request_when_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.allow_request() is True

    def test_allow_request_when_open_cooldown_not_expired(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure()          # trips OPEN
        assert cb.is_open() is True
        assert cb.allow_request() is False
        assert cb._total_rejected == 1

    def test_allow_request_when_open_cooldown_expired_probe_allowed(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=1.0)
        cb.record_failure()          # trips OPEN
        # Simulate cooldown elapsed
        cb._opened_at = time.time() - 2.0
        assert cb.allow_request() is True
        assert cb._probe_in_flight is True

    def test_allow_request_when_open_cooldown_expired_probe_in_flight(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=1.0)
        cb.record_failure()
        cb._opened_at = time.time() - 2.0
        cb._probe_in_flight = True
        assert cb.allow_request() is False
        assert cb._total_rejected == 1

    def test_record_success_no_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_success()          # no log, no crash
        assert cb._consecutive_failures == 0
        assert cb._opened_at is None
        assert cb._probe_in_flight is False

    def test_record_success_with_failures_resets(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb._consecutive_failures = 2
        cb._total_rejected = 5
        cb._opened_at = time.time()
        cb._probe_in_flight = True
        cb.record_success()
        assert cb._consecutive_failures == 0
        assert cb._opened_at is None
        assert cb._probe_in_flight is False

    def test_record_failure_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        assert cb._consecutive_failures == 1
        assert cb._opened_at is None

    def test_record_failure_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0)
        cb.record_failure()
        assert cb._consecutive_failures == 1
        assert cb._opened_at is not None

    def test_record_failure_above_threshold_does_not_reset_opened_at(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0)
        cb.record_failure()
        opened = cb._opened_at
        cb.record_failure()
        assert cb._opened_at == opened

    def test_stats_property(self):
        cb = CircuitBreaker(failure_threshold=2)
        assert cb.stats == {
            "state": "closed",
            "consecutive_failures": 0,
            "total_rejected": 0,
        }
        cb.record_failure()
        assert cb.stats == {
            "state": "closed",
            "consecutive_failures": 1,
            "total_rejected": 0,
        }
        cb.record_failure()
        assert cb.stats == {
            "state": "open",
            "consecutive_failures": 2,
            "total_rejected": 0,
        }


# ---------------------------------------------------------------------------
# check_lm_studio_model
# ---------------------------------------------------------------------------

class TestCheckLmStudioModel:
    def _make_client(self, status_code, json_data):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data
        mock_client.post.return_value = mock_response
        return mock_client

    @patch("httpx.Client")
    def test_200_with_valid_content(self, mock_client_cls):
        client = self._make_client(200, {
            "choices": [{"message": {"content": "OK"}}]
        })
        mock_client_cls.return_value.__enter__.return_value = client
        assert check_lm_studio_model() is True

    @patch("httpx.Client")
    def test_200_with_empty_choices(self, mock_client_cls):
        client = self._make_client(200, {"choices": []})
        mock_client_cls.return_value.__enter__.return_value = client
        assert check_lm_studio_model() is False

    @patch("httpx.Client")
    def test_non_200_response(self, mock_client_cls):
        client = self._make_client(503, {})
        mock_client_cls.return_value.__enter__.return_value = client
        assert check_lm_studio_model() is False

    @patch("httpx.Client")
    def test_exception_raised(self, mock_client_cls):
        mock_client_cls.side_effect = Exception("boom")
        assert check_lm_studio_model() is False


# ---------------------------------------------------------------------------
# wait_for_lm_studio
# ---------------------------------------------------------------------------

class TestWaitForLmStudio:
    @patch("mullet.knowledge.engine.health.check_lm_studio_model", return_value=True)
    def test_first_check_succeeds(self, mock_check):
        assert wait_for_lm_studio() is True
        mock_check.assert_called_once()

    @patch("mullet.knowledge.engine.health.check_lm_studio_model", return_value=False)
    def test_times_out(self, mock_check):
        assert wait_for_lm_studio(timeout=0.01, poll_interval=0.001) is False
        assert mock_check.call_count >= 1
