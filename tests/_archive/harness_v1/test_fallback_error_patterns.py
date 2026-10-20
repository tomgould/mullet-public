"""Unit tests for mullet.harness.fallback.error_patterns."""

from __future__ import annotations

import pytest

from mullet.harness.fallback.error_patterns import (
    classify_error,
    is_retryable_error,
)

pytestmark = pytest.mark.harness


class TestClassifyError:
    def test_classify_rate_limit(self):
        assert classify_error("Rate limit exceeded") == "rate_limit"
        assert classify_error("HTTP 429 too many requests") == "rate_limit"
        assert classify_error("too many requests") == "rate_limit"

    def test_classify_quota(self):
        assert classify_error("Quota will reset at midnight") == "quota"
        assert (
            classify_error("Exhausted capacity for model kimi-k2.6") == "quota"
        )
        assert (
            classify_error(
                "All credentials for model kimi-k2.6 are exhausted"
            )
            == "quota"
        )

    def test_classify_cooldown(self):
        assert classify_error("Model is cooling down, retry later") == "cooldown"
        assert classify_error("cooldown active") == "cooldown"

    def test_classify_model_not_found(self):
        assert classify_error("Model not found: foo-bar") == "model_not_found"
        assert (
            classify_error("This model is not supported by upstream")
            == "model_not_found"
        )
        assert classify_error("HTTP 404 Not Found") == "model_not_found"

    def test_classify_service_unavailable(self):
        assert (
            classify_error("Service unavailable") == "service_unavailable"
        )
        assert classify_error("Server overloaded") == "service_unavailable"
        assert (
            classify_error("temporarily unavailable") == "service_unavailable"
        )
        assert classify_error("Got HTTP 503 from upstream") == "service_unavailable"
        assert classify_error("Got HTTP 502 bad gateway") == "service_unavailable"
        assert classify_error("Got HTTP 504 gateway timeout") == "service_unavailable"

    def test_classify_auth(self):
        assert classify_error("Unauthorized") == "auth"
        assert classify_error("Invalid API key supplied") == "auth"
        assert classify_error("Missing credentials in request") == "auth"
        assert classify_error("HTTP 401") == "auth"
        assert classify_error("HTTP 403 forbidden") == "auth"

    def test_classify_returns_none_for_unrelated_error(self):
        assert classify_error("Something completely unexpected") is None
        assert classify_error("") is None
        assert classify_error("Hello world") is None


class TestIsRetryableError:
    def test_is_retryable_true_for_classified_categories(self):
        assert is_retryable_error("Rate limit exceeded") is True
        assert is_retryable_error("Quota will reset at midnight") is True
        assert is_retryable_error("cooldown active") is True
        assert is_retryable_error("service unavailable") is True

    def test_is_retryable_with_retry_codes(self):
        # Auth is *not* in the retryable set, but if 401 is in retry_codes
        # the caller has opted into retrying it.
        assert (
            is_retryable_error("HTTP 401 unauthorized", retry_codes=[401])
            is True
        )
        # Status appears in the message and is in retry_codes.
        assert (
            is_retryable_error(
                "Some weird error code 418 returned", retry_codes=[418]
            )
            is True
        )

    def test_is_retryable_false_for_unrelated(self):
        assert is_retryable_error("Something unexpected") is False
        assert (
            is_retryable_error("Something unexpected", retry_codes=[599])
            is False
        )

    def test_is_retryable_false_for_auth_without_opt_in(self):
        # Auth alone is NOT retryable (a key being wrong won't fix itself).
        assert is_retryable_error("Invalid API key supplied") is False

    def test_is_retryable_false_for_model_not_found_without_opt_in(self):
        assert is_retryable_error("Model not found: foo-bar") is False
