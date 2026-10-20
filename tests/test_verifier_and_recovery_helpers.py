"""Tests for verifier pure helpers and recovery detection functions."""


from mullet.core.orchestration.verifier import (
    CriterionResult,
    VerificationResult,
    extract_criteria,
    _parse_criterion_result,
)
from mullet.core.agent.recovery import (
    RecoveryAction,
    RecoveryResult,
    detect_context_error,
    detect_edit_error,
    detect_session_error,
    detect_and_recover,
    truncate_for_context,
)


# ---------------------------------------------------------------------------
# CriterionResult
# ---------------------------------------------------------------------------

class TestCriterionResult:
    def test_criterion_stored(self):
        cr = CriterionResult(criterion="test passes")
        assert cr.criterion == "test passes"

    def test_passed_defaults_false(self):
        cr = CriterionResult(criterion="test passes")
        assert cr.passed is False

    def test_evidence_defaults_empty(self):
        cr = CriterionResult(criterion="test passes")
        assert cr.evidence == ""

    def test_attempt_defaults_one(self):
        cr = CriterionResult(criterion="test passes")
        assert cr.attempt == 1

    def test_custom_values(self):
        cr = CriterionResult(criterion="CI runs", passed=True, evidence="saw green", attempt=2)
        assert cr.passed is True
        assert cr.evidence == "saw green"
        assert cr.attempt == 2


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:
    def test_passed_defaults_false(self):
        vr = VerificationResult()
        assert vr.passed is False

    def test_criteria_results_defaults_empty(self):
        vr = VerificationResult()
        assert vr.criteria_results == {}

    def test_details_defaults_empty(self):
        vr = VerificationResult()
        assert vr.details == []

    def test_evidence_defaults_empty(self):
        vr = VerificationResult()
        assert vr.evidence == []

    def test_issues_defaults_empty(self):
        vr = VerificationResult()
        assert vr.issues == []

    def test_attempts_defaults_one(self):
        vr = VerificationResult()
        assert vr.attempts == 1

    def test_summary_returns_string(self):
        vr = VerificationResult()
        assert isinstance(vr.summary, str)

    def test_summary_pass_status(self):
        vr = VerificationResult(passed=True, criteria_results={"a": True}, attempts=1)
        assert "PASS" in vr.summary

    def test_summary_fail_status(self):
        vr = VerificationResult(passed=False, criteria_results={"a": False}, attempts=1)
        assert "FAIL" in vr.summary

    def test_summary_counts_passed(self):
        vr = VerificationResult(
            passed=False,
            criteria_results={"a": True, "b": False, "c": True},
            attempts=1,
        )
        assert "2/3" in vr.summary

    def test_summary_includes_attempt_count(self):
        vr = VerificationResult(passed=True, criteria_results={}, attempts=3)
        assert "3" in vr.summary

    def test_failed_criteria_returns_list(self):
        vr = VerificationResult()
        assert isinstance(vr.failed_criteria, list)

    def test_failed_criteria_correct_keys(self):
        vr = VerificationResult(criteria_results={"a": True, "b": False, "c": False})
        assert set(vr.failed_criteria) == {"b", "c"}

    def test_failed_criteria_empty_when_all_pass(self):
        vr = VerificationResult(criteria_results={"a": True, "b": True})
        assert vr.failed_criteria == []


# ---------------------------------------------------------------------------
# extract_criteria
# ---------------------------------------------------------------------------

class TestExtractCriteria:
    def test_empty_text_returns_empty(self):
        assert extract_criteria("") == []

    def test_no_patterns_returns_empty(self):
        assert extract_criteria("just prose text with no criteria markers") == []

    def test_checkbox_unchecked(self):
        result = extract_criteria("- [ ] login works")
        assert "login works" in result

    def test_checkbox_checked(self):
        result = extract_criteria("- [x] tests pass")
        assert "tests pass" in result

    def test_numbered_item(self):
        result = extract_criteria("1. API returns 200")
        assert "API returns 200" in result

    def test_numbered_item_period(self):
        result = extract_criteria("2. Data is persisted")
        assert "Data is persisted" in result

    def test_ac_prefix(self):
        result = extract_criteria("AC: response within 500ms")
        assert "response within 500ms" in result

    def test_acceptance_criteria_prefix(self):
        result = extract_criteria("Acceptance Criteria: unit tests pass")
        assert "unit tests pass" in result

    def test_bullet_long_enough_included(self):
        result = extract_criteria("- this is long enough criterion")
        assert len(result) >= 1

    def test_bullet_too_short_excluded(self):
        result = extract_criteria("- hi")
        assert result == []

    def test_multiple_criteria_all_extracted(self):
        text = "- [ ] login works\n1. API 200\nAC: fast response"
        result = extract_criteria(text)
        assert len(result) == 3

    def test_returns_list(self):
        assert isinstance(extract_criteria("- [ ] works"), list)


# ---------------------------------------------------------------------------
# _parse_criterion_result
# ---------------------------------------------------------------------------

class TestParseCriterionResult:
    def test_pass_detected(self):
        response = "CRITERION 1: PASS - file found"
        passed, _ = _parse_criterion_result(response, 1)
        assert passed is True

    def test_fail_detected(self):
        response = "CRITERION 1: FAIL - file missing"
        passed, _ = _parse_criterion_result(response, 1)
        assert passed is False

    def test_evidence_returned(self):
        response = "CRITERION 1: PASS - found the function"
        _, evidence = _parse_criterion_result(response, 1)
        assert isinstance(evidence, str)
        assert len(evidence) > 0

    def test_wrong_criterion_index_not_found(self):
        response = "CRITERION 2: PASS - ok"
        passed, evidence = _parse_criterion_result(response, 1)
        assert passed is False
        assert evidence == "Not explicitly verified"

    def test_all_pass_global_fallback(self):
        response = "All checks complete. ALL PASS"
        passed, _ = _parse_criterion_result(response, 99)
        assert passed is True

    def test_returns_tuple(self):
        result = _parse_criterion_result("nothing here", 1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_case_insensitive_matching(self):
        response = "criterion 1: pass - all good"
        passed, _ = _parse_criterion_result(response, 1)
        assert passed is True


# ---------------------------------------------------------------------------
# detect_context_error
# ---------------------------------------------------------------------------

class TestDetectContextError:
    def test_no_match_returns_none(self):
        assert detect_context_error("some random error") is None

    def test_context_length_exceeded(self):
        r = detect_context_error("context length exceeded")
        assert r is not None
        assert r.action == RecoveryAction.TRUNCATE

    def test_too_many_tokens(self):
        r = detect_context_error("too many tokens in the prompt")
        assert r is not None
        assert r.action == RecoveryAction.TRUNCATE

    def test_token_limit_exceeded(self):
        r = detect_context_error("token limit exceeded")
        assert r is not None

    def test_input_too_long(self):
        r = detect_context_error("input is too long for this model")
        assert r is not None

    def test_returns_recovery_result(self):
        r = detect_context_error("context length exceeded")
        assert isinstance(r, RecoveryResult)

    def test_severity_high_for_context(self):
        r = detect_context_error("context length exceeded")
        assert r.severity >= 2

    def test_error_type_includes_context(self):
        r = detect_context_error("context length exceeded")
        assert "context" in r.error_type


# ---------------------------------------------------------------------------
# detect_edit_error
# ---------------------------------------------------------------------------

class TestDetectEditError:
    def test_no_match_returns_none(self):
        assert detect_edit_error("unrelated error") is None

    def test_string_not_found(self):
        r = detect_edit_error("old_string not found in file")
        assert r is not None
        assert r.action == RecoveryAction.RETRY

    def test_file_not_found(self):
        r = detect_edit_error("file not found: src/foo.py")
        assert r is not None
        assert r.action == RecoveryAction.RETRY

    def test_syntax_error(self):
        r = detect_edit_error("syntax error at line 15")
        assert r is not None

    def test_permission_denied(self):
        r = detect_edit_error("permission denied: /etc/hosts")
        assert r is not None

    def test_returns_recovery_result(self):
        r = detect_edit_error("old_string not found in file")
        assert isinstance(r, RecoveryResult)

    def test_error_type_starts_with_edit(self):
        r = detect_edit_error("old_string not found in file")
        assert r.error_type.startswith("edit_")

    def test_message_is_informative(self):
        r = detect_edit_error("old_string not found in file")
        assert len(r.message) > 10


# ---------------------------------------------------------------------------
# detect_session_error
# ---------------------------------------------------------------------------

class TestDetectSessionError:
    def test_no_match_returns_none(self):
        assert detect_session_error("something unrelated") is None

    def test_rate_limit_detected(self):
        r = detect_session_error("rate limit exceeded")
        assert r is not None
        assert r.action == RecoveryAction.RETRY

    def test_rate_limit_low_severity(self):
        r = detect_session_error("rate limit exceeded")
        assert r.severity == 1

    def test_auth_failed_no_retry(self):
        r = detect_session_error("authentication failed")
        assert r is not None
        assert r.action == RecoveryAction.NONE

    def test_auth_failed_high_severity(self):
        r = detect_session_error("authentication failed")
        assert r.severity == 3

    def test_api_key_invalid(self):
        r = detect_session_error("api key invalid")
        assert r is not None
        assert r.action == RecoveryAction.NONE

    def test_connection_refused(self):
        r = detect_session_error("connection refused")
        assert r is not None
        assert r.action == RecoveryAction.RETRY

    def test_returns_recovery_result(self):
        r = detect_session_error("rate limit exceeded")
        assert isinstance(r, RecoveryResult)


# ---------------------------------------------------------------------------
# detect_and_recover
# ---------------------------------------------------------------------------

class TestDetectAndRecover:
    def test_empty_string_returns_none_action(self):
        r = detect_and_recover("")
        assert r.action == RecoveryAction.NONE

    def test_empty_string_severity_zero(self):
        r = detect_and_recover("")
        assert r.severity == 0

    def test_context_error_takes_priority(self):
        r = detect_and_recover("context length exceeded")
        assert "context" in r.error_type

    def test_edit_error_detected(self):
        r = detect_and_recover("old_string not found in file")
        assert "edit" in r.error_type

    def test_session_error_detected(self):
        r = detect_and_recover("rate limit exceeded")
        assert "session" in r.error_type

    def test_unknown_error_returns_none_action(self):
        r = detect_and_recover("completely unknown xyz error type")
        assert r.action == RecoveryAction.NONE
        assert r.error_type == "unknown"

    def test_always_returns_recovery_result(self):
        r = detect_and_recover("anything")
        assert isinstance(r, RecoveryResult)

    def test_context_wins_over_session(self):
        # A message that could match both context and session patterns
        r = detect_and_recover("context length exceeded, connection also failed")
        assert "context" in r.error_type


# ---------------------------------------------------------------------------
# truncate_for_context
# ---------------------------------------------------------------------------

class TestTruncateForContext:
    def _msgs(self, n):
        return [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg {i}"} for i in range(n)
        ]

    def test_short_list_unchanged(self):
        msgs = self._msgs(3)
        result = truncate_for_context(msgs, keep_recent=10)
        assert result == msgs

    def test_long_list_truncated(self):
        msgs = self._msgs(20)
        result = truncate_for_context(msgs, keep_recent=5)
        assert len(result) < len(msgs)

    def test_system_message_preserved(self):
        msgs = self._msgs(20)
        result = truncate_for_context(msgs, keep_recent=5)
        assert result[0]["role"] == "system"

    def test_most_recent_messages_kept(self):
        msgs = self._msgs(20)
        result = truncate_for_context(msgs, keep_recent=5)
        assert result[-1]["content"] == "msg 19"

    def test_result_length_correct(self):
        msgs = self._msgs(20)
        result = truncate_for_context(msgs, keep_recent=5)
        assert len(result) == 6  # 1 system + 5 recent

    def test_returns_list(self):
        msgs = self._msgs(5)
        assert isinstance(truncate_for_context(msgs), list)

    def test_empty_list_returned_unchanged(self):
        result = truncate_for_context([], keep_recent=5)
        assert result == []
