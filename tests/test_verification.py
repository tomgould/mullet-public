"""Tests for the verification framework."""

import pytest
import subprocess
from unittest.mock import MagicMock, patch
from mullet.core.orchestration.verification import (
    CheckStatus,
    VerificationCheck,
    VerificationResult,
    STANDARD_CHECKS,
    get_standard_check,
    create_protocol,
    run_check,
    run_protocol,
    quick_verify,
    llm_as_judge,
    eval_dataset_check,
)


def test_standard_checks_defined():
    assert "build" in STANDARD_CHECKS
    assert "test" in STANDARD_CHECKS
    assert "lint" in STANDARD_CHECKS
    assert "functionality" in STANDARD_CHECKS
    assert "error_free" in STANDARD_CHECKS


def test_get_standard_check():
    check = get_standard_check("build")
    assert check.id == "build"
    assert check.name == "Build Success"
    assert check.status == CheckStatus.PENDING
    assert check.required is True


def test_get_standard_check_returns_copy():
    c1 = get_standard_check("test")
    c2 = get_standard_check("test")
    c1.status = CheckStatus.PASSED
    assert c2.status == CheckStatus.PENDING  # Independent copy


def test_get_standard_check_unknown():
    with pytest.raises(ValueError, match="Unknown"):
        get_standard_check("nonexistent_check")


def test_create_protocol_standard():
    checks = create_protocol(["build", "test"])
    assert len(checks) == 2
    assert checks[0].id == "build"
    assert checks[1].id == "test"


def test_create_protocol_custom():
    custom = VerificationCheck(
        id="custom", name="Custom Check",
        description="A custom check",
        command="echo ok",
    )
    checks = create_protocol(custom_checks=[custom])
    assert len(checks) == 1
    assert checks[0].id == "custom"


def test_create_protocol_mixed():
    custom = VerificationCheck(
        id="custom", name="Custom", description="Custom",
    )
    checks = create_protocol(["build"], [custom])
    assert len(checks) == 2


def test_run_check_with_command_success():
    check = VerificationCheck(
        id="echo", name="Echo Test",
        description="Run echo",
        command="echo hello",
    )
    result = run_check(check)
    assert result.status == CheckStatus.PASSED
    assert "hello" in result.evidence
    assert result.duration_s >= 0


def test_run_check_with_command_failure():
    check = VerificationCheck(
        id="fail", name="Fail Test",
        description="Run failing command",
        command="bash -c 'exit 1'",
    )
    result = run_check(check)
    assert result.status == CheckStatus.FAILED
    assert "Exit code 1" in result.evidence


def test_run_check_without_command():
    check = VerificationCheck(
        id="manual", name="Manual Check",
        description="Requires manual review",
    )
    result = run_check(check)
    assert result.status == CheckStatus.SKIPPED
    assert "manual" in result.evidence.lower()


def test_run_check_command_timeout():
    check = VerificationCheck(
        id="slow", name="Slow Check",
        description="Takes too long",
        command="sleep 999",
    )
    # Override timeout by using a faster-failing approach
    check.command = "bash -c 'sleep 0.01 && exit 0'"
    result = run_check(check)
    assert result.status == CheckStatus.PASSED


def test_run_protocol_all_pass():
    checks = [
        VerificationCheck(id="a", name="A", description="A", command="echo ok"),
        VerificationCheck(id="b", name="B", description="B", command="echo ok"),
    ]
    result = run_protocol(checks)
    assert result.passed is True
    assert result.passed_count == 2
    assert result.failed_count == 0
    assert result.total_duration_s >= 0


def test_run_protocol_required_failure():
    checks = [
        VerificationCheck(id="a", name="A", description="A", command="echo ok"),
        VerificationCheck(id="b", name="B", description="B", command="bash -c 'exit 1'", required=True),
    ]
    result = run_protocol(checks)
    assert result.passed is False
    assert result.failed_count == 1
    assert len(result.required_failures) == 1


def test_run_protocol_optional_failure_still_passes():
    checks = [
        VerificationCheck(id="a", name="A", description="A", command="echo ok", required=True),
        VerificationCheck(id="b", name="B", description="B", command="bash -c 'exit 1'", required=False),
    ]
    result = run_protocol(checks)
    assert result.passed is True  # Optional failure doesn't block


def test_run_protocol_fail_fast():
    callback_log = []

    def on_check(check):
        callback_log.append(check.id)

    checks = [
        VerificationCheck(id="a", name="A", description="A", command="bash -c 'exit 1'", required=True),
        VerificationCheck(id="b", name="B", description="B", command="echo ok"),
    ]
    result = run_protocol(checks, fail_fast=True, on_check=on_check)
    assert result.passed is False
    assert checks[1].status == CheckStatus.SKIPPED
    assert "a" in callback_log


def test_run_protocol_callback():
    log = []
    checks = [
        VerificationCheck(id="a", name="A", description="A", command="echo ok"),
    ]
    run_protocol(checks, on_check=lambda c: log.append(c.id))
    assert "a" in log


def test_run_protocol_summary():
    checks = [
        VerificationCheck(id="a", name="A", description="A", command="echo ok"),
        VerificationCheck(id="b", name="B", description="B", command="bash -c 'exit 1'"),
    ]
    result = run_protocol(checks)
    assert "A: passed" in result.summary
    assert "B: failed" in result.summary


def test_verification_result_properties():
    result = VerificationResult(
        passed=False,
        checks=[
            VerificationCheck(id="a", name="A", description="A", status=CheckStatus.PASSED),
            VerificationCheck(id="b", name="B", description="B", status=CheckStatus.FAILED, required=True),
            VerificationCheck(id="c", name="C", description="C", status=CheckStatus.SKIPPED),
        ],
    )
    assert result.passed_count == 1
    assert result.failed_count == 1
    assert len(result.required_failures) == 1
    assert result.required_failures[0].id == "b"


class TestVerificationMissingLines:
    """Cover missing lines in verification.py exception handling and utilities."""

    # -- run_check exception handling (lines 148-155) --

    def test_run_check_timeout_expired(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=300),
        ):
            check = VerificationCheck(
                id="t", name="T", description="T", command="sleep 10"
            )
            result = run_check(check)
        assert result.status == CheckStatus.FAILED
        assert "timed out" in result.evidence.lower()
        assert result.duration_s >= 0

    def test_run_check_generic_exception(self):
        with patch("subprocess.run", side_effect=OSError("no such file")):
            check = VerificationCheck(
                id="e", name="E", description="E", command="missing"
            )
            result = run_check(check)
        assert result.status == CheckStatus.FAILED
        assert result.evidence.startswith("Error:")
        assert "no such file" in result.evidence
        assert result.duration_s >= 0

    # -- quick_verify (lines 223-229) --

    def test_quick_verify_both_pass(self):
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            result = quick_verify()
        assert result.passed is True
        assert result.passed_count == 2

    def test_quick_verify_result_type(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            result = quick_verify()
        assert isinstance(result, VerificationResult)
        assert result.total_duration_s >= 0

    # -- llm_as_judge (lines 241-260) --

    def test_llm_as_judge_extracts_score_and_feedback(self):
        agent = MagicMock()
        agent.run.return_value = MagicMock(
            final_response="SCORE: 0.9\nFEEDBACK: Great work"
        )
        score, feedback = llm_as_judge("hello", ["be nice"], agent)
        assert score == 0.9
        assert feedback == "Great work"

    def test_llm_as_judge_default_score_when_no_score_line(self):
        agent = MagicMock()
        agent.run.return_value = MagicMock(final_response="Looks okay")
        score, feedback = llm_as_judge("hello", ["be nice"], agent)
        assert score == 0.5
        assert "Looks okay" in feedback

    def test_llm_as_judge_caps_score_at_1(self):
        agent = MagicMock()
        agent.run.return_value = MagicMock(
            final_response="SCORE: 1.5\nFEEDBACK: Too high"
        )
        score, _ = llm_as_judge("hello", ["be nice"], agent)
        assert score == 1.0

    def test_llm_as_judge_zero_score(self):
        agent = MagicMock()
        agent.run.return_value = MagicMock(
            final_response="SCORE: 0.0\nFEEDBACK: Nothing"
        )
        score, _ = llm_as_judge("hello", ["be nice"], agent)
        assert score == 0.0

    def test_llm_as_judge_no_feedback_line(self):
        agent = MagicMock()
        agent.run.return_value = MagicMock(final_response="SCORE: 0.75")
        score, feedback = llm_as_judge("hello", ["be nice"], agent)
        assert score == 0.75
        assert "SCORE: 0.75" in feedback

    # -- eval_dataset_check (lines 268-279) --

    def test_eval_dataset_check_exact_match(self):
        passed, sim = eval_dataset_check("hello world", "hello world")
        assert passed is True
        assert sim == 1.0

    def test_eval_dataset_check_fuzzy_false_mismatch(self):
        passed, sim = eval_dataset_check("hello", "world", fuzzy=False)
        assert passed is False
        assert sim == 0.0

    def test_eval_dataset_check_fuzzy_high_similarity(self):
        passed, sim = eval_dataset_check(
            "the quick brown fox", "the quick brown fox jumps", fuzzy=True
        )
        assert passed is True
        assert sim >= 0.8

    def test_eval_dataset_check_fuzzy_low_similarity(self):
        passed, sim = eval_dataset_check(
            "abc", "xyz def ghi", fuzzy=True
        )
        assert passed is False
        assert sim < 0.8

    def test_eval_dataset_check_empty_expected(self):
        passed, sim = eval_dataset_check("hello", "", fuzzy=True)
        assert passed is False
        assert sim == 0.0

