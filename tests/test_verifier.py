"""Tests for the verification protocol."""

from mullet.core.orchestration.verifier import (
    VerificationResult, CriterionResult, verify_criteria,
    extract_criteria, _parse_criterion_result,
)
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class MockVerifyProvider(Provider):
    name = "test"
    def __init__(self, response: str):
        self._response = response
    def create(self, messages, **kw):
        return LLMResponse(content=self._response, usage=TokenUsage(total_tokens=100))


def test_verification_all_pass():
    agent = AgentRunner(
        provider=MockVerifyProvider(
            "CRITERION 1: PASS - file exists\n"
            "CRITERION 2: PASS - function returns correct value\n"
            "ALL CRITERIA PASS"
        ),
        tool_registry=ToolRegistry(),
        max_iterations=5,
    )
    result = verify_criteria(
        criteria=["file exists", "function returns correct value"],
        context="testing",
        agent=agent,
    )
    assert result.passed is True
    assert len(result.criteria_results) == 2
    assert all(result.criteria_results.values())


def test_verification_partial_fail():
    agent = AgentRunner(
        provider=MockVerifyProvider(
            "CRITERION 1: PASS - works\n"
            "CRITERION 2: FAIL - missing function\n"
        ),
        tool_registry=ToolRegistry(),
        max_iterations=5,
    )
    result = verify_criteria(
        criteria=["component works", "function exists"],
        context="testing",
        agent=agent,
    )
    assert result.passed is False
    assert len(result.issues) > 0


def test_verification_result_summary():
    result = VerificationResult(
        passed=True,
        criteria_results={"a": True, "b": True, "c": True},
    )
    assert "PASS" in result.summary
    assert "3/3" in result.summary


def test_verification_result_fail_summary():
    result = VerificationResult(
        passed=False,
        criteria_results={"a": True, "b": False},
        issues=["b failed"],
    )
    assert "FAIL" in result.summary
    assert "1/2" in result.summary


# --- Extract criteria tests ---

def test_extract_criteria_checkboxes():
    text = "- [ ] User can log in\n- [x] User can register\n- [ ] Password reset works"
    criteria = extract_criteria(text)
    assert len(criteria) == 3
    assert "User can log in" in criteria
    assert "Password reset works" in criteria


def test_extract_criteria_numbered():
    text = "1. File is created\n2. Tests pass\n3. No regressions"
    criteria = extract_criteria(text)
    assert len(criteria) == 3
    assert "File is created" in criteria


def test_extract_criteria_ac_prefix():
    text = "AC: Login form validates email\nAC: Error message shows on failure"
    criteria = extract_criteria(text)
    assert len(criteria) == 2
    assert "Login form validates email" in criteria


def test_extract_criteria_bullets():
    text = "- Users can search for products\n- Cart persists across sessions\n- Checkout flow completes"
    criteria = extract_criteria(text)
    assert len(criteria) == 3


def test_extract_criteria_empty():
    assert extract_criteria("") == []
    assert extract_criteria("no criteria here") == []


# --- Per-criterion parsing ---

def test_parse_criterion_pass():
    response = "CRITERION 1: PASS - the file exists at src/main.py"
    passed, evidence = _parse_criterion_result(response, 1)
    assert passed is True
    assert "PASS" in evidence


def test_parse_criterion_fail():
    response = "CRITERION 2: FAIL - function not found"
    passed, evidence = _parse_criterion_result(response, 2)
    assert passed is False
    assert "FAIL" in evidence.upper()


def test_parse_criterion_global_all_pass():
    response = "Everything looks good. ALL CRITERIA PASS."
    passed, evidence = _parse_criterion_result(response, 1)
    assert passed is True


def test_parse_criterion_not_found():
    response = "I checked things but didn't format the output"
    passed, evidence = _parse_criterion_result(response, 1)
    assert passed is False


# --- Structured results ---

def test_verification_result_failed_criteria():
    result = VerificationResult(
        passed=False,
        criteria_results={"a": True, "b": False, "c": False},
    )
    failed = result.failed_criteria
    assert "b" in failed
    assert "c" in failed
    assert "a" not in failed


def test_criterion_result_fields():
    cr = CriterionResult(criterion="test passes", passed=True, evidence="all green", attempt=2)
    assert cr.criterion == "test passes"
    assert cr.passed is True
    assert cr.attempt == 2


def test_verification_with_retry():
    """Verify that failed criteria are retried."""
    call_count = [0]

    class RetryProvider(Provider):
        name = "test"
        def create(self, messages, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return LLMResponse(
                    content="CRITERION 1: PASS - ok\nCRITERION 2: FAIL - missing",
                    usage=TokenUsage(total_tokens=100),
                )
            else:
                return LLMResponse(
                    content="CRITERION 2: PASS - fixed now",
                    usage=TokenUsage(total_tokens=50),
                )

    agent = AgentRunner(
        provider=RetryProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=5,
    )
    result = verify_criteria(
        criteria=["thing one works", "thing two works"],
        context="testing retry",
        agent=agent,
        max_retries=2,
    )
    assert result.passed is True
    assert call_count[0] == 2  # Called twice (retry)
