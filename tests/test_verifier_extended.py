"""Extended tests for the verification protocol."""

from mullet.core.orchestration.verifier import VerificationResult, verify_criteria, verify_tests
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class VerifyProvider(Provider):
    name = "test"
    def __init__(self, response: str):
        self._resp = response
    def create(self, messages, **kw):
        return LLMResponse(content=self._resp, usage=TokenUsage(total_tokens=100))


def test_all_criteria_pass():
    agent = AgentRunner(
        provider=VerifyProvider("CRITERION 1: PASS - OK\nCRITERION 2: PASS - OK\nALL CRITERIA PASS"),
        tool_registry=ToolRegistry(), max_iterations=5,
    )
    result = verify_criteria(["criterion A", "criterion B"], "context", agent)
    assert result.passed is True
    assert result.summary.startswith("PASS")


def test_mixed_pass_fail():
    agent = AgentRunner(
        provider=VerifyProvider("CRITERION 1: PASS - works\nCRITERION 2: FAIL - broken"),
        tool_registry=ToolRegistry(), max_iterations=5,
    )
    result = verify_criteria(["works", "broken"], "context", agent)
    assert result.passed is False
    assert len(result.issues) > 0


def test_all_criteria_fail():
    agent = AgentRunner(
        provider=VerifyProvider("CRITERION 1: FAIL - missing\nCRITERION 2: FAIL - also missing"),
        tool_registry=ToolRegistry(), max_iterations=5,
    )
    result = verify_criteria(["A", "B"], "context", agent)
    assert result.passed is False
    assert "0/2" in result.summary


def test_no_explicit_markers_but_all_pass():
    agent = AgentRunner(
        provider=VerifyProvider("Everything looks good. ALL CRITERIA PASS."),
        tool_registry=ToolRegistry(), max_iterations=5,
    )
    result = verify_criteria(["A", "B"], "context", agent)
    assert result.passed is True


def test_empty_criteria():
    agent = AgentRunner(
        provider=VerifyProvider("Nothing to verify"),
        tool_registry=ToolRegistry(), max_iterations=5,
    )
    result = verify_criteria([], "context", agent)
    assert result.passed is False  # No criteria = can't verify


def test_verify_tests_all_pass():
    agent = AgentRunner(
        provider=VerifyProvider("Ran pytest: 50 tests passed, 0 failed. ALL TESTS PASS."),
        tool_registry=ToolRegistry(), max_iterations=5,
    )
    result = verify_tests(agent)
    assert result.passed is True


def test_verify_tests_failures():
    agent = AgentRunner(
        provider=VerifyProvider("Ran pytest: 48 passed, 2 FAILED.\ntest_auth.py::test_login FAILED"),
        tool_registry=ToolRegistry(), max_iterations=5,
    )
    result = verify_tests(agent)
    assert result.passed is False


def test_verification_result_summary():
    r = VerificationResult(passed=True, criteria_results={"a": True, "b": True})
    assert "PASS" in r.summary
    assert "2/2" in r.summary


def test_verification_result_fail_summary():
    r = VerificationResult(passed=False, criteria_results={"a": True, "b": False, "c": False})
    assert "FAIL" in r.summary
    assert "1/3" in r.summary
