"""Integration test: autopilot mode with mocked LLM.

Tests the full plan -> execute -> verify lifecycle using mock providers.
No real LLM calls — deterministic and fast.
"""

import pytest
from unittest.mock import patch, MagicMock

from mullet.core.orchestration.autopilot import AutopilotMode, AutopilotResult
from mullet.core.orchestration.ralph import RalphMode, RalphResult
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class SequentialProvider(Provider):
    def __init__(self, responses):
        self.name = "test"
        self._responses = responses
        self._idx = 0

    def create(self, messages, tools=None, **kwargs):
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return LLMResponse(content=resp, usage=TokenUsage(total_tokens=50))


def _ralph_ok():
    r = MagicMock(spec=RalphResult)
    r.completed = True
    r.stories_completed = 1
    r.stories_total = 1
    r.total_tokens = TokenUsage(total_tokens=100)
    r.summary = "All stories completed."
    r.verification_state = None
    return r


def _ralph_fail():
    r = MagicMock(spec=RalphResult)
    r.completed = False
    r.stories_completed = 0
    r.stories_total = 1
    r.total_tokens = TokenUsage(total_tokens=50)
    r.summary = "Failed to complete stories."
    r.verification_state = None
    return r


@patch.object(RalphMode, "execute", return_value=_ralph_ok())
def test_autopilot_plan_execute_verify_success(mock_ralph):
    """Full lifecycle: plan, ralph succeeds, QA passes."""
    agent = AgentRunner(
        provider=SequentialProvider([
            "Plan: 1. Create file. 2. Verify.",
            "PASS: All checks passed.",
            "Summary: Task completed successfully.",
        ]),
        tool_registry=ToolRegistry(),
        max_iterations=10,
    )
    result = AutopilotMode(max_iterations=3).execute("Create greeting.py", agent)

    assert isinstance(result, AutopilotResult)
    assert result.completed is True
    assert result.qa_passed is True
    assert "Plan:" in result.plan
    assert result.total_tokens.total_tokens > 0


def test_autopilot_qa_fail_triggers_fix_then_passes():
    """QA fails first time, ralph re-runs, QA passes second time."""
    call_count = [0]

    def ralph_side(*a, **kw):
        call_count[0] += 1
        return _ralph_ok()

    agent = AgentRunner(
        provider=SequentialProvider([
            "Plan: Create greeting.py.",
            "FAIL: missing content.",
            "PASS: All checks passed.",
            "Summary: Fixed.",
        ]),
        tool_registry=ToolRegistry(),
        max_iterations=10,
    )
    with patch.object(RalphMode, "execute", side_effect=ralph_side):
        result = AutopilotMode(max_iterations=5).execute("Create greeting.py", agent)

    assert isinstance(result, AutopilotResult)
    assert call_count[0] >= 2


@patch.object(RalphMode, "execute", return_value=_ralph_fail())
def test_autopilot_incomplete_when_ralph_fails(mock_ralph):
    """Autopilot returns incomplete when ralph cannot finish stories."""
    agent = AgentRunner(
        provider=SequentialProvider(["Plan: Create greeting.py."]),
        tool_registry=ToolRegistry(),
        max_iterations=5,
    )
    result = AutopilotMode(max_iterations=3).execute("Create greeting.py", agent)

    assert isinstance(result, AutopilotResult)
    assert result.completed is False
    assert "ralph finished 0/1" in result.summary
    assert result.total_tokens.total_tokens > 0
