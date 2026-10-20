"""Tests for autopilot mode."""

import json
import pytest
from mullet.core.orchestration.autopilot import AutopilotMode, AutopilotResult
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


def test_autopilot_runs_plan_then_ralph():
    prd_json = json.dumps({
        "stories": [{"id": "US-001", "title": "Implement feature",
                      "acceptance_criteria": ["feature works"], "priority": 1}]
    })

    agent = AgentRunner(
        provider=SequentialProvider([
            "Plan: implement the feature step by step.",  # Planning phase
            prd_json,                  # Ralph PRD generation
            "Feature implemented.",    # Ralph implementation
            "ALL CRITERIA PASS",       # Ralph verification
            "Summary: everything done.",  # Final summary
        ]),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )

    autopilot = AutopilotMode(max_iterations=5)
    result = autopilot.execute("build a feature", agent)

    assert isinstance(result, AutopilotResult)
    assert result.plan != ""  # Plan was generated
    assert result.total_tokens.total_tokens > 0


def test_autopilot_result_fields():
    result = AutopilotResult()
    assert result.completed is False
    assert result.plan == ""
    assert result.qa_passed is False
    assert result.summary == ""


def test_autopilot_result_tokens_default():
    result = AutopilotResult()
    assert result.total_tokens is not None
    assert result.total_tokens.total_tokens == 0


def test_autopilot_result_qa_failure_count_default():
    result = AutopilotResult()
    assert result.qa_failure_count == 0


def test_autopilot_result_with_values():
    result = AutopilotResult(
        completed=True,
        plan="step 1\nstep 2",
        qa_passed=True,
        summary="all done",
        qa_failure_count=1,
    )
    assert result.completed is True
    assert result.plan == "step 1\nstep 2"
    assert result.qa_passed is True
    assert result.qa_failure_count == 1


def test_autopilot_init_defaults():
    mode = AutopilotMode(max_iterations=10)
    assert mode.max_iterations == 10
    assert mode.qa_failure_count == 0


def test_autopilot_resume_from_saved_state_with_task():
    """Cover lines 78-81: resume from saved state that includes a task."""
    from unittest.mock import MagicMock, patch
    from mullet.core.orchestration.ralph import RalphResult

    mock_state = MagicMock()
    mock_state.load.return_value = {"phase": "execute", "task": "resumed task", "plan": "old plan"}

    with patch("mullet.core.orchestration.autopilot.ModeState", return_value=mock_state):
        with patch("mullet.core.orchestration.autopilot.RalphMode") as MockRalph:
            ralph_instance = MockRalph.return_value
            ralph_instance.execute.return_value = RalphResult(
                completed=True,
                stories_completed=1,
                stories_total=1,
                total_tokens=TokenUsage(total_tokens=10),
                summary="ralph done",
            )

            with patch.object(AutopilotMode, "_run_automated_checks", return_value=MagicMock(passed=True, summary="ok")):
                agent = AgentRunner(
                    provider=SequentialProvider(["QA PASS", "Summary done."]),
                    tool_registry=ToolRegistry(),
                    max_iterations=30,
                )
                autopilot = AutopilotMode(max_iterations=5)
                result = autopilot.execute("", agent)

                assert isinstance(result, AutopilotResult)
                assert result.plan == "old plan"
                assert result.completed is True


def test_autopilot_resume_from_execute_no_plan_replans():
    """Cover lines 85-89: skip plan phase when resuming from 'execute' phase but no plan."""
    from unittest.mock import MagicMock, patch
    from mullet.core.orchestration.ralph import RalphResult

    mock_state = MagicMock()
    mock_state.load.return_value = {"phase": "execute", "task": "resumed task"}

    with patch("mullet.core.orchestration.autopilot.ModeState", return_value=mock_state):
        with patch("mullet.core.orchestration.autopilot.RalphMode") as MockRalph:
            ralph_instance = MockRalph.return_value
            ralph_instance.execute.return_value = RalphResult(
                completed=True,
                stories_completed=1,
                stories_total=1,
                total_tokens=TokenUsage(total_tokens=10),
                summary="ralph done",
            )

            with patch.object(AutopilotMode, "_run_automated_checks", return_value=MagicMock(passed=True, summary="ok")):
                agent = AgentRunner(
                    provider=SequentialProvider([
                        "Plan: do the thing",
                        "QA PASS",
                        "Summary done.",
                    ]),
                    tool_registry=ToolRegistry(),
                    max_iterations=30,
                )
                autopilot = AutopilotMode(max_iterations=5)
                result = autopilot.execute("", agent)

                assert isinstance(result, AutopilotResult)
                assert result.plan == "Plan: do the thing"
                assert result.completed is True


def test_autopilot_qa_phase_with_fix_cycle():
    """Cover lines 126-163: QA phase runs when ralph completes, fix cycle if qa fails."""
    from unittest.mock import MagicMock, patch
    from mullet.core.orchestration.ralph import RalphResult

    mock_state = MagicMock()
    mock_state.load.return_value = None

    with patch("mullet.core.orchestration.autopilot.ModeState", return_value=mock_state):
        with patch("mullet.core.orchestration.autopilot.RalphMode") as MockRalph:
            ralph_instance = MockRalph.return_value
            ralph_instance.execute.return_value = RalphResult(
                completed=True,
                stories_completed=1,
                stories_total=1,
                total_tokens=TokenUsage(total_tokens=10),
                summary="ralph done",
            )

            with patch.object(AutopilotMode, "_run_automated_checks", return_value=MagicMock(passed=False, summary="tests failed")):
                agent = AgentRunner(
                    provider=SequentialProvider([
                        "Plan: fix things",
                        "QA found issues",
                        "Summary after fix.",
                    ]),
                    tool_registry=ToolRegistry(),
                    max_iterations=30,
                )
                autopilot = AutopilotMode(max_iterations=5)
                result = autopilot.execute("fix things", agent)

                assert isinstance(result, AutopilotResult)
                assert result.completed is True
                assert ralph_instance.execute.call_count == 2


def test_run_automated_checks():
    """Cover lines 175-178: _run_automated_checks method."""
    from unittest.mock import patch, MagicMock

    autopilot = AutopilotMode(max_iterations=5)
    mock_checks = [MagicMock(), MagicMock()]
    with patch("mullet.core.orchestration.autopilot.create_protocol", return_value=mock_checks) as mock_create:
        with patch("mullet.core.orchestration.autopilot.run_protocol", return_value=MagicMock(passed=True, summary="all good")) as mock_run:
            result = autopilot._run_automated_checks()
            assert result.passed is True
            mock_create.assert_called_once_with(["build", "test"])
            mock_run.assert_called_once_with(mock_checks, fail_fast=True)
            assert mock_checks[0].command == "python -c 'import mullet'"
            assert mock_checks[1].command == "python -m pytest tests/ --ignore=tests/integration -q --tb=line -x"


def test_qa_method():
    """Cover lines 198-208: _qa method body."""
    from unittest.mock import MagicMock

    autopilot = AutopilotMode(max_iterations=5)
    agent = MagicMock()
    agent.run.return_value = MagicMock(final_response="QA PASS", tokens_used=TokenUsage(total_tokens=10))
    result = autopilot._qa("task", "plan", agent)
    assert result.final_response == "QA PASS"
    assert result.tokens_used.total_tokens == 10
    agent.run.assert_called_once()
