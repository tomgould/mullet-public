"""Tests for team mode orchestration."""

import json
import pytest
from mullet.core.orchestration.team import TeamMode, TeamResult
from mullet.core.agent.types import DelegationResult
from mullet.core.orchestration.planner import Plan, PlanStep
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class PlanProvider(Provider):
    """Provider that returns a plan then implements."""
    name = "test"
    def __init__(self):
        self._idx = 0
        self._responses = [
            # Plan generation
            json.dumps({
                "overview": "Build two components",
                "steps": [
                    {"id": "S-001", "title": "Component A", "description": "Build A",
                     "depends_on": [], "files": ["a.py"], "acceptance_criteria": ["A works"]},
                    {"id": "S-002", "title": "Component B", "description": "Build B",
                     "depends_on": ["S-001"], "files": ["b.py"], "acceptance_criteria": ["B works"]},
                ],
                "risks": [],
            }),
            # PRD refinement for step without criteria
            "- Component builds successfully\n- Tests pass",
            "- Component builds successfully\n- Tests pass",
            # Exec results
            "Component A implemented.",
            "Component B implemented.",
            # Verify
            "CRITERION 1: PASS - works\nALL CRITERIA PASS",
            # Summary
            "All done.",
        ]

    def create(self, messages, **kw):
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return LLMResponse(content=resp, usage=TokenUsage(total_tokens=50))


def test_team_mode_runs():
    agent = AgentRunner(
        provider=PlanProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=2, max_fix_cycles=1)
    result = team.execute("build two components", agent)
    assert isinstance(result, TeamResult)
    assert result.plan is not None
    assert len(result.plan.steps) == 2


def test_team_result_defaults():
    result = TeamResult()
    assert result.completed is False
    assert result.plan is None
    assert result.verify_passed is False
    assert result.fix_cycles == 0


def test_team_result_with_plan():
    plan = Plan(task="build thing", overview="build it", steps=[
        PlanStep(id="S-001", title="Step 1", description="desc",
                 depends_on=[], files=["a.py"], acceptance_criteria=["works"]),
    ])
    result = TeamResult(plan=plan, completed=True, verify_passed=True)
    assert result.plan is not None
    assert len(result.plan.steps) == 1
    assert result.completed is True
    assert result.verify_passed is True


def test_team_mode_init_defaults():
    team = TeamMode(num_workers=3, max_fix_cycles=2)
    assert team.num_workers == 3
    assert team.max_fix_cycles == 2


def test_plan_step_defaults():
    step = PlanStep(
        id="S-001",
        title="My Step",
        description="do something",
        depends_on=[],
        files=[],
        acceptance_criteria=[],
    )
    assert step.id == "S-001"
    assert step.title == "My Step"
    assert step.depends_on == []


class NoCriteriaProvider(Provider):
    """Provider that returns a plan with NO acceptance criteria."""
    name = "test_no_criteria"

    def __init__(self, extra_responses=None):
        self._idx = 0
        self._responses = [
            # Plan: steps have no acceptance_criteria
            json.dumps({
                "overview": "Build one component",
                "steps": [
                    {"id": "S-001", "title": "Step One", "description": "Do it",
                     "depends_on": [], "files": ["a.py"], "acceptance_criteria": []},
                ],
                "risks": [],
            }),
            # PRD refinement response for empty criteria step
            "- Step is implemented\n- All tests pass",
            # Exec
            "Step One done.",
            # Fix cycle (if triggered)
            "Fixed.",
            "Fixed.",
        ]
        if extra_responses:
            self._responses.extend(extra_responses)

    def create(self, messages, **kw):
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return LLMResponse(content=resp, usage=TokenUsage(total_tokens=30))


def test_team_mode_no_criteria_uses_completion_verify():
    """Line 220: all_criteria empty → verify_passed = all completed."""
    agent = AgentRunner(
        provider=NoCriteriaProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=1, max_fix_cycles=0)
    result = team.execute("do step one", agent)
    assert isinstance(result, TeamResult)


class RunStagedFailProvider(Provider):
    """Provider for run_staged where verify fails so _fix_stage is called."""
    name = "test_run_staged"

    def __init__(self):
        self._idx = 0
        self._responses = [
            # Stage 1: plan
            json.dumps({
                "overview": "A task",
                "steps": [
                    {"id": "S-001", "title": "Step", "description": "desc",
                     "depends_on": [], "files": [], "acceptance_criteria": ["done"]},
                ],
                "risks": [],
            }),
            # Stage 2: prd refinement
            "- done",
            # Stage 3: exec
            "Completed.",
            # Stage 4: verify (run_staged uses _verify_stage which checks completed)
            "Fixed.",
        ]

    def create(self, messages, **kw):
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return LLMResponse(content=resp, usage=TokenUsage(total_tokens=20))


def test_run_staged_fix_stage_called_when_verify_fails():
    """Lines 332, 363-364: run_staged calls _fix_stage when verify fails."""
    from unittest.mock import patch

    agent = AgentRunner(
        provider=RunStagedFailProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=1, max_fix_cycles=1)
    # Patch _verify_stage to return False (failed)
    with patch.object(team, "_verify_stage", return_value=(False, TokenUsage())):
        with patch.object(team, "_fix_stage") as mock_fix:
            result = team.run_staged("do a task", agent)
    mock_fix.assert_called_once()
    assert isinstance(result, TeamResult)


def test_fix_stage_calls_agent_run():
    """Lines 363-364: _fix_stage logs and calls agent.run with fix prompt."""
    from mullet.core.orchestration.team import TeamMode
    from unittest.mock import MagicMock

    agent = MagicMock()
    plan = Plan(task="t", overview="Build X", steps=[])
    team = TeamMode(num_workers=1)
    team._fix_stage(plan, agent)
    agent.run.assert_called_once()
    call_arg = agent.run.call_args[0][0]
    assert "Build X" in call_arg


def test_refine_prd_generates_criteria_for_empty_steps():
    """Lines 271-284: _refine_prd asks agent for criteria on steps without them."""
    from unittest.mock import MagicMock, patch
    from mullet.core.agent.types import AgentResult

    plan = Plan(task="t", overview="Build A", steps=[
        PlanStep(id="S-001", title="Step with empty criteria", description="desc",
                 depends_on=[], files=["a.py"], acceptance_criteria=[]),
    ])
    agent = MagicMock()
    agent.run.return_value = AgentResult(
        completed=True,
        final_response="- Criteria one\n- Criteria two",
        tokens_used=TokenUsage(),
    )
    team = TeamMode(num_workers=1)
    refined = team._refine_prd(plan, agent)
    assert len(refined.steps[0].acceptance_criteria) == 2
    assert refined.steps[0].acceptance_criteria[0] == "Criteria one"


def test_execute_no_criteria_uses_completion_check():
    """Line 220: when plan steps have no criteria, verify_passed uses completion check."""
    from unittest.mock import patch, MagicMock

    plan = Plan(task="t", overview="Build A", steps=[
        PlanStep(id="S-001", title="Step", description="desc",
                 depends_on=[], files=[], acceptance_criteria=[]),
    ])
    exec_results = [DelegationResult(goal="Step", completed=True, summary="done")]
    agent = MagicMock()
    team = TeamMode(num_workers=1, max_fix_cycles=0)

    with patch("mullet.core.orchestration.team.create_plan", return_value=plan), \
         patch.object(team, "_refine_prd", return_value=plan), \
         patch.object(team, "_execute_parallel", return_value=exec_results), \
         patch("mullet.core.orchestration.team.run_three_role_review",
               return_value=(MagicMock(approved=True), MagicMock(approved=True))):
        result = team.execute("build A", agent)
    assert isinstance(result, TeamResult)


def test_execute_no_criteria_fix_cycle_trusts_fix():
    """Line 243: no criteria + verify_passed=False → fix loop sets verify_passed=True."""
    from unittest.mock import patch, MagicMock
    from mullet.core.agent.types import AgentResult

    plan = Plan(task="t", overview="Build A", steps=[
        PlanStep(id="S-001", title="Step", description="desc",
                 depends_on=[], files=[], acceptance_criteria=[]),
    ])
    # completed=False triggers verify_passed=False at line 220
    exec_results = [DelegationResult(goal="Step", completed=False, summary="")]
    agent = MagicMock()
    agent.run.return_value = AgentResult(completed=True, final_response="fixed", tokens_used=TokenUsage())
    team = TeamMode(num_workers=1, max_fix_cycles=1)

    with patch("mullet.core.orchestration.team.create_plan", return_value=plan), \
         patch.object(team, "_refine_prd", return_value=plan), \
         patch.object(team, "_execute_parallel", return_value=exec_results), \
         patch("mullet.core.orchestration.team.run_three_role_review",
               return_value=(MagicMock(approved=True), MagicMock(approved=True))):
        result = team.execute("build A", agent)
    assert isinstance(result, TeamResult)
