"""Coverage-gap tests for mullet.core.orchestration.planner."""

import json
from unittest.mock import MagicMock

from mullet.core.orchestration.planner import Plan, PlanStep, create_plan, consensus_plan
from mullet.core.agent.types import AgentResult


class TestCreatePlanGaps:
    """Lines 122, 133-134: agent not completed / JSON/KeyError catch."""

    def test_create_plan_agent_not_completed(self):
        agent = MagicMock()
        agent.run.return_value = AgentResult(completed=False, final_response="")
        plan = create_plan("some task", agent)
        assert plan.task == "some task"
        assert plan.steps == []

    def test_create_plan_json_decode_error(self):
        agent = MagicMock()
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="this is not {valid json",
        )
        plan = create_plan("some task", agent)
        assert plan.task == "some task"
        assert plan.overview == "this is not {valid json"

    def test_create_plan_key_error(self):
        agent = MagicMock()
        agent.run.return_value = AgentResult(
            completed=True,
            final_response='{"overview": "ok", "steps": [{"id": 1}]}',
        )
        plan = create_plan("some task", agent)
        assert plan.task == "some task"
        assert "ok" in plan.overview


class TestConsensusPlanGaps:
    """Lines 141-149: consensus_plan full implementation."""

    def test_consensus_plan_approves_immediately(self):
        agent = MagicMock()
        agent.run.side_effect = [
            MagicMock(final_response="plan A"),
            MagicMock(final_response="I APPROVE this plan"),
            MagicMock(final_response="NO ISSUES found"),
        ]
        plan = consensus_plan("build api", agent, max_rounds=3)
        assert plan.task == "build api"
        assert plan.overview == "plan A"

    def test_consensus_plan_never_approves(self):
        agent = MagicMock()
        agent.run.side_effect = [
            MagicMock(final_response="plan A"),
            MagicMock(final_response="needs work"),
            MagicMock(final_response="issues found"),
            MagicMock(final_response="plan B"),
            MagicMock(final_response="needs work"),
            MagicMock(final_response="issues found"),
            MagicMock(final_response="plan C"),
            MagicMock(final_response="needs work"),
            MagicMock(final_response="issues found"),
        ]
        plan = consensus_plan("build api", agent, max_rounds=3)
        assert plan.task == "build api"
        assert plan.overview == "plan C"

    def test_consensus_plan_approves_on_second_round(self):
        agent = MagicMock()
        agent.run.side_effect = [
            MagicMock(final_response="plan A"),
            MagicMock(final_response="needs work"),   # review round 1
            MagicMock(final_response="issues found"), # critique round 1
            MagicMock(final_response="plan B"),
            MagicMock(final_response="APPROVE"),      # review round 2
            MagicMock(final_response="no issues"),    # critique round 2
        ]
        plan = consensus_plan("build api", agent, max_rounds=3)
        assert plan.task == "build api"
        assert plan.overview == "plan B"
