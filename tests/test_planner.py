"""Tests for the planner module."""

import json
from mullet.core.orchestration.planner import Plan, PlanStep, create_plan
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


def test_plan_execution_order_simple():
    plan = Plan(
        task="test",
        steps=[
            PlanStep(id="S-001", title="First"),
            PlanStep(id="S-002", title="Second"),
        ],
    )
    order = plan.execution_order
    assert [s.id for s in order] == ["S-001", "S-002"]


def test_plan_execution_order_with_deps():
    plan = Plan(
        task="test",
        steps=[
            PlanStep(id="S-003", title="Third", depends_on=["S-001", "S-002"]),
            PlanStep(id="S-001", title="First"),
            PlanStep(id="S-002", title="Second", depends_on=["S-001"]),
        ],
    )
    order = plan.execution_order
    ids = [s.id for s in order]
    assert ids.index("S-001") < ids.index("S-002")
    assert ids.index("S-002") < ids.index("S-003")


def test_plan_serialization():
    plan = Plan(
        task="build something",
        overview="Overview text",
        steps=[
            PlanStep(id="S-001", title="Step 1", files=["a.py"], acceptance_criteria=["works"]),
        ],
        risks=["might break"],
    )
    data = plan.to_dict()
    restored = Plan.from_dict(data)
    assert restored.task == "build something"
    assert len(restored.steps) == 1
    assert restored.steps[0].files == ["a.py"]
    assert restored.risks == ["might break"]


def test_create_plan_parses_json():
    plan_json = json.dumps({
        "overview": "Build a REST API",
        "steps": [
            {"id": "S-001", "title": "Create models", "description": "Define data models",
             "depends_on": [], "files": ["models.py"], "acceptance_criteria": ["models exist"]},
        ],
        "risks": ["schema changes"],
    })

    class TestProvider(Provider):
        name = "test"
        def create(self, messages, **kw):
            return LLMResponse(content=plan_json, usage=TokenUsage(total_tokens=100))

    agent = AgentRunner(provider=TestProvider(), tool_registry=ToolRegistry(), max_iterations=5)
    plan = create_plan("build an API", agent)

    assert len(plan.steps) == 1
    assert plan.steps[0].title == "Create models"
    assert plan.risks == ["schema changes"]


def test_create_plan_handles_bad_json():
    class TestProvider(Provider):
        name = "test"
        def create(self, messages, **kw):
            return LLMResponse(content="Here is my plan:\n- step 1\n- step 2", usage=TokenUsage())

    agent = AgentRunner(provider=TestProvider(), tool_registry=ToolRegistry(), max_iterations=5)
    plan = create_plan("do something", agent)

    # Should fallback gracefully — overview contains the text
    assert plan.overview != ""
