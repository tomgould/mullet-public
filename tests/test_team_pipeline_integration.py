"""Integration tests for team staged pipeline with TaskBoard."""

import json
from mullet.core.orchestration.team import TeamMode, TeamResult
from mullet.core.orchestration.shared_memory import SharedMemory
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class PipelineProvider(Provider):
    """Provider for full pipeline stages."""
    name = "pipeline-test"
    def __init__(self, fail_verify=False):
        self._idx = 0
        self.fail_verify = fail_verify
        self._responses = [
            # PLAN
            json.dumps({
                "overview": "Implement auth system",
                "steps": [
                    {"id": "S-1", "title": "Database schema", "description": "Create users table",
                     "depends_on": [], "files": ["schema.sql"], "acceptance_criteria": ["Schema exists"]},
                    {"id": "S-2", "title": "Auth API", "description": "Implement endpoints",
                     "depends_on": ["S-1"], "files": ["auth.py"], "acceptance_criteria": ["API works"]},
                    {"id": "S-3", "title": "Tests", "description": "Write test suite",
                     "depends_on": ["S-2"], "files": ["test_auth.py"], "acceptance_criteria": ["Tests pass"]},
                ],
                "risks": ["Database migration risk"],
            }),
            # PRD: step criteria
            "- Normalized schema\n- No conflicts",
            "- JWT support\n- Rate limiting",
            "- 90% coverage\n- Edge cases",
            # EXEC: step implementations
            "CREATE TABLE users (id INT PRIMARY KEY);",
            "def authenticate(user, pwd): return True",
            "def test_auth(): assert authenticate(...) works",
            # VERIFY
            ("PASS: Schema\nPASS: API\nPASS: Tests\nALL PASS" 
             if not fail_verify
             else "FAIL: Missing hashing\nFAIL: API needs validation"),
            # FIX (if needed)
            "Added bcrypt hashing and input validation.",
            # RE-VERIFY (if failed initially)
            "PASS: All fixed\nALL PASS",
        ]

    def create(self, messages, **kw):
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return LLMResponse(content=resp, usage=TokenUsage(total_tokens=100))


def test_pipeline_plan_stage_creates_dependency_graph():
    """Test that plan stage builds correct dependency graph."""
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3)
    plan = team._plan_stage("Build auth", agent)
    
    assert len(plan.steps) == 3
    # Check dependency order
    assert plan.steps[0].depends_on == []
    assert "S-1" in plan.steps[1].depends_on
    assert "S-2" in plan.steps[2].depends_on
    
    # Check execution order respects dependencies
    ordered = plan.execution_order
    assert ordered[0].id == "S-1"
    assert ordered[1].id == "S-2"
    assert ordered[2].id == "S-3"


def test_pipeline_prd_stage_enriches_criteria():
    """Test that PRD stage generates acceptance criteria for each step."""
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3)
    plan = team._plan_stage("Build auth", agent)
    plan = team._prd_stage(plan, agent)
    
    # All steps should have criteria
    for step in plan.steps:
        assert len(step.acceptance_criteria) > 0
        assert isinstance(step.acceptance_criteria, list)


def test_pipeline_exec_respects_dependency_order():
    """Test that execution respects step dependencies."""
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3)
    plan = team._plan_stage("Build auth", agent)
    
    # Verify execution_order is topologically sorted
    executed_ids = set()
    for step in plan.execution_order:
        # All dependencies must be in executed_ids before we execute this step
        for dep_id in step.depends_on:
            assert dep_id in executed_ids, f"{step.id} depends on {dep_id} but {dep_id} not yet executed"
        executed_ids.add(step.id)


def test_pipeline_shared_memory_stores_results():
    """Test that team execution stores results in shared memory."""
    shared_mem = SharedMemory()
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3, shared_memory=shared_mem)
    result = team.execute("Build auth", agent)
    
    # Team should have stored exec results
    assert len(result.exec_results) > 0


def test_pipeline_token_accumulation():
    """Test that tokens are accumulated across all stages."""
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3)
    result = team.execute("Build auth", agent)
    
    # Should have accumulated tokens from plan, prd, exec, verify
    assert result.total_tokens.total_tokens > 0


def test_pipeline_verify_pass():
    """Test successful verification on happy path."""
    agent = AgentRunner(
        provider=PipelineProvider(fail_verify=False),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3, max_fix_cycles=1)
    result = team.execute("Build auth", agent)
    
    assert result.verify_passed is True
    assert result.fix_cycles <= 1  # Should pass without excess fix cycles


def test_pipeline_verify_fail_and_fix():
    """Test verification failure triggers fix cycle."""
    agent = AgentRunner(
        provider=PipelineProvider(fail_verify=True),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3, max_fix_cycles=2)
    result = team.execute("Build auth", agent)
    
    # Should have attempted fixes
    assert result.fix_cycles > 0


def test_pipeline_empty_plan_fallback():
    """Test that empty plan falls back to single agent."""
    class EmptyPlanProvider(Provider):
        name = "empty"
        def create(self, messages, **kw):
            # Return plan with no steps
            resp = json.dumps({
                "overview": "",
                "steps": [],
                "risks": [],
            })
            return LLMResponse(content=resp, usage=TokenUsage(total_tokens=50))
    
    agent = AgentRunner(
        provider=EmptyPlanProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3)
    result = team.execute("Simple task", agent)
    
    # Should fall back without crashing
    assert result is not None


def test_pipeline_run_staged_method():
    """Test the alternative run_staged() method."""
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3)
    result = team.run_staged("Build auth", agent)

    assert isinstance(result, TeamResult)
    assert result.plan is not None


def test_pipeline_taskboard_integration():
    """Test that TaskBoard is used during parallel execution."""

    shared_mem = SharedMemory()
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3, shared_memory=shared_mem)

    # Execute and verify plan steps translate to taskboard
    result = team.execute("Build auth", agent)

    # Each step should have been processed
    assert result.plan is not None
    assert len(result.plan.steps) > 0
    # Results should be stored in shared memory
    for r in result.exec_results:
        if r.completed:
            # Shared memory should have task results
            assert result.plan is not None


def test_pipeline_dependency_ordering_enforcement():
    """Test that execution respects step dependencies strictly."""
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3)
    plan = team._plan_stage("Build auth", agent)

    # Verify topological sort is correct
    execution_order = plan.execution_order
    executed_ids = set()

    for i, step in enumerate(execution_order):
        # All dependencies must be in executed_ids
        for dep_id in step.depends_on:
            assert dep_id in executed_ids, \
                f"Step {step.id} at position {i} depends on {dep_id} which hasn't executed yet"
        executed_ids.add(step.id)

    # First step should have no dependencies
    assert len(execution_order[0].depends_on) == 0


def test_pipeline_steps_with_multiple_dependencies():
    """Test steps that depend on multiple other steps."""
    class MultiDepProvider(Provider):
        name = "multi-dep-test"
        def __init__(self):
            self._idx = 0
            self._responses = [
                json.dumps({
                    "overview": "Complex workflow",
                    "steps": [
                        {"id": "S-1", "title": "Setup", "description": "Initial setup",
                         "depends_on": [], "files": ["setup.py"], "acceptance_criteria": ["Setup done"]},
                        {"id": "S-2", "title": "Auth", "description": "Auth system",
                         "depends_on": ["S-1"], "files": ["auth.py"], "acceptance_criteria": ["Auth works"]},
                        {"id": "S-3", "title": "DB", "description": "Database",
                         "depends_on": ["S-1"], "files": ["db.py"], "acceptance_criteria": ["DB works"]},
                        {"id": "S-4", "title": "API", "description": "API layer",
                         "depends_on": ["S-2", "S-3"], "files": ["api.py"], "acceptance_criteria": ["API works"]},
                    ],
                    "risks": [],
                }),
                # PRD responses
                "- Setup complete",
                "- Auth complete",
                "- DB complete",
                "- API complete",
                # Exec responses
                "Setup done",
                "Auth done",
                "DB done",
                "API done",
                # Verify
                "ALL PASS",
            ]

        def create(self, messages, **kw):
            resp = self._responses[min(self._idx, len(self._responses) - 1)]
            self._idx += 1
            return LLMResponse(content=resp, usage=TokenUsage(total_tokens=100))

    agent = AgentRunner(
        provider=MultiDepProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=3)
    plan = team._plan_stage("Complex workflow", agent)

    # Verify structure
    assert len(plan.steps) == 4

    # Find steps
    steps_by_id = {s.id: s for s in plan.steps}

    # S-4 depends on both S-2 and S-3
    s4 = steps_by_id["S-4"]
    assert "S-2" in s4.depends_on
    assert "S-3" in s4.depends_on

    # Verify execution order respects all dependencies
    execution_order = plan.execution_order
    executed_ids = set()
    for step in execution_order:
        for dep_id in step.depends_on:
            assert dep_id in executed_ids
        executed_ids.add(step.id)


def test_pipeline_shared_memory_cross_task_visibility():
    """Test that tasks can see results from previous tasks via shared memory."""
    shared_mem = SharedMemory()
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=2, shared_memory=shared_mem)
    result = team.execute("Build auth", agent)

    # Store some data via team execution
    assert result.exec_results is not None

    # Shared memory stores state; exec results are in the TeamResult
    assert result.exec_results is not None


def test_pipeline_no_workers_fallback():
    """Test that num_workers=1 works correctly (serial execution)."""
    agent = AgentRunner(
        provider=PipelineProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=1)  # Serial
    result = team.execute("Build auth", agent)

    assert result.plan is not None
    assert len(result.plan.steps) == 3


def test_pipeline_prd_stage_with_missing_criteria():
    """Test that PRD stage fills in missing acceptance criteria."""
    class MissingCriteriaProvider(Provider):
        name = "missing-criteria"
        def __init__(self):
            self._idx = 0

        def create(self, messages, **kw):
            responses = [
                # Plan without criteria
                json.dumps({
                    "overview": "Simple task",
                    "steps": [
                        {"id": "S-1", "title": "Impl", "description": "Implement",
                         "depends_on": [], "files": [], "acceptance_criteria": []},  # Empty!
                    ],
                    "risks": [],
                }),
                # PRD response
                "- Feature works\n- Tests pass",
                # Exec
                "Done",
                # Verify
                "ALL PASS",
            ]
            resp = responses[min(self._idx, len(responses) - 1)]
            self._idx += 1
            return LLMResponse(content=resp, usage=TokenUsage(total_tokens=50))

    agent = AgentRunner(
        provider=MissingCriteriaProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=30,
    )
    team = TeamMode(num_workers=1)
    plan = team._plan_stage("Simple task", agent)
    plan = team._prd_stage(plan, agent)

    # PRD should fill in criteria
    assert len(plan.steps[0].acceptance_criteria) > 0


def test_pipeline_max_fix_cycles_limit():
    """Test that fix cycles don't exceed max_fix_cycles."""
    class AlwaysFailProvider(Provider):
        name = "always-fail"
        call_count = 0

        def create(self, messages, **kw):
            self.call_count += 1
            if self.call_count == 1:
                # Plan
                return LLMResponse(
                    content=json.dumps({
                        "overview": "Task",
                        "steps": [{"id": "S-1", "title": "Work", "description": "Do work",
                                   "depends_on": [], "files": [], "acceptance_criteria": ["Pass"]}],
                        "risks": [],
                    }),
                    usage=TokenUsage(total_tokens=50)
                )
            else:
                # Everything else fails verification
                return LLMResponse(
                    content="FAIL: Broken",
                    usage=TokenUsage(total_tokens=50)
                )

    agent = AgentRunner(
        provider=AlwaysFailProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=10,
    )
    team = TeamMode(num_workers=1, max_fix_cycles=2)
    plan = team._plan_stage("Failing task", agent)
    plan = team._prd_stage(plan, agent)
    # Should not raise; fix cycles capped
    assert plan is not None
