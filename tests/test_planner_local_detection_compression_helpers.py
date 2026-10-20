"""Tests for Plan/PlanStep pure logic, local_detection cache helpers, and compression constants."""


from mullet.core.orchestration.planner import Plan, PlanStep
from mullet.core.providers import local_detection as ld
from mullet.core.agent.compression import (
    get_context_length,
    CHARS_PER_TOKEN,
    MODEL_CONTEXT_LENGTHS,
)


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------

class TestPlanStep:
    def test_id_stored(self):
        s = PlanStep(id="S-001", title="step")
        assert s.id == "S-001"

    def test_title_stored(self):
        s = PlanStep(id="S-001", title="My Title")
        assert s.title == "My Title"

    def test_description_defaults_empty(self):
        s = PlanStep(id="S-001", title="step")
        assert s.description == ""

    def test_depends_on_defaults_empty(self):
        s = PlanStep(id="S-001", title="step")
        assert s.depends_on == []

    def test_files_defaults_empty(self):
        s = PlanStep(id="S-001", title="step")
        assert s.files == []

    def test_acceptance_criteria_defaults_empty(self):
        s = PlanStep(id="S-001", title="step")
        assert s.acceptance_criteria == []

    def test_assigned_to_defaults_none(self):
        s = PlanStep(id="S-001", title="step")
        assert s.assigned_to is None

    def test_custom_values_stored(self):
        s = PlanStep(
            id="S-002",
            title="deploy",
            description="Deploy the app",
            depends_on=["S-001"],
            files=["app.py"],
            acceptance_criteria=["runs"],
            assigned_to="agent-1",
        )
        assert s.id == "S-002"
        assert s.depends_on == ["S-001"]
        assert s.files == ["app.py"]
        assert s.acceptance_criteria == ["runs"]
        assert s.assigned_to == "agent-1"


# ---------------------------------------------------------------------------
# Plan — construction and to_dict
# ---------------------------------------------------------------------------

class TestPlanConstruction:
    def test_task_stored(self):
        p = Plan(task="build app")
        assert p.task == "build app"

    def test_overview_defaults_empty(self):
        p = Plan(task="t")
        assert p.overview == ""

    def test_steps_defaults_empty(self):
        p = Plan(task="t")
        assert p.steps == []

    def test_risks_defaults_empty(self):
        p = Plan(task="t")
        assert p.risks == []


class TestPlanToDict:
    def test_returns_dict(self):
        p = Plan(task="t")
        assert isinstance(p.to_dict(), dict)

    def test_task_in_dict(self):
        p = Plan(task="build app")
        assert p.to_dict()["task"] == "build app"

    def test_overview_in_dict(self):
        p = Plan(task="t", overview="do things")
        assert p.to_dict()["overview"] == "do things"

    def test_empty_steps_in_dict(self):
        p = Plan(task="t")
        assert p.to_dict()["steps"] == []

    def test_risks_in_dict(self):
        p = Plan(task="t", risks=["risk1", "risk2"])
        assert p.to_dict()["risks"] == ["risk1", "risk2"]

    def test_steps_serialized(self):
        s = PlanStep(id="S-001", title="first", files=["f.py"])
        p = Plan(task="t", steps=[s])
        d = p.to_dict()
        assert len(d["steps"]) == 1
        assert d["steps"][0]["id"] == "S-001"
        assert d["steps"][0]["files"] == ["f.py"]

    def test_step_dict_has_required_keys(self):
        s = PlanStep(id="S-001", title="step")
        p = Plan(task="t", steps=[s])
        step_d = p.to_dict()["steps"][0]
        for key in ("id", "title", "description", "depends_on", "files", "acceptance_criteria"):
            assert key in step_d


# ---------------------------------------------------------------------------
# Plan — from_dict
# ---------------------------------------------------------------------------

class TestPlanFromDict:
    def _data(self, **kw):
        base = {"task": "mytask", "overview": "overview text", "steps": [], "risks": []}
        base.update(kw)
        return base

    def test_task_set(self):
        p = Plan.from_dict(self._data())
        assert p.task == "mytask"

    def test_overview_set(self):
        p = Plan.from_dict(self._data())
        assert p.overview == "overview text"

    def test_empty_steps(self):
        p = Plan.from_dict(self._data())
        assert p.steps == []

    def test_risks_set(self):
        p = Plan.from_dict(self._data(risks=["r1"]))
        assert p.risks == ["r1"]

    def test_steps_parsed(self):
        step_data = {"id": "S-001", "title": "step1", "description": "desc",
                     "depends_on": [], "files": ["f.py"], "acceptance_criteria": ["AC"]}
        p = Plan.from_dict(self._data(steps=[step_data]))
        assert len(p.steps) == 1
        assert p.steps[0].id == "S-001"
        assert p.steps[0].files == ["f.py"]
        assert p.steps[0].acceptance_criteria == ["AC"]

    def test_step_optional_fields_default(self):
        step_data = {"id": "S-001", "title": "minimal"}
        p = Plan.from_dict(self._data(steps=[step_data]))
        assert p.steps[0].description == ""
        assert p.steps[0].depends_on == []

    def test_missing_task_defaults_empty(self):
        p = Plan.from_dict({})
        assert p.task == ""

    def test_roundtrip(self):
        original = Plan(
            task="build",
            overview="Build the thing",
            steps=[PlanStep(id="S-001", title="A", files=["a.py"], depends_on=[])],
            risks=["r1"],
        )
        restored = Plan.from_dict(original.to_dict())
        assert restored.task == original.task
        assert restored.overview == original.overview
        assert restored.risks == original.risks
        assert len(restored.steps) == 1
        assert restored.steps[0].id == "S-001"
        assert restored.steps[0].files == ["a.py"]


# ---------------------------------------------------------------------------
# Plan — execution_order
# ---------------------------------------------------------------------------

class TestPlanExecutionOrder:
    def test_empty_plan_returns_empty(self):
        p = Plan(task="t")
        assert p.execution_order == []

    def test_single_step_no_deps(self):
        p = Plan(task="t", steps=[PlanStep(id="S-001", title="A")])
        assert [s.id for s in p.execution_order] == ["S-001"]

    def test_independent_steps_all_returned(self):
        steps = [PlanStep(id=f"S-00{i}", title=f"step{i}") for i in range(1, 4)]
        p = Plan(task="t", steps=steps)
        assert len(p.execution_order) == 3

    def test_sequential_dependencies_respected(self):
        steps = [
            PlanStep(id="S-001", title="A", depends_on=[]),
            PlanStep(id="S-002", title="B", depends_on=["S-001"]),
            PlanStep(id="S-003", title="C", depends_on=["S-002"]),
        ]
        p = Plan(task="t", steps=steps)
        order = [s.id for s in p.execution_order]
        assert order.index("S-001") < order.index("S-002")
        assert order.index("S-002") < order.index("S-003")

    def test_fan_out_dependencies(self):
        steps = [
            PlanStep(id="S-001", title="root", depends_on=[]),
            PlanStep(id="S-002", title="B", depends_on=["S-001"]),
            PlanStep(id="S-003", title="C", depends_on=["S-001"]),
        ]
        p = Plan(task="t", steps=steps)
        order = [s.id for s in p.execution_order]
        assert order[0] == "S-001"
        assert "S-002" in order
        assert "S-003" in order

    def test_circular_deps_all_steps_returned(self):
        steps = [
            PlanStep(id="S-001", title="A", depends_on=["S-002"]),
            PlanStep(id="S-002", title="B", depends_on=["S-001"]),
        ]
        p = Plan(task="t", steps=steps)
        order = p.execution_order
        assert len(order) == 2

    def test_returns_list(self):
        p = Plan(task="t")
        assert isinstance(p.execution_order, list)

    def test_all_plan_steps_present(self):
        steps = [PlanStep(id=f"S-{i:03d}", title=f"step{i}") for i in range(5)]
        p = Plan(task="t", steps=steps)
        order_ids = {s.id for s in p.execution_order}
        expected_ids = {f"S-{i:03d}" for i in range(5)}
        assert order_ids == expected_ids


# ---------------------------------------------------------------------------
# local_detection — pure cache helpers
# ---------------------------------------------------------------------------

class TestLocalDetectionConstants:
    def test_local_ports_is_list(self):
        assert isinstance(ld.LOCAL_PORTS, list)

    def test_local_ports_nonempty(self):
        assert len(ld.LOCAL_PORTS) > 0

    def test_lm_studio_port_present(self):
        assert 1234 in ld.LOCAL_PORTS

    def test_ollama_port_present(self):
        assert 11434 in ld.LOCAL_PORTS

    def test_cache_ttl_positive(self):
        assert ld._CACHE_TTL > 0

    def test_cache_ttl_reasonable(self):
        assert ld._CACHE_TTL <= 300.0


class TestClearLocalProviderCache:
    def setup_method(self):
        ld.clear_local_provider_cache()

    def test_clears_cached_provider(self):
        ld.clear_local_provider_cache()
        assert ld._cached_local_provider is None

    def test_clears_timestamp(self):
        ld.clear_local_provider_cache()
        assert ld._cache_timestamp == 0.0

    def test_clears_refresh_task(self):
        ld.clear_local_provider_cache()
        assert ld._refresh_task is None


class TestCacheIsFresh:
    def setup_method(self):
        ld.clear_local_provider_cache()

    def test_returns_false_when_cold(self):
        assert ld._cache_is_fresh() is False

    def test_returns_bool(self):
        assert isinstance(ld._cache_is_fresh(), bool)

    def test_false_when_no_provider(self):
        ld._cached_local_provider = None
        assert ld._cache_is_fresh() is False

    def test_false_when_timestamp_is_zero(self):
        ld._cache_timestamp = 0.0
        assert ld._cache_is_fresh() is False

    def teardown_method(self):
        ld.clear_local_provider_cache()


# ---------------------------------------------------------------------------
# compression — get_context_length and constants
# ---------------------------------------------------------------------------

class TestCompressionConstants:
    def test_chars_per_token_positive(self):
        assert CHARS_PER_TOKEN > 0

    def test_chars_per_token_value(self):
        assert CHARS_PER_TOKEN == 4

    def test_model_context_lengths_is_dict(self):
        assert isinstance(MODEL_CONTEXT_LENGTHS, dict)

    def test_default_key_present(self):
        assert "default" in MODEL_CONTEXT_LENGTHS

    def test_default_context_positive(self):
        assert MODEL_CONTEXT_LENGTHS["default"] > 0

    def test_gpt4_context_large(self):
        assert MODEL_CONTEXT_LENGTHS["gpt-4"] >= 128_000


class TestGetContextLength:
    def test_returns_int(self):
        assert isinstance(get_context_length("gpt-4"), int)

    def test_positive_for_known_model(self):
        assert get_context_length("gpt-4") > 0

    def test_gpt4_context_length(self):
        assert get_context_length("gpt-4") == 128_000

    def test_gpt4o_context_length(self):
        assert get_context_length("gpt-4o") == 128_000

    def test_claude_opus_large_context(self):
        assert get_context_length("claude-opus") > 100_000

    def test_gemini_large_context(self):
        assert get_context_length("gemini") > 100_000

    def test_unknown_model_returns_positive(self):
        assert get_context_length("totally-unknown-xyz") > 0

    def test_empty_string_returns_positive(self):
        assert get_context_length("") > 0

    def test_empty_string_returns_default(self):
        assert get_context_length("") == get_context_length("truly-no-match-xyz")
