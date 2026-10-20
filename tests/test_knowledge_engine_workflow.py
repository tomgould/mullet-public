import sqlite3
import threading
import pytest
from pathlib import Path
from mullet.knowledge.engine.workflow import WorkflowEngine


@pytest.fixture
def engine(tmp_path):
    e = WorkflowEngine(tmp_path / "workflow.db")
    yield e
    e.close()


@pytest.fixture
def mem_engine():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    e = WorkflowEngine(Path("/tmp/fake.db"), connection=conn)
    yield e
    # don't close — external connection ownership


class TestInit:
    def test_creates_tables(self, mem_engine):
        # tables exist
        tables = {r[0] for r in mem_engine._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "workflows" in tables
        assert "workflow_steps" in tables

    def test_external_connection_not_owned(self, tmp_path):
        conn = sqlite3.connect(":memory:")
        e = WorkflowEngine(tmp_path / "wf.db", connection=conn)
        assert not e._owns_conn

    def test_external_lock_used(self, tmp_path):
        lock = threading.Lock()
        e = WorkflowEngine(tmp_path / "wf.db", lock=lock)
        assert e._lock is lock
        e.close()


class TestCreate:
    def test_empty_steps_raises(self, engine):
        with pytest.raises(ValueError, match="at least one step"):
            engine.create("test", [])

    def test_duplicate_step_name_raises(self, engine):
        with pytest.raises(ValueError, match="Duplicate step name"):
            engine.create("test", [{"name": "a"}, {"name": "a"}])

    def test_unknown_dependency_raises(self, engine):
        with pytest.raises(ValueError, match="unknown step"):
            engine.create("test", [{"name": "a", "depends_on": ["nonexistent"]}])

    def test_circular_dependency_raises(self, engine):
        with pytest.raises(ValueError, match="circular"):
            engine.create("test", [
                {"name": "a", "depends_on": ["b"]},
                {"name": "b", "depends_on": ["a"]},
            ])

    def test_single_step_created(self, engine):
        wf_id = engine.create("my workflow", [{"name": "step1", "description": "do it"}])
        assert wf_id
        wf = engine.get(wf_id)
        assert wf["name"] == "my workflow"
        assert len(wf["steps"]) == 1
        assert wf["steps"][0]["name"] == "step1"

    def test_linear_dependency_created_in_order(self, engine):
        wf_id = engine.create("pipeline", [
            {"name": "a"},
            {"name": "b", "depends_on": ["a"]},
            {"name": "c", "depends_on": ["b"]},
        ])
        wf = engine.get(wf_id)
        names = [s["name"] for s in wf["steps"]]
        assert names.index("a") < names.index("b") < names.index("c")

    def test_step_without_description_defaults_to_empty(self, engine):
        wf_id = engine.create("test", [{"name": "step"}])
        wf = engine.get(wf_id)
        assert wf["steps"][0]["description"] == ""

    def test_returns_string_uuid(self, engine):
        wf_id = engine.create("test", [{"name": "s"}])
        assert isinstance(wf_id, str) and len(wf_id) == 36  # UUID format


class TestGet:
    def test_nonexistent_returns_none(self, engine):
        assert engine.get("bad-id") is None

    def test_get_has_required_keys(self, engine):
        wf_id = engine.create("test", [{"name": "s"}])
        wf = engine.get(wf_id)
        for key in ("id", "name", "status", "created_at", "updated_at", "steps"):
            assert key in wf

    def test_step_has_required_keys(self, engine):
        wf_id = engine.create("test", [{"name": "s", "depends_on": []}])
        step = engine.get(wf_id)["steps"][0]
        for key in ("id", "name", "description", "depends_on", "status", "result", "started_at", "completed_at", "sort_order"):
            assert key in step

    def test_step_depends_on_is_list(self, engine):
        wf_id = engine.create("test", [{"name": "a"}, {"name": "b", "depends_on": ["a"]}])
        steps = {s["name"]: s for s in engine.get(wf_id)["steps"]}
        assert isinstance(steps["b"]["depends_on"], list)
        assert len(steps["b"]["depends_on"]) == 1


class TestList:
    def test_empty_returns_empty_list(self, engine):
        assert engine.list() == []

    def test_lists_created_workflow(self, engine):
        engine.create("wf1", [{"name": "s"}])
        result = engine.list()
        assert len(result) == 1
        assert result[0]["name"] == "wf1"

    def test_list_has_step_counts(self, engine):
        engine.create("wf", [{"name": "a"}, {"name": "b"}])
        result = engine.list()
        assert result[0]["total_steps"] == 2
        assert result[0]["completed_steps"] == 0

    def test_status_filter(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        step_id = engine.get(wf_id)["steps"][0]["id"]
        engine.update_step(wf_id, step_id, "completed")
        assert len(engine.list(status="completed")) == 1
        assert len(engine.list(status="pending")) == 0

    def test_multiple_workflows_ordered_descending(self, engine):
        engine.create("first", [{"name": "s"}])
        engine.create("second", [{"name": "s"}])
        names = [w["name"] for w in engine.list()]
        assert names[0] == "second"  # most recent first


class TestNextSteps:
    def test_empty_workflow_id_returns_empty(self, engine):
        assert engine.next_steps("nonexistent") == []

    def test_first_step_with_no_deps_is_ready(self, engine):
        wf_id = engine.create("wf", [{"name": "start"}])
        ready = engine.next_steps(wf_id)
        assert len(ready) == 1
        assert ready[0]["name"] == "start"

    def test_step_with_pending_dep_not_ready(self, engine):
        wf_id = engine.create("wf", [
            {"name": "a"},
            {"name": "b", "depends_on": ["a"]},
        ])
        ready_names = {s["name"] for s in engine.next_steps(wf_id)}
        assert "a" in ready_names
        assert "b" not in ready_names

    def test_step_ready_after_dep_completed(self, engine):
        wf_id = engine.create("wf", [
            {"name": "a"},
            {"name": "b", "depends_on": ["a"]},
        ])
        steps = {s["name"]: s for s in engine.get(wf_id)["steps"]}
        engine.update_step(wf_id, steps["a"]["id"], "completed")
        ready_names = {s["name"] for s in engine.next_steps(wf_id)}
        assert "b" in ready_names

    def test_step_ready_after_dep_skipped(self, engine):
        wf_id = engine.create("wf", [
            {"name": "a"},
            {"name": "b", "depends_on": ["a"]},
        ])
        steps = {s["name"]: s for s in engine.get(wf_id)["steps"]}
        engine.update_step(wf_id, steps["a"]["id"], "skipped")
        ready_names = {s["name"] for s in engine.next_steps(wf_id)}
        assert "b" in ready_names


class TestUpdateStep:
    def test_invalid_status_raises(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        step_id = engine.get(wf_id)["steps"][0]["id"]
        with pytest.raises(ValueError, match="Invalid status"):
            engine.update_step(wf_id, step_id, "BOGUS")

    def test_nonexistent_step_returns_false(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        assert engine.update_step(wf_id, "bad-step-id", "running") is False

    def test_running_sets_started_at(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        step_id = engine.get(wf_id)["steps"][0]["id"]
        engine.update_step(wf_id, step_id, "running")
        step = engine.get(wf_id)["steps"][0]
        assert step["started_at"] is not None

    def test_completed_sets_completed_at(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        step_id = engine.get(wf_id)["steps"][0]["id"]
        engine.update_step(wf_id, step_id, "completed", result="done")
        step = engine.get(wf_id)["steps"][0]
        assert step["completed_at"] is not None
        assert step["result"] == "done"

    def test_failed_sets_completed_at(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        step_id = engine.get(wf_id)["steps"][0]["id"]
        engine.update_step(wf_id, step_id, "failed")
        step = engine.get(wf_id)["steps"][0]
        assert step["completed_at"] is not None

    def test_skipped_sets_completed_at(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        step_id = engine.get(wf_id)["steps"][0]["id"]
        engine.update_step(wf_id, step_id, "skipped")
        step = engine.get(wf_id)["steps"][0]
        assert step["completed_at"] is not None

    def test_all_completed_updates_workflow_to_completed(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        step_id = engine.get(wf_id)["steps"][0]["id"]
        engine.update_step(wf_id, step_id, "completed")
        assert engine.get(wf_id)["status"] == "completed"

    def test_running_step_updates_workflow_to_running(self, engine):
        wf_id = engine.create("wf", [{"name": "a"}, {"name": "b"}])
        steps = engine.get(wf_id)["steps"]
        engine.update_step(wf_id, steps[0]["id"], "running")
        assert engine.get(wf_id)["status"] == "running"

    def test_failed_with_pending_updates_workflow_to_running(self, engine):
        wf_id = engine.create("wf", [{"name": "a"}, {"name": "b"}])
        steps = engine.get(wf_id)["steps"]
        engine.update_step(wf_id, steps[0]["id"], "failed")
        # b is still pending — workflow should be "running" (progress made + pending work)
        wf = engine.get(wf_id)
        assert wf["status"] == "running"

    def test_failed_with_no_pending_updates_workflow_to_failed(self, engine):
        wf_id = engine.create("wf", [{"name": "a"}])
        step_id = engine.get(wf_id)["steps"][0]["id"]
        engine.update_step(wf_id, step_id, "failed")
        assert engine.get(wf_id)["status"] == "failed"

    def test_some_completed_updates_workflow_to_running(self, engine):
        wf_id = engine.create("wf", [{"name": "a"}, {"name": "b", "depends_on": ["a"]}])
        steps = {s["name"]: s for s in engine.get(wf_id)["steps"]}
        engine.update_step(wf_id, steps["a"]["id"], "completed")
        assert engine.get(wf_id)["status"] == "running"

    def test_returns_true_on_success(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        step_id = engine.get(wf_id)["steps"][0]["id"]
        assert engine.update_step(wf_id, step_id, "running") is True

    def test_all_pending_keeps_workflow_pending(self, engine):
        """Line 390: wf_status='pending' when step set back to pending with nothing running/failed."""
        wf_id = engine.create("wf", [{"name": "a"}, {"name": "b"}])
        steps = engine.get(wf_id)["steps"]
        # Start a, then reset it to pending — all steps pending, workflow stays pending
        engine.update_step(wf_id, steps[0]["id"], "running")
        engine.update_step(wf_id, steps[0]["id"], "pending")
        assert engine.get(wf_id)["status"] == "pending"


class TestDelete:
    def test_nonexistent_returns_false(self, engine):
        assert engine.delete("nonexistent-id") is False

    def test_existing_returns_true(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        assert engine.delete(wf_id) is True

    def test_deleted_workflow_not_found(self, engine):
        wf_id = engine.create("wf", [{"name": "s"}])
        engine.delete(wf_id)
        assert engine.get(wf_id) is None


class TestClose:
    def test_close_owned_connection(self, tmp_path):
        e = WorkflowEngine(tmp_path / "wf.db")
        e.close()  # should not raise

    def test_close_external_connection_no_close(self, tmp_path):
        conn = sqlite3.connect(":memory:")
        e = WorkflowEngine(tmp_path / "wf.db", connection=conn)
        e.close()  # should not close conn
        # conn still usable
        conn.execute("SELECT 1").fetchone()
