"""Tests for mullet.harness.workflows.api.WorkflowAPI."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from mullet.harness.workflows.api import WorkflowAPI, WorkflowNotFoundError
from mullet.harness.workflows.store import WorkflowStore

pytestmark = pytest.mark.harness


@pytest.fixture
def api(tmp_path: Path) -> WorkflowAPI:
    """Yield a WorkflowAPI backed by a fresh tmp-path SQLite db."""
    store = WorkflowStore(db_path=tmp_path / "wf.db")
    return WorkflowAPI(store=store)


def test_create_run_returns_workflow_run_with_pending_status(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    assert run.workflow_type == "agent.run"
    assert run.status == "pending"
    assert run.run_id  # uuid hex
    assert run.created_at <= run.updated_at
    assert run.metadata == {}
    assert run.output is None
    assert run.parent_run_id is None


def test_create_run_persists(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run", metadata={"k": "v"})
    listed = api.list_runs()
    assert len(listed) == 1
    assert listed[0].run_id == run.run_id
    assert listed[0].metadata == {"k": "v"}


def test_append_event_increments_sequence(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    e1 = api.append_event(run.run_id, "step.completed")
    e2 = api.append_event(run.run_id, "step.completed")
    e3 = api.append_event(run.run_id, "step.completed")
    assert (e1.sequence, e2.sequence, e3.sequence) == (1, 2, 3)


def test_append_event_unknown_run_raises(api: WorkflowAPI) -> None:
    with pytest.raises(WorkflowNotFoundError):
        api.append_event("does-not-exist", "started")


def test_append_event_updates_status_for_started(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    assert run.status == "pending"
    api.append_event(run.run_id, "started")
    assert api.get_run(run.run_id).status == "running"  # type: ignore[union-attr]


def test_append_event_updates_status_for_completed(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    api.append_event(run.run_id, "started")
    api.append_event(run.run_id, "completed")
    assert api.get_run(run.run_id).status == "completed"  # type: ignore[union-attr]


def test_append_event_updates_status_for_error_and_failed(api: WorkflowAPI) -> None:
    run_a = api.create_run("agent.run")
    api.append_event(run_a.run_id, "error")
    assert api.get_run(run_a.run_id).status == "failed"  # type: ignore[union-attr]

    run_b = api.create_run("agent.run")
    api.append_event(run_b.run_id, "failed")
    assert api.get_run(run_b.run_id).status == "failed"  # type: ignore[union-attr]


def test_append_event_unknown_type_does_not_change_status(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    api.append_event(run.run_id, "step.completed")
    assert api.get_run(run.run_id).status == "pending"  # type: ignore[union-attr]


def test_append_message_increments_sequence_per_run(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    api.append_event(run.run_id, "started")
    api.append_event(run.run_id, "step.completed")
    m1 = api.append_message(run.run_id, "user", "hi")
    m2 = api.append_message(run.run_id, "assistant", "hello")
    # Messages keep their own sequence, independent of events.
    assert (m1.sequence, m2.sequence) == (1, 2)
    events = api.get_events(run.run_id)
    assert [e.sequence for e in events] == [1, 2]


def test_get_messages_ordered_by_sequence(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    api.append_message(run.run_id, "user", "first")
    api.append_message(run.run_id, "assistant", "second")
    api.append_message(run.run_id, "user", "third")
    msgs = api.get_messages(run.run_id)
    assert [m.sequence for m in msgs] == [1, 2, 3]
    assert [m.content for m in msgs] == ["first", "second", "third"]


def test_get_events_ordered_by_sequence(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    api.append_event(run.run_id, "started")
    api.append_event(run.run_id, "tool.call", payload={"name": "Bash"})
    api.append_event(run.run_id, "completed")
    events = api.get_events(run.run_id)
    assert [e.sequence for e in events] == [1, 2, 3]
    assert [e.event_type for e in events] == ["started", "tool.call", "completed"]
    assert events[1].payload == {"name": "Bash"}


def test_list_runs_newest_first(api: WorkflowAPI) -> None:
    r1 = api.create_run("agent.run")
    r2 = api.create_run("agent.run")
    r3 = api.create_run("agent.run")
    listed = api.list_runs()
    assert [r.run_id for r in listed] == [r3.run_id, r2.run_id, r1.run_id]


def test_list_runs_filter_by_status_and_type(api: WorkflowAPI) -> None:
    r_a = api.create_run("agent.run")
    r_b = api.create_run("delegate.parallel")
    r_c = api.create_run("agent.run")
    api.append_event(r_a.run_id, "started")  # status -> running
    api.append_event(r_c.run_id, "completed")

    by_type = api.list_runs(workflow_type="agent.run")
    assert {r.run_id for r in by_type} == {r_a.run_id, r_c.run_id}

    by_status = api.list_runs(status="completed")
    assert [r.run_id for r in by_status] == [r_c.run_id]

    by_both = api.list_runs(workflow_type="delegate.parallel", status="pending")
    assert [r.run_id for r in by_both] == [r_b.run_id]


def test_list_runs_limit(api: WorkflowAPI) -> None:
    for _ in range(5):
        api.create_run("agent.run")
    assert len(api.list_runs(limit=2)) == 2


def test_replay_events_reconstructs_status(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    api.append_event(run.run_id, "started")
    api.append_event(run.run_id, "step.completed")
    api.append_event(run.run_id, "completed")

    # Manually corrupt the materialised status.
    conn = api.store._conn()
    conn.execute(
        "UPDATE workflow_runs SET status = ? WHERE run_id = ?",
        ("pending", run.run_id),
    )
    assert api.get_run(run.run_id).status == "pending"  # type: ignore[union-attr]

    replayed = api.replay_events(run.run_id)
    assert replayed.status == "completed"
    assert api.get_run(run.run_id).status == "completed"  # type: ignore[union-attr]


def test_replay_events_no_events_stays_pending(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    replayed = api.replay_events(run.run_id)
    assert replayed.status == "pending"


def test_update_run_changes_status_and_metadata(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run", metadata={"a": 1})
    updated = api.update_run(
        run.run_id,
        status="running",
        metadata={"a": 1, "b": 2},
        output={"final": True},
    )
    assert updated.status == "running"
    assert updated.metadata == {"a": 1, "b": 2}
    assert updated.output == {"final": True}
    assert updated.updated_at >= run.updated_at

    fetched = api.get_run(run.run_id)
    assert fetched is not None
    assert fetched.status == "running"
    assert fetched.output == {"final": True}


def test_update_run_unknown_run_raises(api: WorkflowAPI) -> None:
    with pytest.raises(WorkflowNotFoundError):
        api.update_run("missing", status="running")


def test_cancel_appends_event_and_sets_status(api: WorkflowAPI) -> None:
    run = api.create_run("agent.run")
    api.append_event(run.run_id, "started")
    cancelled = api.cancel(run.run_id)
    assert cancelled.status == "cancelled"

    events = api.get_events(run.run_id)
    assert events[-1].event_type == "cancelled"


def test_parent_run_id_persists(api: WorkflowAPI) -> None:
    parent = api.create_run("agent.run")
    child = api.create_run("delegate.parallel", parent_run_id=parent.run_id)
    fetched = api.get_run(child.run_id)
    assert fetched is not None
    assert fetched.parent_run_id == parent.run_id


def test_append_message_unknown_run_raises(api: WorkflowAPI) -> None:
    with pytest.raises(WorkflowNotFoundError):
        api.append_message("nope", "user", "hi")


def test_get_run_missing_returns_none(api: WorkflowAPI) -> None:
    assert api.get_run("does-not-exist") is None


def test_concurrent_appends_from_threads(tmp_path: Path) -> None:
    """Five threads each append 10 events; sequences must be contiguous 1..50."""
    store = WorkflowStore(db_path=tmp_path / "concurrent.db")
    api = WorkflowAPI(store=store)
    run = api.create_run("agent.run")

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(10):
                api.append_event(run.run_id, "step.completed")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    events = api.get_events(run.run_id)
    sequences = sorted(e.sequence for e in events)
    assert sequences == list(range(1, 51))
    assert len(events) == 50
