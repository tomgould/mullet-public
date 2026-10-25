"""Tests for mullet.jobs.ledger."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mullet.jobs.ledger import Ledger, valid_job_id
from mullet.jobs.types import (
    REQUIRED_GATES,
    AttemptOutcome,
    GateResult,
    GateStatus,
    JobStatus,
    NodeSpec,
    NodeStatus,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(db_path: Path):
    with Ledger(db_path) as led:
        yield led


def _spec(node_id: str) -> NodeSpec:
    return NodeSpec(id=node_id, goal=f"do {node_id}")


def _new_job(ledger: Ledger, max_attempts: int = 3) -> str:
    return ledger.create_job("/repo", "make it better", "main", "base000", {"checks": {"lint": "ruff"}},
                             max_attempts=max_attempts)


def _running_job(ledger: Ledger, nodes: tuple[str, ...] = ("a",), max_attempts: int = 3) -> str:
    job_id = _new_job(ledger, max_attempts)
    ledger.set_plan(job_id, [(_spec(n), None) for n in nodes])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    ledger.set_job_status(job_id, JobStatus.RUNNING)
    return job_id


def _gates(ledger: Ledger, attempt_id: int, overrides: dict[str, GateStatus] | None = None,
           omit: tuple[str, ...] = ()) -> None:
    overrides = overrides or {}
    for gate in REQUIRED_GATES:
        if gate in omit:
            continue
        status = overrides.get(gate, GateStatus.PASS)
        ledger.record_gate(attempt_id, GateResult(gate, status, f"{gate} {status.value}", {"n": 1}))


def _tamper(db_path: Path, sql: str, params: tuple = ()) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class TestJobs:
    def test_create_job_defaults(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        assert valid_job_id(job_id)
        job = ledger.get_job(job_id)
        assert job.status is JobStatus.PLANNING
        assert job.profile == {"checks": {"lint": "ruff"}}
        assert job.budget is None
        assert job.max_attempts == 3
        assert job.approved_at is None
        assert [e.kind for e in ledger.events(job_id)] == ["job_created"]

    def test_valid_job_id_rejects_unsafe_names(self) -> None:
        assert not valid_job_id("../etc")
        assert not valid_job_id("20261024-120000-abc")
        assert not valid_job_id("20261024-120000-abcdef/x")

    def test_create_job_rejects_zero_attempts(self, ledger: Ledger) -> None:
        with pytest.raises(ValueError):
            _new_job(ledger, max_attempts=0)

    def test_get_job_missing_raises_key_error(self, ledger: Ledger) -> None:
        with pytest.raises(KeyError):
            ledger.get_job("20260101-000000-000000")

    def test_list_jobs_newest_first_with_limit(self, ledger: Ledger) -> None:
        ids = [_new_job(ledger) for _ in range(3)]
        assert [j.id for j in ledger.list_jobs()] == list(reversed(ids))
        assert [j.id for j in ledger.list_jobs(limit=2)] == list(reversed(ids))[:2]

    def test_full_lifecycle_to_merged(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        assert ledger.get_job(job_id).approved_at is not None
        ledger.set_job_status(job_id, JobStatus.PAUSED)
        ledger.set_job_status(job_id, JobStatus.RUNNING)
        ledger.set_job_status(job_id, JobStatus.BLOCKED, error="rebase conflict")
        assert ledger.get_job(job_id).error == "rebase conflict"
        ledger.set_job_status(job_id, JobStatus.RUNNING)
        assert ledger.get_job(job_id).error is None
        ledger.set_job_status(job_id, JobStatus.DONE)
        ledger.set_job_status(job_id, JobStatus.MERGED)
        transitions = [(e.data["from"], e.data["to"]) for e in ledger.events(job_id) if e.kind == "job_status"]
        assert transitions[0] == ("planning", "awaiting_approval")
        assert transitions[-1] == ("done", "merged")

    @pytest.mark.parametrize("target", [JobStatus.RUNNING, JobStatus.DONE, JobStatus.MERGED, JobStatus.PLANNING])
    def test_illegal_transition_from_planning(self, ledger: Ledger, target: JobStatus) -> None:
        job_id = _new_job(ledger)
        with pytest.raises(ValueError):
            ledger.set_job_status(job_id, target)
        assert ledger.get_job(job_id).status is JobStatus.PLANNING

    def test_approved_only_through_approve(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        ledger.set_plan(job_id, [(_spec("a"), None)])
        ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
        with pytest.raises(ValueError, match="approve"):
            ledger.set_job_status(job_id, JobStatus.APPROVED)
        ledger.approve(job_id)
        assert ledger.get_job(job_id).status is JobStatus.APPROVED

    def test_approve_requires_awaiting_approval_and_a_plan(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        with pytest.raises(ValueError):
            ledger.approve(job_id)
        ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
        with pytest.raises(ValueError, match="no plan"):
            ledger.approve(job_id)

    def test_running_job_must_pause_before_discard(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        with pytest.raises(ValueError):
            ledger.set_job_status(job_id, JobStatus.DISCARDED)
        ledger.set_job_status(job_id, JobStatus.PAUSED)
        ledger.set_job_status(job_id, JobStatus.DISCARDED)

    def test_terminal_statuses_go_nowhere(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        ledger.set_job_status(job_id, JobStatus.DISCARDED)
        for target in (JobStatus.PLANNING, JobStatus.RUNNING, JobStatus.FAILED):
            with pytest.raises(ValueError):
                ledger.set_job_status(job_id, target)

    def test_set_job_paths(self, ledger: Ledger, tmp_path: Path) -> None:
        job_id = _new_job(ledger)
        ledger.set_job_paths(job_id, tmp_path / "clone", tmp_path / "git", f"mullet/{job_id}")
        job = ledger.get_job(job_id)
        assert job.clone_dir == str(tmp_path / "clone")
        assert job.git_dir == str(tmp_path / "git")
        assert job.branch == f"mullet/{job_id}"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class TestPlan:
    def test_nodes_stored_in_order_with_parents(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        ledger.set_plan(job_id, [(_spec("root"), None), (_spec("child"), "root"), (_spec("leaf"), "child")])
        nodes = ledger.nodes(job_id)
        assert [n.node_id for n in nodes] == ["root", "child", "leaf"]
        assert [n.seq for n in nodes] == [0, 1, 2]
        assert [n.parent_id for n in nodes] == [None, "root", "child"]
        assert all(n.status is NodeStatus.PENDING for n in nodes)
        assert ledger.get_node(job_id, "leaf").spec == _spec("leaf")

    def test_duplicate_node_ids_rejected(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        with pytest.raises(ValueError, match="duplicate"):
            ledger.set_plan(job_id, [(_spec("a"), None), (_spec("a"), None)])

    def test_unknown_or_later_parent_rejected(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        with pytest.raises(ValueError, match="parent"):
            ledger.set_plan(job_id, [(_spec("a"), "ghost")])
        with pytest.raises(ValueError, match="parent"):
            ledger.set_plan(job_id, [(_spec("a"), "b"), (_spec("b"), None)])
        assert ledger.nodes(job_id) == []

    def test_empty_plan_rejected(self, ledger: Ledger) -> None:
        with pytest.raises(ValueError):
            ledger.set_plan(_new_job(ledger), [])

    def test_plan_replaced_while_awaiting_approval(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        ledger.set_plan(job_id, [(_spec("a"), None)])
        ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
        ledger.set_plan(job_id, [(_spec("x"), None), (_spec("y"), "x")])
        assert [n.node_id for n in ledger.nodes(job_id)] == ["x", "y"]

    def test_plan_cannot_change_after_approval(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        ledger.set_plan(job_id, [(_spec("a"), None)])
        ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
        ledger.approve(job_id)
        with pytest.raises(ValueError, match="before approval"):
            ledger.set_plan(job_id, [(_spec("b"), None)])
        assert [n.node_id for n in ledger.nodes(job_id)] == ["a"]

    def test_get_node_missing_raises_key_error(self, ledger: Ledger) -> None:
        with pytest.raises(KeyError):
            ledger.get_node(_new_job(ledger), "nope")


# ---------------------------------------------------------------------------
# Attempts
# ---------------------------------------------------------------------------

class TestAttempts:
    def test_happy_path_to_passed(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "sha-start")
        node = ledger.get_node(job_id, "a")
        assert node.status is NodeStatus.RUNNING
        assert node.start_sha == "sha-start"

        ledger.record_executor_result(attempt_id, 0, {"turns": 4}, {"done": True, "tests": "all green"}, "/t.jsonl")
        _gates(ledger, attempt_id)
        assert ledger.judge_attempt(attempt_id) is AttemptOutcome.PASSED
        assert ledger.get_attempt(attempt_id).outcome is None

        assert ledger.finish_attempt(attempt_id, end_sha="sha-end") is AttemptOutcome.PASSED
        node = ledger.get_node(job_id, "a")
        assert node.status is NodeStatus.PASSED
        assert node.end_sha == "sha-end"
        attempt = ledger.get_attempt(attempt_id)
        assert attempt.outcome is AttemptOutcome.PASSED
        assert attempt.finished_at is not None
        assert attempt.claim == {"done": True, "tests": "all green"}
        assert attempt.usage == {"turns": 4}
        assert ledger.attempt_count(job_id, "a") == 1
        results = ledger.gate_results(attempt_id)
        assert [r.gate for r in results] == list(REQUIRED_GATES)
        assert all(r.status is GateStatus.PASS for r in results)
        assert results[0].details == {"n": 1}
        assert ledger.audit(job_id) == []

    def test_pass_without_end_sha_is_refused_and_leaves_attempt_open(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, attempt_id)
        with pytest.raises(ValueError, match="end_sha"):
            ledger.finish_attempt(attempt_id)
        assert ledger.get_attempt(attempt_id).finished_at is None
        assert ledger.get_node(job_id, "a").status is NodeStatus.RUNNING
        assert ledger.finish_attempt(attempt_id, "s1") is AttemptOutcome.PASSED

    def test_executor_claim_does_not_pass_a_node(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        ledger.record_executor_result(attempt_id, 0, None, {"outcome": "passed", "all_gates": "pass"}, None)
        assert ledger.finish_attempt(attempt_id, "s1") is AttemptOutcome.FAILED
        assert ledger.get_node(job_id, "a").status is NodeStatus.FAILED

    def test_missing_required_gate_fails(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, attempt_id, omit=("G5",))
        assert ledger.finish_attempt(attempt_id) is AttemptOutcome.FAILED
        assert ledger.get_node(job_id, "a").status is NodeStatus.FAILED
        assert "G5" in ledger.get_attempt(attempt_id).reason

    def test_not_run_gate_makes_unverified(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, attempt_id, overrides={"G5": GateStatus.NOT_RUN})
        assert ledger.finish_attempt(attempt_id, "s1") is AttemptOutcome.UNVERIFIED
        node = ledger.get_node(job_id, "a")
        assert node.status is NodeStatus.UNVERIFIED
        assert node.end_sha == "s1"
        assert "G5" in ledger.get_attempt(attempt_id).reason
        assert ledger.audit(job_id) == []

    def test_failures_block_at_max_attempts(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger, max_attempts=2)
        first = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, first, overrides={"G2": GateStatus.FAIL})
        assert ledger.finish_attempt(first) is AttemptOutcome.FAILED
        assert ledger.get_node(job_id, "a").status is NodeStatus.FAILED

        second = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, second, overrides={"G4": GateStatus.FAIL})
        assert ledger.finish_attempt(second) is AttemptOutcome.FAILED
        node = ledger.get_node(job_id, "a")
        assert node.status is NodeStatus.BLOCKED
        assert "2 of 2" in node.block_reason
        assert ledger.attempt_count(job_id, "a") == 2
        with pytest.raises(ValueError, match="blocked"):
            ledger.start_attempt(job_id, "a", "claude_code", "s0")

    def test_abort_and_pause_do_not_count(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger, max_attempts=1)
        aborted = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        ledger.record_gate(aborted, GateResult("G1", GateStatus.PASS, "changed"))
        ledger.abort_attempt(aborted, "killed by stop")
        assert ledger.get_node(job_id, "a").status is NodeStatus.PENDING
        paused = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        ledger.pause_attempt(paused, "usage limit")
        assert ledger.get_node(job_id, "a").status is NodeStatus.PENDING
        assert ledger.attempt_count(job_id, "a") == 0
        assert ledger.get_attempt(aborted).outcome is AttemptOutcome.ABORTED
        assert ledger.get_attempt(paused).outcome is AttemptOutcome.PAUSED

        failed = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        assert ledger.finish_attempt(failed) is AttemptOutcome.FAILED
        assert ledger.get_node(job_id, "a").status is NodeStatus.BLOCKED
        assert ledger.attempt_count(job_id, "a") == 1
        assert ledger.audit(job_id) == []

    def test_cannot_start_on_passed_or_running_node(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger, nodes=("a", "b"))
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, attempt_id)
        ledger.finish_attempt(attempt_id, "s1")
        with pytest.raises(ValueError, match="passed"):
            ledger.start_attempt(job_id, "a", "claude_code", "s1")
        ledger.start_attempt(job_id, "b", "claude_code", "s1")
        with pytest.raises(ValueError, match="running"):
            ledger.start_attempt(job_id, "b", "claude_code", "s1")

    def test_cannot_start_unless_job_running(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        ledger.set_job_status(job_id, JobStatus.PAUSED)
        with pytest.raises(ValueError, match="not running"):
            ledger.start_attempt(job_id, "a", "claude_code", "s0")
        ledger.set_job_status(job_id, JobStatus.RUNNING)
        with pytest.raises(KeyError):
            ledger.start_attempt(job_id, "ghost", "claude_code", "s0")

    def test_finish_refused_when_job_not_running(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, attempt_id)
        ledger.set_job_status(job_id, JobStatus.PAUSED)
        with pytest.raises(ValueError, match="not running"):
            ledger.finish_attempt(attempt_id, "s1")
        ledger.pause_attempt(attempt_id, "stopped")
        assert ledger.get_node(job_id, "a").status is NodeStatus.PENDING

    def test_writes_to_finished_attempt_rejected(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, attempt_id, omit=("G6",))
        ledger.finish_attempt(attempt_id)
        with pytest.raises(ValueError, match="already finished"):
            ledger.record_gate(attempt_id, GateResult("G6", GateStatus.PASS, "late"))
        with pytest.raises(ValueError, match="already finished"):
            ledger.record_executor_result(attempt_id, 0, None, None, None)
        assert ledger.judge_attempt(attempt_id) is AttemptOutcome.FAILED

    def test_duplicate_gate_rejected(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        ledger.record_gate(attempt_id, GateResult("G1", GateStatus.FAIL, "empty diff"))
        with pytest.raises(ValueError, match="already reported"):
            ledger.record_gate(attempt_id, GateResult("G1", GateStatus.PASS, "second try"))
        assert [r.status for r in ledger.gate_results(attempt_id)] == [GateStatus.FAIL]

    def test_finishing_twice_raises(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, attempt_id)
        ledger.finish_attempt(attempt_id, "s1")
        with pytest.raises(ValueError, match="already finished"):
            ledger.finish_attempt(attempt_id, "s1")
        with pytest.raises(ValueError, match="already finished"):
            ledger.abort_attempt(attempt_id, "late")
        assert ledger.get_node(job_id, "a").status is NodeStatus.PASSED

    def test_recover_interrupted(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger, nodes=("a", "b", "c"))
        first = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        second = ledger.start_attempt(job_id, "b", "claude_code", "s0")
        assert ledger.recover_interrupted(job_id) == 2
        for attempt_id in (first, second):
            attempt = ledger.get_attempt(attempt_id)
            assert attempt.outcome is AttemptOutcome.ABORTED
            assert attempt.reason == "interrupted"
        assert [n.status for n in ledger.nodes(job_id)] == [NodeStatus.PENDING] * 3
        assert ledger.attempt_count(job_id, "a") == 0
        assert ledger.recover_interrupted(job_id) == 0
        assert ledger.audit(job_id) == []

    def test_block_and_unblock_node(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger, nodes=("a", "b"))
        ledger.block_node(job_id, "a", "rebase conflict in src/x.py")
        node = ledger.get_node(job_id, "a")
        assert node.status is NodeStatus.BLOCKED
        assert node.block_reason == "rebase conflict in src/x.py"
        with pytest.raises(ValueError):
            ledger.start_attempt(job_id, "a", "claude_code", "s0")
        ledger.start_attempt(job_id, "b", "claude_code", "s0")
        with pytest.raises(ValueError, match="cannot be blocked"):
            ledger.block_node(job_id, "b", "nope")
        ledger.unblock_node(job_id, "a")
        node = ledger.get_node(job_id, "a")
        assert node.status is NodeStatus.PENDING
        assert node.block_reason is None
        with pytest.raises(ValueError, match="not blocked"):
            ledger.unblock_node(job_id, "a")
        ledger.start_attempt(job_id, "a", "claude_code", "s0")

    def test_attempts_listing(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger, nodes=("a", "b"))
        a1 = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        ledger.abort_attempt(a1, "x")
        b1 = ledger.start_attempt(job_id, "b", "opencode", "s0")
        assert [a.id for a in ledger.attempts(job_id)] == [a1, b1]
        assert [a.id for a in ledger.attempts(job_id, "b")] == [b1]
        assert ledger.attempts(job_id, "b")[0].executor == "opencode"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_events_written_automatically(self, ledger: Ledger) -> None:
        job_id = _running_job(ledger)
        attempt_id = ledger.start_attempt(job_id, "a", "claude_code", "s0")
        _gates(ledger, attempt_id)
        ledger.finish_attempt(attempt_id, "s1")
        kinds = [e.kind for e in ledger.events(job_id)]
        assert kinds[:2] == ["job_created", "plan_set"]
        assert kinds.count("job_status") == 3
        assert kinds.count("attempt_started") == 1
        assert kinds.count("gate_result") == len(REQUIRED_GATES)
        assert kinds.count("attempt_finished") == 1
        finished = next(e for e in ledger.events(job_id) if e.kind == "attempt_finished")
        assert finished.data["outcome"] == "passed"
        node_moves = [(e.data["from"], e.data["to"]) for e in ledger.events(job_id) if e.kind == "node_status"]
        assert node_moves == [("pending", "running"), ("running", "passed")]

    def test_add_event_and_since_id(self, ledger: Ledger) -> None:
        job_id = _new_job(ledger)
        first = ledger.add_event(job_id, "note", {"text": "hello"})
        second = ledger.add_event(job_id, "note", {"text": "again"})
        assert second > first
        assert [e.data["text"] for e in ledger.events(job_id, since_id=first)] == ["again"]
        with pytest.raises(KeyError):
            ledger.add_event("20260101-000000-000000", "note", {})

    def test_events_are_append_only(self, ledger: Ledger, db_path: Path) -> None:
        job_id = _new_job(ledger)
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            _tamper(db_path, "UPDATE events SET kind = 'forged' WHERE job_id = ?", (job_id,))
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            _tamper(db_path, "DELETE FROM events WHERE job_id = ?", (job_id,))


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _mixed_run(ledger: Ledger) -> str:
    job_id = _running_job(ledger, nodes=("a", "b", "c"))
    passed = ledger.start_attempt(job_id, "a", "claude_code", "s0")
    _gates(ledger, passed)
    ledger.finish_attempt(passed, "s1")
    failed = ledger.start_attempt(job_id, "b", "claude_code", "s1")
    _gates(ledger, failed, overrides={"G3": GateStatus.FAIL})
    ledger.finish_attempt(failed)
    unverified = ledger.start_attempt(job_id, "c", "claude_code", "s1")
    _gates(ledger, unverified, overrides={"G5": GateStatus.NOT_RUN})
    ledger.finish_attempt(unverified, "s2")
    return job_id


class TestAudit:
    def test_clean_after_normal_run(self, ledger: Ledger) -> None:
        assert ledger.audit(_mixed_run(ledger)) == []

    def test_node_set_passed_by_raw_sql(self, ledger: Ledger, db_path: Path) -> None:
        job_id = _running_job(ledger, nodes=("a",))
        _tamper(db_path, "UPDATE nodes SET status = 'passed', end_sha = 'x' WHERE job_id = ?", (job_id,))
        problems = ledger.audit(job_id)
        assert len(problems) == 1
        assert "node a is passed" in problems[0]

    def test_failed_node_flipped_to_passed(self, ledger: Ledger, db_path: Path) -> None:
        job_id = _mixed_run(ledger)
        _tamper(db_path, "UPDATE nodes SET status = 'passed', end_sha = 's9' WHERE job_id = ? AND node_id = 'b'",
                (job_id,))
        assert any("node b" in p for p in ledger.audit(job_id))

    def test_attempt_outcome_forged(self, ledger: Ledger, db_path: Path) -> None:
        job_id = _mixed_run(ledger)
        _tamper(db_path, "UPDATE attempts SET outcome = 'passed', end_sha = 's9' WHERE node_id = 'b'")
        _tamper(db_path, "UPDATE nodes SET status = 'passed', end_sha = 's9' WHERE node_id = 'b'")
        problems = ledger.audit(job_id)
        assert any("recorded as passed but its gate rows judge to failed" in p for p in problems)
        assert any("node b is passed" in p for p in problems)

    def test_gate_row_deleted(self, ledger: Ledger, db_path: Path) -> None:
        job_id = _mixed_run(ledger)
        attempt_id = ledger.attempts(job_id, "a")[0].id
        _tamper(db_path, "DELETE FROM gate_results WHERE attempt_id = ? AND gate = 'G5'", (attempt_id,))
        problems = ledger.audit(job_id)
        assert any(f"attempt {attempt_id}" in p and "judge to failed" in p for p in problems)
        assert any("node a" in p for p in problems)

    def test_gate_status_upgraded(self, ledger: Ledger, db_path: Path) -> None:
        job_id = _mixed_run(ledger)
        _tamper(db_path, "UPDATE gate_results SET status = 'pass' WHERE status = 'not_run'")
        assert any("node c" in p for p in ledger.audit(job_id))

    def test_end_sha_mismatch(self, ledger: Ledger, db_path: Path) -> None:
        job_id = _mixed_run(ledger)
        _tamper(db_path, "UPDATE nodes SET end_sha = 'other' WHERE job_id = ? AND node_id = 'a'", (job_id,))
        problems = ledger.audit(job_id)
        assert len(problems) == 1
        assert "end_sha" in problems[0]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class TestStorage:
    def test_second_instance_sees_same_data(self, ledger: Ledger, db_path: Path) -> None:
        job_id = _mixed_run(ledger)
        with Ledger(db_path) as other:
            assert other.get_job(job_id) == ledger.get_job(job_id)
            assert other.nodes(job_id) == ledger.nodes(job_id)
            assert other.attempts(job_id) == ledger.attempts(job_id)
            assert other.audit(job_id) == []
            other.set_job_status(job_id, JobStatus.DONE)
        assert ledger.get_job(job_id).status is JobStatus.DONE

    def test_wal_and_foreign_keys(self, ledger: Ledger, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()
        assert ledger._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_newer_schema_version_refused(self, db_path: Path) -> None:
        Ledger(db_path).close()
        _tamper(db_path, "UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        with pytest.raises(RuntimeError, match="schema version 99"):
            Ledger(db_path)

    def test_default_path_under_jobs_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
        with Ledger() as led:
            assert led.path == (tmp_path / "jobs" / "ledger.db").resolve()
        assert (tmp_path / "jobs" / "ledger.db").exists()
