"""SQLite ledger of Mullet jobs, their plans, attempts, gate results and events.

The ledger is where an attempt's outcome is decided, so it is the one place
that has to hold the line: a node becomes passed or unverified only through
`Ledger.finish_attempt`, which judges the gate rows already recorded for that
attempt. No method takes an outcome from the caller, and `Ledger.audit`
re-checks the stored evidence so a hand-edited database shows up.
"""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mullet.jobs.paths import jobs_home
from mullet.jobs.types import (
    REQUIRED_GATES,
    AttemptOutcome,
    GateResult,
    GateStatus,
    JobStatus,
    NodeSpec,
    NodeStatus,
    judge,
)

SCHEMA_VERSION = 1

_BUSY_TIMEOUT_S = 30.0
_JOB_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")

# approve() is the only way into APPROVED, so no status lists it as a target.
_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.PLANNING: frozenset({JobStatus.AWAITING_APPROVAL, JobStatus.FAILED, JobStatus.DISCARDED}),
    JobStatus.AWAITING_APPROVAL: frozenset({JobStatus.PLANNING, JobStatus.DISCARDED}),
    JobStatus.APPROVED: frozenset({JobStatus.RUNNING, JobStatus.FAILED, JobStatus.DISCARDED}),
    # A running job is paused (stopped) before it can be discarded.
    JobStatus.RUNNING: frozenset({JobStatus.PAUSED, JobStatus.BLOCKED, JobStatus.DONE, JobStatus.FAILED}),
    JobStatus.PAUSED: frozenset({JobStatus.RUNNING, JobStatus.DISCARDED}),
    JobStatus.BLOCKED: frozenset({JobStatus.RUNNING, JobStatus.DISCARDED}),
    JobStatus.DONE: frozenset({JobStatus.MERGED, JobStatus.DISCARDED}),
    JobStatus.FAILED: frozenset({JobStatus.DISCARDED}),
    JobStatus.MERGED: frozenset(),
    JobStatus.DISCARDED: frozenset(),
}

_COUNTED = (AttemptOutcome.PASSED, AttemptOutcome.UNVERIFIED, AttemptOutcome.FAILED)
_STARTABLE = (NodeStatus.PENDING, NodeStatus.FAILED)
_BLOCKABLE = (NodeStatus.PENDING, NodeStatus.FAILED, NodeStatus.PASSED, NodeStatus.UNVERIFIED)

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        repo_path TEXT NOT NULL,
        goal TEXT NOT NULL,
        base_ref TEXT NOT NULL,
        base_sha TEXT NOT NULL,
        branch TEXT,
        status TEXT NOT NULL,
        max_attempts INTEGER NOT NULL DEFAULT 3,
        profile_json TEXT NOT NULL,
        budget_json TEXT,
        clone_dir TEXT,
        git_dir TEXT,
        error TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        approved_at REAL
    )""",
    """CREATE TABLE IF NOT EXISTS nodes (
        job_id TEXT NOT NULL REFERENCES jobs(id),
        node_id TEXT NOT NULL,
        parent_id TEXT,
        seq INTEGER NOT NULL,
        spec_json TEXT NOT NULL,
        status TEXT NOT NULL,
        start_sha TEXT,
        end_sha TEXT,
        block_reason TEXT,
        updated_at REAL NOT NULL,
        PRIMARY KEY (job_id, node_id),
        FOREIGN KEY (job_id, parent_id) REFERENCES nodes(job_id, node_id)
    )""",
    """CREATE TABLE IF NOT EXISTS attempts (
        id INTEGER PRIMARY KEY,
        job_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        executor TEXT NOT NULL,
        start_sha TEXT NOT NULL,
        end_sha TEXT,
        started_at REAL NOT NULL,
        finished_at REAL,
        outcome TEXT,
        reason TEXT,
        exit_code INTEGER,
        usage_json TEXT,
        claim_json TEXT,
        transcript_path TEXT,
        FOREIGN KEY (job_id, node_id) REFERENCES nodes(job_id, node_id)
    )""",
    """CREATE TABLE IF NOT EXISTS gate_results (
        id INTEGER PRIMARY KEY,
        attempt_id INTEGER NOT NULL REFERENCES attempts(id),
        gate TEXT NOT NULL,
        status TEXT NOT NULL,
        summary TEXT NOT NULL,
        details_json TEXT NOT NULL,
        log_path TEXT,
        duration_s REAL NOT NULL,
        created_at REAL NOT NULL,
        UNIQUE (attempt_id, gate)
    )""",
    """CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES jobs(id),
        ts REAL NOT NULL,
        kind TEXT NOT NULL,
        data_json TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS attempts_by_node ON attempts(job_id, node_id)",
    "CREATE INDEX IF NOT EXISTS events_by_job ON events(job_id, id)",
    """CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
        BEGIN SELECT RAISE(ABORT, 'events are append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
        BEGIN SELECT RAISE(ABORT, 'events are append-only'); END""",
)


@dataclass(frozen=True)
class JobRecord:
    """A job as stored in the ledger."""

    id: str
    repo_path: str
    goal: str
    base_ref: str
    base_sha: str
    branch: str | None
    status: JobStatus
    max_attempts: int
    profile: dict[str, Any]
    budget: dict[str, Any] | None
    clone_dir: str | None
    git_dir: str | None
    error: str | None
    created_at: float
    updated_at: float
    approved_at: float | None


@dataclass(frozen=True)
class NodeRecord:
    """A planned node and where it has got to."""

    job_id: str
    node_id: str
    parent_id: str | None
    seq: int
    spec: NodeSpec
    status: NodeStatus
    start_sha: str | None
    end_sha: str | None
    block_reason: str | None
    updated_at: float


@dataclass(frozen=True)
class AttemptRecord:
    """One executor attempt at a node. `claim` is the executor's own report, not evidence."""

    id: int
    job_id: str
    node_id: str
    executor: str
    start_sha: str
    end_sha: str | None
    started_at: float
    finished_at: float | None
    outcome: AttemptOutcome | None
    reason: str | None
    exit_code: int | None
    usage: dict[str, Any] | None
    claim: dict[str, Any] | None
    transcript_path: str | None


@dataclass(frozen=True)
class EventRecord:
    """One entry in a job's append-only event log."""

    id: int
    job_id: str
    ts: float
    kind: str
    data: dict[str, Any]


def valid_job_id(job_id: str) -> bool:
    """True if `job_id` has the shape `create_job` generates, so it is safe as a directory name."""
    return bool(_JOB_ID_RE.fullmatch(job_id))


def _now() -> float:
    return time.time()


def _dumps(value: dict[str, Any] | None) -> str | None:
    return None if value is None else json.dumps(value)


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def required_gates(budget: dict[str, Any] | None) -> tuple[str, ...]:
    """The gates every attempt in a job must report: G1 to G6, plus G7 when review is on."""
    return (*REQUIRED_GATES, "G7") if (budget or {}).get("review") else REQUIRED_GATES


def _job_required(job_row: sqlite3.Row) -> tuple[str, ...]:
    return required_gates(_loads(job_row["budget_json"]))


def _explain(results: list[GateResult], outcome: AttemptOutcome,
             required: tuple[str, ...] = REQUIRED_GATES) -> str | None:
    if outcome is AttemptOutcome.PASSED:
        return None
    reported = {r.gate for r in results}
    parts = []
    failed = [r.gate for r in results if r.status is GateStatus.FAIL]
    if failed:
        parts.append("failed: " + ", ".join(failed))
    missing = [gate for gate in required if gate not in reported]
    if missing:
        parts.append("never reported: " + ", ".join(missing))
    not_run = [
        r.gate for r in results
        if r.gate in required and r.status in (GateStatus.NOT_RUN, GateStatus.SKIPPED)
    ]
    if not_run:
        parts.append("did not run: " + ", ".join(not_run))
    return "; ".join(parts) or None


class Ledger:
    """The SQLite record of jobs, nodes, attempts, gate results and events."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else jobs_home() / "ledger.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None leaves transactions to _tx(), so each one is explicit.
        self._conn = sqlite3.connect(str(self.path), timeout=_BUSY_TIMEOUT_S, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_schema()
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Jobs

    def create_job(
        self,
        repo_path: str | Path,
        goal: str,
        base_ref: str,
        base_sha: str,
        profile: dict[str, Any],
        budget: dict[str, Any] | None = None,
        max_attempts: int = 3,
    ) -> str:
        """Record a new job in PLANNING and return its id."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        profile_json = json.dumps(profile)
        budget_json = _dumps(budget)
        with self._tx() as conn:
            now = _now()
            for _ in range(5):
                job_id = (
                    datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
                )
                try:
                    conn.execute(
                        "INSERT INTO jobs (id, repo_path, goal, base_ref, base_sha, status, max_attempts,"
                        " profile_json, budget_json, created_at, updated_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (job_id, str(repo_path), goal, base_ref, base_sha, JobStatus.PLANNING.value,
                         max_attempts, profile_json, budget_json, now, now),
                    )
                except sqlite3.IntegrityError:
                    continue
                break
            else:
                raise RuntimeError("could not generate a unique job id")
            self._event(conn, job_id, "job_created", {
                "goal": goal, "base_ref": base_ref, "base_sha": base_sha, "status": JobStatus.PLANNING.value,
            })
        return job_id

    def get_job(self, job_id: str) -> JobRecord:
        """Return one job, or raise KeyError."""
        return _job_record(self._job_row(self._conn, job_id))

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        """Return up to `limit` jobs, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_job_record(row) for row in rows]

    def set_job_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        """Move a job to `status`, or raise ValueError if the move is not allowed.

        `error` describes the new status and replaces any earlier one, so a
        resumed job does not keep the error it was blocked with.
        """
        target = JobStatus(status)
        if target is JobStatus.APPROVED:
            raise ValueError("use approve() to approve a job")
        with self._tx() as conn:
            current = JobStatus(self._job_row(conn, job_id)["status"])
            if target not in _TRANSITIONS[current]:
                allowed = ", ".join(sorted(s.value for s in _TRANSITIONS[current])) or "nothing"
                raise ValueError(
                    f"job {job_id} cannot go from {current.value} to {target.value} (allowed: {allowed})"
                )
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (target.value, error, _now(), job_id),
            )
            self._event(conn, job_id, "job_status", {"from": current.value, "to": target.value, "error": error})

    def approve(self, job_id: str) -> None:
        """Approve a job's current plan. This is the only way into APPROVED."""
        with self._tx() as conn:
            current = JobStatus(self._job_row(conn, job_id)["status"])
            if current is not JobStatus.AWAITING_APPROVAL:
                raise ValueError(f"job {job_id} is {current.value}, not awaiting approval")
            if conn.execute("SELECT COUNT(*) FROM nodes WHERE job_id = ?", (job_id,)).fetchone()[0] == 0:
                raise ValueError(f"job {job_id} has no plan to approve")
            now = _now()
            conn.execute(
                "UPDATE jobs SET status = ?, error = NULL, approved_at = ?, updated_at = ? WHERE id = ?",
                (JobStatus.APPROVED.value, now, now, job_id),
            )
            self._event(conn, job_id, "job_status", {
                "from": current.value, "to": JobStatus.APPROVED.value, "error": None,
            })

    def set_job_paths(self, job_id: str, clone_dir: str | Path, git_dir: str | Path, branch: str) -> None:
        """Record where the job's private clone and git dir live, and its branch."""
        with self._tx() as conn:
            self._job_row(conn, job_id)
            conn.execute(
                "UPDATE jobs SET clone_dir = ?, git_dir = ?, branch = ?, updated_at = ? WHERE id = ?",
                (str(clone_dir), str(git_dir), branch, _now(), job_id),
            )
            self._event(conn, job_id, "job_paths", {
                "clone_dir": str(clone_dir), "git_dir": str(git_dir), "branch": branch,
            })

    # Plan

    def set_plan(self, job_id: str, nodes: list[tuple[NodeSpec, str | None]]) -> None:
        """Replace a job's plan with `nodes`, given as (spec, parent_id) in order.

        Only allowed before approval. A parent must appear earlier in the list.
        """
        if not nodes:
            raise ValueError("a plan needs at least one node")
        seen: set[str] = set()
        for spec, parent_id in nodes:
            if spec.id in seen:
                raise ValueError(f"duplicate node id {spec.id!r}")
            if parent_id is not None and parent_id not in seen:
                raise ValueError(f"node {spec.id!r} names parent {parent_id!r}, which is not an earlier node")
            seen.add(spec.id)
        with self._tx() as conn:
            current = JobStatus(self._job_row(conn, job_id)["status"])
            if current not in (JobStatus.PLANNING, JobStatus.AWAITING_APPROVAL):
                raise ValueError(f"job {job_id} is {current.value}; its plan can only change before approval")
            now = _now()
            conn.execute("DELETE FROM nodes WHERE job_id = ?", (job_id,))
            conn.executemany(
                "INSERT INTO nodes (job_id, node_id, parent_id, seq, spec_json, status, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (job_id, spec.id, parent_id, seq, spec.model_dump_json(), NodeStatus.PENDING.value, now)
                    for seq, (spec, parent_id) in enumerate(nodes)
                ],
            )
            conn.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (now, job_id))
            self._event(conn, job_id, "plan_set", {"nodes": [spec.id for spec, _ in nodes]})

    def nodes(self, job_id: str) -> list[NodeRecord]:
        """Return a job's nodes in plan order."""
        rows = self._conn.execute("SELECT * FROM nodes WHERE job_id = ? ORDER BY seq", (job_id,)).fetchall()
        return [_node_record(row) for row in rows]

    def get_node(self, job_id: str, node_id: str) -> NodeRecord:
        """Return one node, or raise KeyError."""
        return _node_record(self._node_row(self._conn, job_id, node_id))

    def block_node(self, job_id: str, node_id: str, reason: str) -> None:
        """Mark a node BLOCKED for a person to look at, for example after a rebase conflict."""
        with self._tx() as conn:
            current = NodeStatus(self._node_row(conn, job_id, node_id)["status"])
            if current not in _BLOCKABLE:
                raise ValueError(f"node {node_id} is {current.value} and cannot be blocked")
            self._set_node(conn, job_id, node_id, NodeStatus.BLOCKED, block_reason=reason)

    def unblock_node(self, job_id: str, node_id: str) -> None:
        """Return a BLOCKED node to PENDING once a person has dealt with it.

        Its earlier attempts still count, so a node blocked for running out of
        attempts gets one more before it blocks again.
        """
        with self._tx() as conn:
            current = NodeStatus(self._node_row(conn, job_id, node_id)["status"])
            if current is not NodeStatus.BLOCKED:
                raise ValueError(f"node {node_id} is {current.value}, not blocked")
            self._set_node(conn, job_id, node_id, NodeStatus.PENDING, block_reason=None, end_sha=None)

    # Attempts

    def start_attempt(self, job_id: str, node_id: str, executor: str, start_sha: str) -> int:
        """Open an attempt at a PENDING or FAILED node of a RUNNING job and return its id."""
        with self._tx() as conn:
            job_status = JobStatus(self._job_row(conn, job_id)["status"])
            if job_status is not JobStatus.RUNNING:
                raise ValueError(f"job {job_id} is {job_status.value}, not running")
            node_status = NodeStatus(self._node_row(conn, job_id, node_id)["status"])
            if node_status not in _STARTABLE:
                raise ValueError(f"node {node_id} is {node_status.value} and cannot be attempted")
            cur = conn.execute(
                "INSERT INTO attempts (job_id, node_id, executor, start_sha, started_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, node_id, executor, start_sha, _now()),
            )
            attempt_id = int(cur.lastrowid)
            self._event(conn, job_id, "attempt_started", {
                "attempt_id": attempt_id, "node_id": node_id, "executor": executor, "start_sha": start_sha,
            })
            self._set_node(conn, job_id, node_id, NodeStatus.RUNNING, start_sha=start_sha)
        return attempt_id

    def record_executor_result(
        self,
        attempt_id: int,
        exit_code: int | None,
        usage: dict[str, Any] | None,
        claim: dict[str, Any] | None,
        transcript_path: str | None,
    ) -> None:
        """Store what the executor process left behind. The claim is kept as data and judges nothing."""
        with self._tx() as conn:
            attempt = self._open_attempt(conn, attempt_id)
            conn.execute(
                "UPDATE attempts SET exit_code = ?, usage_json = ?, claim_json = ?, transcript_path = ?"
                " WHERE id = ?",
                (exit_code, _dumps(usage), _dumps(claim), transcript_path, attempt_id),
            )
            self._event(conn, attempt["job_id"], "executor_result", {
                "attempt_id": attempt_id, "node_id": attempt["node_id"], "exit_code": exit_code,
            })

    def record_progress(self, attempt_id: int, usage: dict[str, Any]) -> None:
        """Store an open attempt's usage so far (tokens, cost), so a killed attempt keeps what it spent.

        `record_executor_result` replaces it when the attempt ends. Judges nothing.
        """
        with self._tx() as conn:
            self._open_attempt(conn, attempt_id)
            conn.execute("UPDATE attempts SET usage_json = ? WHERE id = ?", (_dumps(usage), attempt_id))

    def record_gate(self, attempt_id: int, result: GateResult) -> None:
        """Store one gate's result for an open attempt. Each gate reports once per attempt."""
        status = GateStatus(result.status)
        details_json = json.dumps(result.details)
        with self._tx() as conn:
            attempt = self._open_attempt(conn, attempt_id)
            if conn.execute(
                "SELECT 1 FROM gate_results WHERE attempt_id = ? AND gate = ?", (attempt_id, result.gate)
            ).fetchone():
                raise ValueError(f"gate {result.gate} already reported for attempt {attempt_id}")
            conn.execute(
                "INSERT INTO gate_results (attempt_id, gate, status, summary, details_json, log_path,"
                " duration_s, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (attempt_id, result.gate, status.value, result.summary, details_json, result.log_path,
                 float(result.duration_s), _now()),
            )
            self._event(conn, attempt["job_id"], "gate_result", {
                "attempt_id": attempt_id, "node_id": attempt["node_id"], "gate": result.gate,
                "status": status.value, "summary": result.summary,
            })

    def judge_attempt(self, attempt_id: int) -> AttemptOutcome:
        """Return what the attempt's recorded gates add up to, without changing anything."""
        attempt = self._attempt_row(self._conn, attempt_id)
        required = _job_required(self._job_row(self._conn, attempt["job_id"]))
        return judge(self._gate_rows(self._conn, attempt_id), required)

    def finish_attempt(self, attempt_id: int, end_sha: str | None = None, *,
                       block_when_out: bool = True) -> AttemptOutcome:
        """Close an attempt with the outcome its gate rows judge to, and move its node to match.

        A passed or unverified attempt needs `end_sha`, the commit Mullet made
        for it. A failed one leaves the node FAILED, or BLOCKED once the job's
        `max_attempts` have been used. With `block_when_out=False` it stays
        FAILED whatever the count: the delegation ladder gives a step one
        attempt per rung and blocks it itself (`block_node`) when none is left.
        """
        with self._tx() as conn:
            attempt = self._open_attempt(conn, attempt_id)
            job_id, node_id = attempt["job_id"], attempt["node_id"]
            job = self._job_row(conn, job_id)
            if JobStatus(job["status"]) is not JobStatus.RUNNING:
                raise ValueError(f"job {job_id} is {job['status']}, not running")
            results = self._gate_rows(conn, attempt_id)
            required = _job_required(job)
            outcome = judge(results, required)
            if outcome is not AttemptOutcome.FAILED and not end_sha:
                raise ValueError(f"attempt {attempt_id} is {outcome.value} but no end_sha was given")
            conn.execute(
                "UPDATE attempts SET outcome = ?, reason = ?, end_sha = ?, finished_at = ? WHERE id = ?",
                (outcome.value, _explain(results, outcome, required), end_sha, _now(), attempt_id),
            )
            self._event(conn, job_id, "attempt_finished", {
                "attempt_id": attempt_id, "node_id": node_id, "outcome": outcome.value, "end_sha": end_sha,
            })
            if outcome is AttemptOutcome.FAILED:
                used = self._attempt_count(conn, job_id, node_id)
                if block_when_out and used >= job["max_attempts"]:
                    self._set_node(
                        conn, job_id, node_id, NodeStatus.BLOCKED,
                        block_reason=f"out of attempts ({used} of {job['max_attempts']} used)",
                    )
                else:
                    self._set_node(conn, job_id, node_id, NodeStatus.FAILED)
            else:
                self._set_node(
                    conn, job_id, node_id, NodeStatus(outcome.value), end_sha=end_sha, block_reason=None
                )
        return outcome

    def abort_attempt(self, attempt_id: int, reason: str) -> None:
        """Throw an attempt away, for example after a kill. It does not use up an attempt."""
        self._close_uncounted(attempt_id, AttemptOutcome.ABORTED, reason)

    def pause_attempt(self, attempt_id: int, reason: str) -> None:
        """Set an attempt aside because a usage limit was hit. It does not use up an attempt."""
        self._close_uncounted(attempt_id, AttemptOutcome.PAUSED, reason)

    def recover_interrupted(self, job_id: str) -> int:
        """Abort every open attempt of a job and return its RUNNING nodes to PENDING.

        Returns the number of attempts aborted.
        """
        with self._tx() as conn:
            self._job_row(conn, job_id)
            open_rows = conn.execute(
                "SELECT id, node_id FROM attempts WHERE job_id = ? AND finished_at IS NULL ORDER BY id",
                (job_id,),
            ).fetchall()
            now = _now()
            for row in open_rows:
                conn.execute(
                    "UPDATE attempts SET outcome = ?, reason = ?, finished_at = ? WHERE id = ?",
                    (AttemptOutcome.ABORTED.value, "interrupted", now, row["id"]),
                )
                self._event(conn, job_id, "attempt_finished", {
                    "attempt_id": row["id"], "node_id": row["node_id"],
                    "outcome": AttemptOutcome.ABORTED.value, "reason": "interrupted",
                })
            running = conn.execute(
                "SELECT node_id FROM nodes WHERE job_id = ? AND status = ? ORDER BY seq",
                (job_id, NodeStatus.RUNNING.value),
            ).fetchall()
            for row in running:
                self._set_node(conn, job_id, row["node_id"], NodeStatus.PENDING)
        return len(open_rows)

    def attempt_count(self, job_id: str, node_id: str) -> int:
        """Attempts at a node that used up a try: passed, unverified or failed."""
        return self._attempt_count(self._conn, job_id, node_id)

    def attempts(self, job_id: str, node_id: str | None = None) -> list[AttemptRecord]:
        """Return a job's attempts, or one node's, oldest first."""
        if node_id is None:
            rows = self._conn.execute("SELECT * FROM attempts WHERE job_id = ? ORDER BY id", (job_id,))
        else:
            rows = self._conn.execute(
                "SELECT * FROM attempts WHERE job_id = ? AND node_id = ? ORDER BY id", (job_id, node_id)
            )
        return [_attempt_record(row) for row in rows.fetchall()]

    def get_attempt(self, attempt_id: int) -> AttemptRecord:
        """Return one attempt, or raise KeyError."""
        return _attempt_record(self._attempt_row(self._conn, attempt_id))

    def gate_results(self, attempt_id: int) -> list[GateResult]:
        """Return an attempt's gate results in the order they were recorded."""
        return self._gate_rows(self._conn, attempt_id)

    # Events and audit

    def add_event(self, job_id: str, kind: str, data: dict[str, Any]) -> int:
        """Append an event to a job's log and return its id."""
        with self._tx() as conn:
            self._job_row(conn, job_id)
            return self._event(conn, job_id, kind, data)

    def events(self, job_id: str, since_id: int = 0) -> list[EventRecord]:
        """Return a job's events with an id above `since_id`, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE job_id = ? AND id > ? ORDER BY id", (job_id, since_id)
        ).fetchall()
        return [
            EventRecord(id=row["id"], job_id=row["job_id"], ts=row["ts"], kind=row["kind"],
                        data=json.loads(row["data_json"]))
            for row in rows
        ]

    def audit(self, job_id: str) -> list[str]:
        """Re-check a job's stored evidence and return every inconsistency found.

        A passed or unverified node must have, as its latest attempt, a
        finished attempt with that same outcome, gate rows that still judge
        to it, and the node's end_sha. Every judged attempt's gate rows must
        still judge to its stored outcome. An empty list means clean.
        """
        problems: list[str] = []
        with self._tx(write=False) as conn:
            required = _job_required(self._job_row(conn, job_id))
            verdicts: dict[int, AttemptOutcome | None] = {}
            latest: dict[str, sqlite3.Row] = {}
            attempts = conn.execute("SELECT * FROM attempts WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
            for attempt in attempts:
                attempt_id = attempt["id"]
                label = f"attempt {attempt_id} on node {attempt['node_id']}"
                latest[attempt["node_id"]] = attempt
                try:
                    verdicts[attempt_id] = judge(self._gate_rows(conn, attempt_id), required)
                except ValueError as exc:
                    verdicts[attempt_id] = None
                    problems.append(f"{label} has an unreadable gate row: {exc}")
                if (attempt["outcome"] is None) != (attempt["finished_at"] is None):
                    problems.append(f"{label} has an outcome without a finish time, or the reverse")
                try:
                    outcome = None if attempt["outcome"] is None else AttemptOutcome(attempt["outcome"])
                except ValueError:
                    problems.append(f"{label} has unknown outcome {attempt['outcome']!r}")
                    continue
                verdict = verdicts[attempt_id]
                if outcome in _COUNTED and verdict is not None and verdict is not outcome:
                    problems.append(
                        f"{label} is recorded as {outcome.value} but its gate rows judge to {verdict.value}"
                    )
                if outcome in (AttemptOutcome.PASSED, AttemptOutcome.UNVERIFIED) and not attempt["end_sha"]:
                    problems.append(f"{label} is {outcome.value} without an end_sha")

            nodes = conn.execute("SELECT * FROM nodes WHERE job_id = ? ORDER BY seq", (job_id,)).fetchall()
            for node in nodes:
                node_id = node["node_id"]
                try:
                    status = NodeStatus(node["status"])
                except ValueError:
                    problems.append(f"node {node_id} has unknown status {node['status']!r}")
                    continue
                last = latest.get(node_id)
                if status is NodeStatus.RUNNING and (last is None or last["finished_at"] is not None):
                    problems.append(f"node {node_id} is running with no open attempt")
                if status not in (NodeStatus.PASSED, NodeStatus.UNVERIFIED):
                    continue
                if last is None:
                    problems.append(f"node {node_id} is {status.value} but has no attempts")
                elif last["finished_at"] is None or last["outcome"] != status.value:
                    problems.append(
                        f"node {node_id} is {status.value} but its latest attempt {last['id']}"
                        f" is {last['outcome'] or 'unfinished'}"
                    )
                elif verdicts.get(last["id"]) is not AttemptOutcome(status.value):
                    problems.append(
                        f"node {node_id} is {status.value} but the gate rows of attempt {last['id']}"
                        " do not judge to it"
                    )
                elif not node["end_sha"] or node["end_sha"] != last["end_sha"]:
                    problems.append(
                        f"node {node_id} end_sha {node['end_sha']!r} does not match attempt {last['id']}"
                        f" end_sha {last['end_sha']!r}"
                    )
        return problems

    # Internals

    @contextmanager
    def _tx(self, write: bool = True) -> Iterator[sqlite3.Connection]:
        # IMMEDIATE takes the write lock up front, so a check and the write it
        # guards can't be split by another process.
        self._conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield self._conn
        except BaseException:
            # SQLite rolls some errors back by itself, and a second ROLLBACK would hide the first error.
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _create_schema(self) -> None:
        with self._tx() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is not None and row["value"] != str(SCHEMA_VERSION):
                raise RuntimeError(
                    f"{self.path} has ledger schema version {row['value']}; this Mullet expects {SCHEMA_VERSION}"
                )
            for statement in _SCHEMA:
                conn.execute(statement)
            if row is None:
                conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def _event(self, conn: sqlite3.Connection, job_id: str, kind: str, data: dict[str, Any]) -> int:
        cur = conn.execute(
            "INSERT INTO events (job_id, ts, kind, data_json) VALUES (?, ?, ?, ?)",
            (job_id, _now(), kind, json.dumps(data)),
        )
        return int(cur.lastrowid)

    def _set_node(
        self, conn: sqlite3.Connection, job_id: str, node_id: str, status: NodeStatus, **fields: Any
    ) -> None:
        previous = self._node_row(conn, job_id, node_id)["status"]
        # Column names come from this module's own keyword arguments, never from a caller.
        columns = {"status": status.value, "updated_at": _now(), **fields}
        conn.execute(
            "UPDATE nodes SET " + ", ".join(f"{name} = ?" for name in columns)
            + " WHERE job_id = ? AND node_id = ?",
            (*columns.values(), job_id, node_id),
        )
        data: dict[str, Any] = {"node_id": node_id, "from": previous, "to": status.value}
        if fields.get("block_reason"):
            data["reason"] = fields["block_reason"]
        self._event(conn, job_id, "node_status", data)

    def _close_uncounted(self, attempt_id: int, outcome: AttemptOutcome, reason: str) -> None:
        with self._tx() as conn:
            attempt = self._open_attempt(conn, attempt_id)
            conn.execute(
                "UPDATE attempts SET outcome = ?, reason = ?, finished_at = ? WHERE id = ?",
                (outcome.value, reason, _now(), attempt_id),
            )
            self._event(conn, attempt["job_id"], "attempt_finished", {
                "attempt_id": attempt_id, "node_id": attempt["node_id"], "outcome": outcome.value, "reason": reason,
            })
            self._set_node(conn, attempt["job_id"], attempt["node_id"], NodeStatus.PENDING)

    @staticmethod
    def _job_row(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"no job {job_id!r}")
        return row

    @staticmethod
    def _node_row(conn: sqlite3.Connection, job_id: str, node_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM nodes WHERE job_id = ? AND node_id = ?", (job_id, node_id)
        ).fetchone()
        if row is None:
            raise KeyError(f"no node {node_id!r} in job {job_id!r}")
        return row

    @staticmethod
    def _attempt_row(conn: sqlite3.Connection, attempt_id: int) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()
        if row is None:
            raise KeyError(f"no attempt {attempt_id}")
        return row

    def _open_attempt(self, conn: sqlite3.Connection, attempt_id: int) -> sqlite3.Row:
        row = self._attempt_row(conn, attempt_id)
        if row["finished_at"] is not None:
            raise ValueError(f"attempt {attempt_id} is already finished ({row['outcome']})")
        return row

    @staticmethod
    def _gate_rows(conn: sqlite3.Connection, attempt_id: int) -> list[GateResult]:
        rows = conn.execute(
            "SELECT * FROM gate_results WHERE attempt_id = ? ORDER BY id", (attempt_id,)
        ).fetchall()
        return [
            GateResult(
                gate=row["gate"],
                status=GateStatus(row["status"]),
                summary=row["summary"],
                details=json.loads(row["details_json"]),
                log_path=row["log_path"],
                duration_s=row["duration_s"],
            )
            for row in rows
        ]

    @staticmethod
    def _attempt_count(conn: sqlite3.Connection, job_id: str, node_id: str) -> int:
        return conn.execute(
            "SELECT COUNT(*) FROM attempts WHERE job_id = ? AND node_id = ? AND outcome IN (?, ?, ?)",
            (job_id, node_id, *(o.value for o in _COUNTED)),
        ).fetchone()[0]


def _job_record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        repo_path=row["repo_path"],
        goal=row["goal"],
        base_ref=row["base_ref"],
        base_sha=row["base_sha"],
        branch=row["branch"],
        status=JobStatus(row["status"]),
        max_attempts=row["max_attempts"],
        profile=json.loads(row["profile_json"]),
        budget=_loads(row["budget_json"]),
        clone_dir=row["clone_dir"],
        git_dir=row["git_dir"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        approved_at=row["approved_at"],
    )


def _node_record(row: sqlite3.Row) -> NodeRecord:
    return NodeRecord(
        job_id=row["job_id"],
        node_id=row["node_id"],
        parent_id=row["parent_id"],
        seq=row["seq"],
        spec=NodeSpec.model_validate_json(row["spec_json"]),
        status=NodeStatus(row["status"]),
        start_sha=row["start_sha"],
        end_sha=row["end_sha"],
        block_reason=row["block_reason"],
        updated_at=row["updated_at"],
    )


def _attempt_record(row: sqlite3.Row) -> AttemptRecord:
    return AttemptRecord(
        id=row["id"],
        job_id=row["job_id"],
        node_id=row["node_id"],
        executor=row["executor"],
        start_sha=row["start_sha"],
        end_sha=row["end_sha"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        outcome=None if row["outcome"] is None else AttemptOutcome(row["outcome"]),
        reason=row["reason"],
        exit_code=row["exit_code"],
        usage=_loads(row["usage_json"]),
        claim=_loads(row["claim_json"]),
        transcript_path=row["transcript_path"],
    )
