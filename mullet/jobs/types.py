"""Shared types for Mullet jobs.

A job is one goal run against one repository. It is planned into nodes; each
node is attempted by an executor and then judged by gates that Mullet runs
itself. An attempt's outcome comes from `judge()` and nothing else: what an
executor says about its own work is recorded as a claim, never as a result.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    """Lifecycle of a job."""

    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    RUNNING = "running"
    PAUSED = "paused"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    DISCARDED = "discarded"
    MERGED = "merged"


class NodeStatus(str, enum.Enum):
    """Lifecycle of a node within a job."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    UNVERIFIED = "unverified"  # nothing failed, but a required gate could not run
    FAILED = "failed"  # the last attempt failed; another may follow
    BLOCKED = "blocked"  # out of attempts, or needs a person


class GateStatus(str, enum.Enum):
    """Result of a single gate."""

    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"  # the check could not run, e.g. no tests configured or a red baseline
    SKIPPED = "skipped"  # an earlier gate failed, so this one was not attempted


class AttemptOutcome(str, enum.Enum):
    """Outcome of one executor attempt at a node."""

    PASSED = "passed"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    ABORTED = "aborted"  # killed or interrupted; does not use up an attempt
    PAUSED = "paused"  # usage limit reached; does not use up an attempt


REQUIRED_GATES: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")


@dataclass
class GateResult:
    """What one gate found. `details` must be JSON-serialisable."""

    gate: str
    status: GateStatus
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    log_path: str | None = None
    duration_s: float = 0.0


def judge(results: list[GateResult], required: tuple[str, ...] = REQUIRED_GATES) -> AttemptOutcome:
    """Turn gate results into an attempt outcome.

    Any failure fails the attempt, and so does a required gate that never
    reported, so a crash or a forgotten gate can't pass a node by omission.
    A gate that could not run makes the attempt unverified, never passed.
    """
    if any(r.status is GateStatus.FAIL for r in results):
        return AttemptOutcome.FAILED
    reported = {r.gate for r in results}
    if any(gate not in reported for gate in required):
        return AttemptOutcome.FAILED
    if any(r.status is not GateStatus.PASS for r in results if r.gate in required):
        return AttemptOutcome.UNVERIFIED
    return AttemptOutcome.PASSED


NodeKind = Literal["mechanical", "design", "analysis"]
# "one-way": hard to walk back once merged (a public API, a default users build
# on, a removal, a persisted format). Such a step is always treated as risky.
NodeDoor = Literal["two-way", "one-way"]


class NodeSpec(BaseModel):
    """One unit of work in a job plan, and the limits its gates enforce.

    `after` names nodes that must be finished first; the plan lists nodes so
    that they are (`validate_plan` checks it), so the runner's order holds.
    """

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    goal: str = Field(min_length=1)
    kind: NodeKind = "design"
    door: NodeDoor = "two-way"
    after: list[str] = Field(default_factory=list)
    scope: list[str] = Field(default_factory=lambda: ["**"])
    acceptance: list[str] = Field(default_factory=list)
    allow_protected: list[str] = Field(default_factory=list)
    allow_test_weakening: bool = False
    expects_change: bool = True
    max_files: int = Field(default=50, ge=1)
    max_lines: int = Field(default=2000, ge=1)
