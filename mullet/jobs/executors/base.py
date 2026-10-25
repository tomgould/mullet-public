"""The executor interface.

An executor gets a task and a work tree and changes files in it. It does not
report which files it changed or whether it succeeded: Mullet reads the work
tree itself and the gates decide. Whatever the executor says about its own
work is kept as `claim`, for the report, and never used as a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mullet.jobs.types import NodeSpec


@dataclass
class NodeContext:
    """What Mullet knows about the job around a node, for the executor's brief.

    The runner gathers it from the ledger and the clone; `mullet.jobs.context`
    turns it into the brief and picks files to pre-load.
    """

    parent_goal: str | None = None
    plan_notes: str = ""
    history: str = ""  # what earlier nodes changed (git log --stat), trimmed
    files: list[str] = field(default_factory=list)  # tracked files where the node starts
    checks: dict[str, str] = field(default_factory=dict)  # check name -> profile command
    exclude: list[str] = field(default_factory=list)  # protected and copy_in globs, never pre-loaded


@dataclass
class Task:
    """What an executor is asked to do in one attempt."""

    job_goal: str
    node: NodeSpec
    feedback: list[str] = field(default_factory=list)  # gate failures from earlier attempts
    context: NodeContext | None = None
    label: str = ""  # job and node, for session titles


@dataclass
class Budget:
    """Limits for one attempt."""

    max_turns: int = 60
    wall_clock_s: float = 3600.0


@dataclass
class ExecResult:
    """How an attempt ended, as far as the executor process is concerned."""

    exit_code: int | None
    transcript_path: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    claim: dict[str, Any] | None = None
    split_request: list[dict[str, Any]] | None = None
    paused_until: float | None = None  # epoch seconds when a usage limit resets
    aborted: bool = False
    error: str | None = None
    session_id: str | None = None  # set when the run can be continued with `resume`
    # The provider failed the run (rate limit, spending cap, bad key, unknown
    # model, unreachable), not the worker: the ladder moves up without counting it.
    provider_error: bool = False


class Executor(Protocol):
    """Anything that can attempt a task in a work tree.

    An executor may also offer `resume(task, work, budget, session_id)`: carry
    on an earlier session with the gate feedback in `task.feedback[-1]`, on
    the work tree that session left behind. The runner uses it, when present,
    to have a worker fix its own failed attempt instead of starting over.
    """

    name: str

    def run(self, task: Task, work: Path, budget: Budget) -> ExecResult:
        """Attempt `task` by editing files under `work`."""
        ...


class ExecutorUnavailable(RuntimeError):
    """The executor can't be used for this job on this machine."""
