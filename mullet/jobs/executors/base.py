"""The executor interface.

An executor gets a task and a work tree and changes files in it. It does not
report which files it changed or whether it succeeded: Mullet reads the work
tree itself and the gates decide. Whatever the executor says about its own
work is kept as `claim`, for the report, and never used as a result.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from mullet.jobs.types import NodeSpec

if TYPE_CHECKING:
    from mullet.jobs.settings import JobSettings

# A worker silent for this long is stopped, unless the settings say otherwise
# or the profile's checks take longer (see `idle_seconds`).
MIN_IDLE_S = 15 * 60.0
# A worker's own run of the slowest check prints nothing until it ends.
IDLE_CHECK_FACTOR = 1.5
DEFAULT_GRACE_S = 60.0


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
    """Limits for one attempt.

    `idle_s` stops a worker that has printed nothing for that long (None: no
    limit); `grace_s` is how long a worker may take to exit once it has given
    its final reply.
    """

    max_turns: int = 60
    wall_clock_s: float = 3600.0
    idle_s: float | None = MIN_IDLE_S
    grace_s: float = DEFAULT_GRACE_S


def idle_seconds(idle_minutes: float | None, check_timeouts: Iterable[float]) -> float:
    """The idle limit: `idle_minutes` when set, else the larger of 15 minutes and
    1.5 times the longest check timeout, since a worker running that check prints nothing."""
    if idle_minutes is not None:
        return float(idle_minutes) * 60
    return max(MIN_IDLE_S, IDLE_CHECK_FACTOR * max(check_timeouts, default=0.0))


def with_worker_limits(budget: Budget, settings: JobSettings, check_timeouts: Iterable[float]) -> Budget:
    """`budget` with the idle limit and grace window from the ``jobs:`` settings."""
    return replace(budget, idle_s=idle_seconds(settings.idle_minutes, check_timeouts),
                   grace_s=float(settings.grace_seconds))


def idle_text(idle_s: float) -> str:
    """How an idle stop is described: "no output for 15 minutes"."""
    amount, unit = (round(idle_s / 60, 1), "minute") if idle_s >= 60 else (round(idle_s, 1), "second")
    return f"no output for {amount:g} {unit}{'' if amount == 1 else 's'}"


@dataclass
class Watch:
    """When a worker last printed anything, and when it gave its final reply.

    Reader threads call `output` for every line and `finished` on the final
    reply; the executor's wait loop asks `verdict` whether to stop the worker.
    """

    idle_s: float | None
    grace_s: float
    last_output: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    _file_size: int = -1

    def output(self, now: float | None = None) -> None:
        """The worker printed a line."""
        self.last_output = time.monotonic() if now is None else now

    def finished(self, now: float | None = None) -> None:
        """The worker gave its final reply; from here only the grace window counts."""
        self.finished_at = time.monotonic() if now is None else now

    def resumed(self) -> None:
        """The worker carried on after a reply that looked final."""
        self.finished_at = None

    def poll_file(self, path: Path, now: float | None = None) -> None:
        """Count growth of a log file the worker writes to (its stderr) as output."""
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        if size != self._file_size:
            if self._file_size >= 0:
                self.output(now)
            self._file_size = size

    def verdict(self, now: float | None = None) -> Literal["idle", "grace"] | None:
        """Return "grace" once a finished worker has had its grace window, "idle" once a
        working one has been silent past the idle limit, else None."""
        now = time.monotonic() if now is None else now
        if self.finished_at is not None:
            return "grace" if now - self.finished_at > self.grace_s else None
        if self.idle_s is not None and now - self.last_output > self.idle_s:
            return "idle"
        return None


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
    # The worker printed nothing for the idle limit and was stopped: aborted and
    # a provider error, so it doesn't count against the worker.
    idle_stopped: bool = False


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
