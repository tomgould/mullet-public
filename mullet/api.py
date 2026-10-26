"""Mullet's Python API: drive gated jobs from a script or CI without the CLI.

Everything here is a thin layer over what ``mullet job`` does. The same
checks, the session strategy and the budget envelope apply, and a failure
raises a ``MulletError`` subclass (see ``mullet.errors``) with the CLI's
message instead of exiting or printing::

    from mullet import api

    job = api.plan("add a subtract function with a test", repo=".", plan_file="plan.yaml")
    api.approve(job)
    result = api.run(job)              # runs in this process until done, blocked or paused
    print(api.report(job))
    if result.status == "done" and not result.nodes_unverified:
        api.merge(job)                 # copies the branch into your repo; nothing else changes

Only the names in ``__all__`` are the stable API. Results are frozen
dataclasses of plain values.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from mullet.cli import job_cli
from mullet.errors import (
    JobError, JobNotFound, MulletError, NotApproved, NotRunnable, PlanFailed, RunnerBusy, SpendLimit,
    StrategyOff, Unavailable, Unverified,
)
from mullet.jobs.ladder import load_config, spend_state
from mullet.jobs.ledger import JobRecord, Ledger
from mullet.jobs.report import build_report
from mullet.jobs.runner import JobBusy, JobRunner
from mullet.jobs.types import NodeStatus

__all__ = [
    "JobError", "JobHandle", "JobNotFound", "JobStatusInfo", "JobSummary", "MergeResult", "MulletError",
    "NotApproved", "NotRunnable", "PlanFailed", "RunnerBusy", "SpendLimit", "StrategyOff", "Unavailable",
    "Unverified", "approve", "list_jobs", "merge", "plan", "report", "resume", "run", "status", "stop",
]

_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(frozen=True)
class JobHandle:
    """A planned job. Pass it, or just its `id`, to the other functions."""

    id: str
    notes: str = ""
    open_questions: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True)
class JobSummary:
    """One job's state. `status` is the job's lifecycle word, e.g. "awaiting_approval", "done"."""

    id: str
    status: str
    goal: str
    branch: str | None
    nodes_total: int
    nodes_passed: int
    nodes_unverified: int
    nodes_blocked: int
    spent_usd: float
    error: str | None


@dataclass(frozen=True)
class JobStatusInfo(JobSummary):
    """A job's state, plus whether a runner is working on it now."""

    runner_alive: bool = False


@dataclass(frozen=True)
class MergeResult:
    """The branch handed back into the job's repo."""

    job_id: str
    branch: str
    sha: str
    unverified: tuple[str, ...]
    diffstat: str


def _job_id(job: str | JobHandle) -> str:
    return job.id if isinstance(job, JobHandle) else str(job)


def _translate(fn: _F) -> _F:
    """Turn the ledger's KeyError and ValueError into API errors; leave `MulletError`s alone."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except MulletError:
            raise
        except KeyError as exc:
            raise JobNotFound(str(exc.args[0]) if exc.args else "no such job") from exc
        except ValueError as exc:
            raise JobError(str(exc)) from exc

    return wrapper  # type: ignore[return-value]


@contextmanager
def _ledger() -> Iterator[Ledger]:
    with Ledger() as ledger:
        yield ledger


def _cap(value: float | None) -> float | None:
    if value is not None and not math.isfinite(float(value)):
        raise SpendLimit(f"not an amount: {value!r}")
    return value


def _summary(ledger: Ledger, job: JobRecord) -> dict[str, Any]:
    nodes = ledger.nodes(job.id)

    def count(status: NodeStatus) -> int:
        return sum(1 for n in nodes if n.status is status)

    usages = [a.usage for a in ledger.attempts(job.id)]
    return {
        "id": job.id, "status": job.status.value, "goal": job.goal, "branch": job.branch,
        "nodes_total": len(nodes), "nodes_passed": count(NodeStatus.PASSED),
        "nodes_unverified": count(NodeStatus.UNVERIFIED), "nodes_blocked": count(NodeStatus.BLOCKED),
        "spent_usd": round(spend_state(usages, 0.0, load_config()).spent, 6), "error": job.error,
    }


def _info(ledger: Ledger, job_id: str) -> JobStatusInfo:
    return JobStatusInfo(**_summary(ledger, ledger.get_job(job_id)), runner_alive=job_cli._runner_alive(job_id))


def _make_runner(ledger: Ledger, job_id: str) -> JobRunner:
    from mullet.jobs import reconcile

    return JobRunner(ledger, job_id, snapshot=reconcile.probe_providers)


@_translate
def plan(goal: str, repo: str | Path | None = None, *, ref: str | None = None, max_spend: float | None = None,
         ladder: bool | None = None, executor: str | None = None, plan_file: str | Path | None = None) -> JobHandle:
    """Plan a job on `repo` (default: the current directory) and leave it awaiting approval.

    Without `plan_file`, a read-only Claude Code session writes the task tree. `ref` is the commit or
    branch to start from (default HEAD). `max_spend` caps dollars on paid workers; `ladder` forces the
    delegation ladder on or off (None follows your config); `executor` is "claude" or "opencode".
    A fresh session strategy applies as it does for `mullet job plan`.
    """
    if executor not in (None, "claude", "opencode"):
        raise JobError(f"executor must be 'claude' or 'opencode', not {executor!r}")
    opts = job_cli.PlanOptions(
        goal=goal, repo=str(repo) if repo is not None else ".", ref=ref or "HEAD",
        plan_file=str(plan_file) if plan_file is not None else None, max_spend=_cap(max_spend),
        ladder=ladder, executor=executor)
    with _ledger() as ledger:
        outcome = job_cli.create_plan(ledger, opts, say=lambda text: None)
    return JobHandle(outcome.job_id, outcome.notes, outcome.open_questions)


@_translate
def approve(job: str | JobHandle) -> None:
    """Approve a planned job so it can run."""
    with _ledger() as ledger:
        ledger.approve(_job_id(job))


@_translate
def run(job: str | JobHandle, *, wait: bool = True, detach: bool = False) -> JobStatusInfo:
    """Run an approved, paused or blocked job.

    `wait=True` runs it in this process until it is done, blocked or paused, then returns its status.
    `detach=True` starts the same background runner as `mullet job run` and returns at once.
    """
    job_id = _job_id(job)
    if not (wait or detach):
        raise JobError("run needs wait=True or detach=True")
    with _ledger() as ledger:
        job_cli.prepare_run(ledger, job_id, say=lambda text: None)
        if detach:
            job_cli.start_runner(job_id)
        else:
            try:
                _make_runner(ledger, job_id).run()
            except JobBusy as exc:
                raise RunnerBusy(str(exc)) from exc
        return _info(ledger, job_id)


@_translate
def status(job: str | JobHandle) -> JobStatusInfo:
    """A job's state now."""
    with _ledger() as ledger:
        return _info(ledger, _job_id(job))


@_translate
def report(job: str | JobHandle) -> str:
    """The job's Markdown report."""
    with _ledger() as ledger:
        return build_report(ledger, _job_id(job))


@_translate
def stop(job: str | JobHandle) -> None:
    """Ask a job's runner to stop; the job is paused, not lost."""
    with _ledger() as ledger:
        ledger.get_job(_job_id(job))
    job_cli.stop_runner(_job_id(job))


@_translate
def resume(job: str | JobHandle, *, max_spend: float | None = None) -> JobStatusInfo:
    """Restart a paused or blocked job in the background, optionally with a new spending cap."""
    job_id = _job_id(job)
    with _ledger() as ledger:
        job_cli.prepare_run(ledger, job_id, _cap(max_spend), say=lambda text: None)
        job_cli.start_runner(job_id)
        return _info(ledger, job_id)


@_translate
def merge(job: str | JobHandle, *, allow_unverified: bool = False, branch: str | None = None) -> MergeResult:
    """Copy a finished job's branch into its repo as a new branch; nothing else changes.

    Raises `Unverified` if some nodes could not be fully checked, unless `allow_unverified` is True.
    """
    job_id = _job_id(job)
    with _ledger() as ledger:
        outcome = job_cli.merge_job(ledger, job_id, branch=branch, allow_unverified=allow_unverified,
                                    say=lambda text: None)
    if outcome is None:  # only when a confirm callback declines, and none is passed
        raise NotRunnable("the branch was not handed back")
    return MergeResult(job_id, outcome.branch, outcome.sha, outcome.unverified, outcome.diffstat)


@_translate
def list_jobs(limit: int = 20) -> list[JobSummary]:
    """Recent jobs, newest first."""
    with _ledger() as ledger:
        return [JobSummary(**_summary(ledger, job)) for job in ledger.list_jobs(limit=limit)]
