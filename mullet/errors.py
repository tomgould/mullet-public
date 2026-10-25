"""Exceptions raised by Mullet's public Python API (``mullet.api``).

``JobError`` is also what the ``mullet job`` command catches and prints, so a
message reads the same in a script and at the command line.
"""

from __future__ import annotations


class MulletError(Exception):
    """Base class for every error Mullet raises on purpose."""


class JobError(MulletError):
    """A job could not be planned, run, stopped or handed back."""


class JobNotFound(JobError):
    """No job has this id."""


class StrategyOff(JobError):
    """The session strategy turns delegation off, so no job is planned."""


class PlanFailed(JobError):
    """The plan was rejected, or the planner could not produce one."""


class NotApproved(JobError):
    """The job has not been approved yet."""


class NotRunnable(JobError):
    """The job is in a state that cannot do what was asked (run, merge, stop...)."""


class RunnerBusy(JobError):
    """A runner is already working on the job."""


class Unavailable(JobError):
    """This machine lacks something the job needs (sandbox, Claude Code, opencode...)."""


class Unverified(JobError):
    """Some nodes could not be fully checked, and handing the branch back was not allowed."""


class SpendLimit(JobError):
    """A spending cap is invalid or does not apply to this job."""
