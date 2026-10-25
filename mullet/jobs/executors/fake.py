"""A scripted executor for tests.

Each call to `run` plays the next step: a function that edits the work tree
the way a model might, honestly or not, and returns the claim it would make.
It lets tests prove that the gates catch bad work whatever the claim says.
A step may also return a whole `ExecResult`, to act out a usage limit or a
crash.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mullet.jobs.executors.base import Budget, ExecResult, Task

Step = Callable[[Path, Task], "dict[str, Any] | ExecResult | None"]


class FakeExecutor:
    """Plays scripted steps, one per attempt."""

    name = "fake"

    def __init__(self, steps: list[Step]):
        self._steps = list(steps)
        self.calls: list[Task] = []

    def run(self, task: Task, work: Path, budget: Budget) -> ExecResult:
        """Apply the next step to `work`; it always claims success unless told otherwise."""
        self.calls.append(task)
        if not self._steps:
            return ExecResult(exit_code=1, error="fake executor has no steps left")
        outcome = self._steps.pop(0)(work, task)
        if isinstance(outcome, ExecResult):
            return outcome
        return ExecResult(exit_code=0, claim=outcome if outcome is not None else {"status": "done"})
