"""Tests for a worker's idle limit and grace window, and the limits the runner passes on."""

import pytest

from mullet.jobs import sandbox
from mullet.jobs.executors.base import (
    MIN_IDLE_S, Budget, ExecResult, Watch, idle_seconds, idle_text, with_worker_limits,
)
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.runner import JobRunner
from mullet.jobs.settings import JobSettings
from mullet.jobs.types import JobStatus, NodeSpec

from tests.jobs_helpers import good_change
from tests import test_jobs_runner
from tests.test_jobs_runner import add_docs, make_job

env = test_jobs_runner.env  # the fixture: a source repo, a jobs dir and a ledger

needs_sandbox = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)


# Limits


def test_explicit_idle_minutes_win():
    assert idle_seconds(2.5, [3600, 900]) == 150.0


def test_derived_idle_is_at_least_fifteen_minutes():
    assert idle_seconds(None, [60, 120]) == MIN_IDLE_S == 900.0
    assert idle_seconds(None, []) == MIN_IDLE_S


def test_derived_idle_covers_one_and_a_half_times_the_longest_check():
    # A worker's own long test run prints nothing while it runs.
    assert idle_seconds(None, [900, 3600]) == 5400.0


def test_worker_limits_keep_turns_and_wall_clock():
    settings = JobSettings(idle_minutes=None, grace_seconds=12.0)
    budget = with_worker_limits(Budget(max_turns=7, wall_clock_s=99.0), settings, [600, 800])
    assert (budget.max_turns, budget.wall_clock_s) == (7, 99.0)
    assert budget.idle_s == 1200.0 and budget.grace_s == 12.0


def test_default_budget_has_an_idle_limit_and_a_grace_window():
    budget = Budget()
    assert budget.idle_s == MIN_IDLE_S and budget.grace_s == 60.0


def test_idle_text_names_minutes_or_seconds():
    assert idle_text(900.0) == "no output for 15 minutes"
    assert idle_text(90.0) == "no output for 1.5 minutes"
    assert idle_text(2.0) == "no output for 2 seconds"


def test_watch_stops_a_silent_worker_and_lets_a_finished_one_exit():
    watch = Watch(idle_s=10.0, grace_s=5.0)
    start = watch.last_output
    assert watch.verdict(start + 9) is None
    assert watch.verdict(start + 11) == "idle"
    watch.output(start + 11)
    assert watch.verdict(start + 20) is None
    watch.finished(start + 20)
    # Once finished, silence is expected: only the grace window counts.
    assert watch.verdict(start + 24) is None
    assert watch.verdict(start + 26) == "grace"
    watch.output(start + 26)
    watch.resumed()
    assert watch.verdict(start + 30) is None
    assert watch.verdict(start + 37) == "idle"


def test_watch_without_an_idle_limit_never_goes_idle():
    watch = Watch(idle_s=None, grace_s=5.0)
    assert watch.verdict(watch.last_output + 10**6) is None


def test_watch_sees_stderr_growth_as_output(tmp_path):
    log = tmp_path / "stderr.log"
    log.write_text("")
    watch = Watch(idle_s=10.0, grace_s=5.0)
    start = watch.last_output
    watch.poll_file(log, start + 8)
    assert watch.verdict(start + 11) == "idle"
    log.write_text("still compiling\n")
    watch.poll_file(log, start + 11)
    assert watch.verdict(start + 15) is None


def test_exec_result_has_an_idle_flag():
    assert ExecResult(exit_code=None).idle_stopped is False


# What the runner passes on


class BudgetSpy(FakeExecutor):
    """A fake executor that remembers the budget of each attempt."""

    def __init__(self, steps):
        super().__init__(steps)
        self.budgets = []

    def run(self, task, work, budget):
        self.budgets.append(budget)
        return super().run(task, work, budget)


@needs_sandbox
def test_the_worker_gets_the_idle_and_grace_limits(env, monkeypatch, tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yaml").write_text("jobs:\n  grace_seconds: 7\n")
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(config))
    executor = BudgetSpy([lambda w, t: good_change(w)])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], timeouts={"check": 60, "test_full": 1000})
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE
    budget = executor.budgets[0]
    assert budget.grace_s == 7.0 and budget.idle_s == 1500.0
    assert budget.max_turns == 5 and budget.wall_clock_s == 60


@needs_sandbox
def test_explicit_idle_minutes_reach_the_worker(env, monkeypatch, tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yaml").write_text("jobs:\n  idle_minutes: 3\n")
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(config))
    executor = BudgetSpy([add_docs])
    job_id = make_job(env, [NodeSpec(id="n1", goal="write notes")])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE
    assert executor.budgets[0].idle_s == 180.0
