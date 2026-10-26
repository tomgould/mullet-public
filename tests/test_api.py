"""The Python API (mullet/api.py): the same job flow as ``mullet job``, driven from code."""

import dataclasses
import time
from types import SimpleNamespace

import pytest
import yaml

from mullet import api
from mullet.cli import job_cli
from mullet.core import strategy
from mullet.core.strategy import Strategy
from mullet.errors import (
    JobError, JobNotFound, MulletError, NotApproved, NotRunnable, PlanFailed, RunnerBusy, SpendLimit,
    StrategyOff, Unverified,
)
from mullet.jobs import sandbox
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.ledger import Ledger
from mullet.jobs.runner import JobRunner
from mullet.jobs.types import GateResult, GateStatus, JobStatus, NodeStatus

from tests.jobs_helpers import git, good_change
from tests.test_jobs_cli import repo, store_profile  # noqa: F401  (fixtures)

needs_sandbox = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)


@pytest.fixture
def plan_file(tmp_path):
    path = tmp_path / "plan.yaml"
    nodes = [{"id": "n1", "goal": "add a sub function", "scope": ["app.py", "tests/**"]}]
    path.write_text(yaml.safe_dump({"nodes": nodes}))
    return path


@pytest.fixture
def fake_worker(monkeypatch):
    """In-process runs use a fake worker that makes a good change and a reviewer that approves."""
    monkeypatch.setattr(job_cli, "preflight", lambda status, job=None: None)

    def change_and_claim(work, task):
        good_change(work)
        return {"status": "done", "summary": "added sub() with a test"}

    def reviewer(ctx):
        return GateResult("G7", GateStatus.PASS, "reviewer approved")

    monkeypatch.setattr(api, "_make_runner", lambda ledger, job_id: JobRunner(
        ledger, job_id, FakeExecutor([change_and_claim]), reviewer=reviewer))


def planned(repo, plan_file, **kwargs):
    return api.plan("add a sub function", repo, plan_file=plan_file, **kwargs)


@needs_sandbox
def test_plan_approve_run_report_merge_end_to_end(repo, plan_file, fake_worker):
    store_profile(repo)
    job = planned(repo, plan_file)
    assert api.status(job).status == "awaiting_approval"
    api.approve(job)
    result = api.run(job)
    assert result.status == "done" and result.nodes_passed == 1 and result.nodes_unverified == 0
    assert result.spent_usd == 0.0 and result.error is None and not result.runner_alive
    assert "n1: passed" in api.report(job)
    head = git(repo, "rev-parse", "HEAD")
    merged = api.merge(job)
    assert merged.branch == f"mullet/{job.id}" and "app.py" in merged.diffstat and merged.unverified == ()
    assert git(repo, "rev-parse", "HEAD") == head
    assert "def sub" in git(repo, "show", f"{merged.branch}:app.py")
    assert api.status(job.id).status == "merged"
    assert [j.id for j in api.list_jobs()] == [job.id]


def test_a_job_id_string_works_as_well_as_a_handle(repo, plan_file):
    job = planned(repo, plan_file)
    assert api.status(job.id) == api.status(job) and str(job) == job.id
    api.approve(job.id)
    assert api.status(job).status == "approved"


def test_detach_starts_the_background_runner(repo, plan_file, monkeypatch):
    started = []
    monkeypatch.setattr(job_cli, "preflight", lambda status, job=None: None)
    job = planned(repo, plan_file)
    api.approve(job)
    monkeypatch.setattr(job_cli.subprocess, "Popen",
                        lambda cmd, **kw: started.append((cmd, kw)) or SimpleNamespace(pid=4242))
    assert api.run(job, detach=True).status == "approved"
    (cmd, kwargs), = started
    assert cmd[-2:] == ["mullet.jobs.runner", job.id] and kwargs["start_new_session"] is True


def test_run_needs_wait_or_detach(repo, plan_file):
    job = planned(repo, plan_file)
    with pytest.raises(JobError, match="wait=True or detach=True"):
        api.run(job, wait=False)


def test_errors_map_to_mullet_errors(repo, plan_file, monkeypatch):
    with pytest.raises(JobNotFound):
        api.status("20990101-000000-abcdef")
    with pytest.raises(JobNotFound):
        api.report("20990101-000000-abcdef")
    job = planned(repo, plan_file)
    with pytest.raises(NotApproved, match="only approved, paused or blocked jobs can run"):
        api.run(job)
    api.approve(job)
    with pytest.raises(JobError, match="not awaiting approval"):
        api.approve(job)
    with pytest.raises(NotRunnable, match="only a finished job"):
        api.merge(job)
    with pytest.raises(NotRunnable, match="no runner"):
        api.stop(job)
    monkeypatch.setattr(job_cli, "preflight", lambda status, job=None: None)
    with pytest.raises(SpendLimit, match="ladder"):
        api.resume(job, max_spend=1.0)
    with pytest.raises(SpendLimit):
        api.plan("x", repo, plan_file=plan_file, max_spend=float("nan"))
    with pytest.raises(SpendLimit, match="negative"):
        api.plan("x", repo, plan_file=plan_file, max_spend=-1)
    monkeypatch.setattr(job_cli, "_runner_alive", lambda job_id: True)
    with pytest.raises(RunnerBusy):
        api.run(job)
    assert issubclass(JobNotFound, MulletError) and issubclass(PlanFailed, JobError)


def test_a_bad_plan_file_raises_and_leaves_no_job(repo, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"nodes": [{"id": "a", "goal": "x", "scope": ["**"], "max_lines": 10**9}]}))
    with pytest.raises(JobError, match="max_lines"):
        api.plan("goal", repo, plan_file=bad)
    assert api.list_jobs() == []


def test_unknown_executor_is_refused(repo, plan_file):
    with pytest.raises(JobError, match="executor"):
        api.plan("goal", repo, plan_file=plan_file, executor="gpt")


def _done_job_with_unverified_node(repo, plan_file):
    job = planned(repo, plan_file)
    api.approve(job)
    with Ledger() as ledger:
        ledger.set_job_status(job.id, JobStatus.RUNNING)
        with ledger._tx() as conn:
            ledger._set_node(conn, job.id, "n1", NodeStatus.UNVERIFIED, end_sha="abc")
        ledger.set_job_status(job.id, JobStatus.DONE)
    return job


def test_unverified_merge_is_refused_unless_allowed(repo, plan_file, monkeypatch):
    class FakeClone:
        def __init__(self, root):
            pass

        def git(self, *args):
            return SimpleNamespace(stdout="1 file changed")

        def fetch_into(self, target_repo, branch):
            return "0123456789abcdef"

    monkeypatch.setattr(job_cli, "JobClone", FakeClone)
    job = _done_job_with_unverified_node(repo, plan_file)
    info = api.status(job)
    assert info.status == "done" and info.nodes_unverified == 1 and info.nodes_passed == 0
    with pytest.raises(Unverified, match="n1"):
        api.merge(job)
    assert api.status(job).status == "done"
    merged = api.merge(job, allow_unverified=True)
    assert merged.unverified == ("n1",) and merged.sha == "0123456789abcdef"
    assert api.status(job).status == "merged"


def test_strategy_off_refuses_plan_and_leaves_no_job(repo, plan_file):
    strategy.save(Strategy(None, "off", time.time()))
    with pytest.raises(StrategyOff, match="mullet strategy set --delegation"):
        planned(repo, plan_file)
    assert api.list_jobs() == []


def test_strategy_budget_becomes_the_spending_cap_unless_given(repo, plan_file):
    strategy.save(Strategy(0.4, "ladder", time.time()))
    job = planned(repo, plan_file)
    explicit = planned(repo, plan_file, max_spend=0.9)
    with Ledger() as ledger:
        assert ledger.get_job(job.id).budget["max_spend"] == 0.4
        assert ledger.get_job(explicit.id).budget["max_spend"] == 0.9


def test_results_are_frozen_plain_values(repo, plan_file):
    job = planned(repo, plan_file)
    info = api.status(job)
    for value in (job, info, api.list_jobs()[0]):
        assert dataclasses.is_dataclass(value)
        with pytest.raises(dataclasses.FrozenInstanceError):
            value.id = "other"
    assert isinstance(info.spent_usd, float) and isinstance(info.nodes_total, int)
    assert {"plan", "approve", "run", "status", "report", "stop", "resume", "merge", "list_jobs"} <= set(api.__all__)


def test_stop_asks_the_live_runner(repo, plan_file, monkeypatch):
    killed = []
    monkeypatch.setattr(job_cli, "_runner_alive", lambda job_id: True)
    monkeypatch.setattr(job_cli, "_runner_pid", lambda job_id: 99)
    monkeypatch.setattr(job_cli.os, "kill", lambda pid, sig: killed.append(pid))
    api.stop(planned(repo, plan_file))
    assert killed == [99]


def test_plan_handle_carries_the_planners_open_questions(repo, tmp_path):
    plan_file = tmp_path / "q.yaml"
    plan_file.write_text(yaml.safe_dump({
        "nodes": [{"id": "n1", "goal": "add a sub function", "scope": ["app.py"]}],
        "open_questions": ["Which rounding mode?"]}))
    handle = api.plan("goal", repo, plan_file=plan_file)
    assert handle.open_questions == ("Which rounding mode?",)
