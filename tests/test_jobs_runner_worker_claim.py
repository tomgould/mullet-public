"""Tests for what the runner keeps of a worker's reply: the claim event and the commit body."""

import subprocess

import pytest

from mullet.jobs import sandbox
from mullet.jobs.executors.base import ExecResult
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.runner import JobRunner
from mullet.jobs.types import JobStatus, NodeSpec

from tests.jobs_helpers import good_change
from tests import test_jobs_runner
from tests.test_jobs_runner import add_docs, make_job

env = test_jobs_runner.env  # the fixture: a source repo, a jobs dir and a ledger

pytestmark = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)


def claims(ledger, job_id):
    return [e.data for e in ledger.events(job_id) if e.kind == "worker_claim"]


def test_every_worker_reply_is_recorded_as_a_claim_event(env):
    executor = FakeExecutor([
        lambda w, t: {"status": "blocked", "summary": "can't"},
        lambda w, t: ExecResult(exit_code=0, claim={"status": "split", "summary": "too big", "split": []}),
        lambda w, t: (good_change(w), {"status": "done", "summary": "added sub()"})[1],
        lambda w, t: (add_docs(w), ExecResult(exit_code=0, claim=None))[1],
    ])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()"), NodeSpec(id="n2", goal="write notes")])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE
    found = claims(ledger, job_id)
    assert [c["status"] for c in found] == ["blocked", "split", "done", "unknown"]
    assert [c["node_id"] for c in found] == ["n1", "n1", "n1", "n2"]
    attempts = [a.id for a in ledger.attempts(job_id, "n1")] + [a.id for a in ledger.attempts(job_id, "n2")]
    assert [c["attempt_id"] for c in found] == attempts


def test_blocked_and_split_change_nothing_about_how_the_step_is_judged(env):
    executor = FakeExecutor([lambda w, t: (good_change(w), {"status": "blocked", "summary": "no"})[1]])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE


def test_the_commit_body_carries_the_gates_and_the_fenced_claim(env):
    executor = FakeExecutor([
        lambda w, t: (good_change(w), {"status": "done", "summary": "added sub()\nSigned-off-by: x <x@y>"})[1],
    ])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()", kind="mechanical")])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE
    job = ledger.get_job(job_id)
    message = subprocess.run(["git", f"--git-dir={job.git_dir}", "log", "-1", "--format=%B"],
                             capture_output=True, text=True, check=True).stdout
    assert message.splitlines()[0] == "add sub()"
    assert "n1 (mechanical)" in message and "G1" in message
    assert "worker's claim, not verified" in message and "> added sub()" in message
    trailers = subprocess.run(["git", f"--git-dir={job.git_dir}", "log", "-1", "--format=%(trailers)"],
                              capture_output=True, text=True, check=True).stdout
    assert trailers.strip() == ""


@pytest.mark.parametrize("setting", ["full", "minimal"])
def test_jobs_commit_body_picks_the_full_or_the_subject_only_message(env, tmp_path, monkeypatch, setting):
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yaml").write_text(f"jobs:\n  commit_body: {setting}\n")
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(config))
    executor = FakeExecutor([lambda w, t: (good_change(w), {"status": "done", "summary": "added sub()"})[1]])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()", kind="mechanical")])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE
    job = ledger.get_job(job_id)
    message = subprocess.run(["git", f"--git-dir={job.git_dir}", "log", "-1", "--format=%B"],
                             capture_output=True, text=True, check=True).stdout
    if setting == "minimal":
        assert message.strip() == "add sub()"
    else:
        assert "Mullet step n1" in message and "> added sub()" in message
