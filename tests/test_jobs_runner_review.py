"""Tests for how the runner wires the built-in G7 reviewer: review standards and asking again for a verdict."""

import pytest

from mullet.jobs import sandbox
from mullet.jobs.executors.base import ExecResult
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.ledger import Ledger
from mullet.jobs.runner import JobRunner
from mullet.jobs.types import JobStatus, NodeSpec

from tests.jobs_helpers import BASE_FILES, good_change, make_source_repo
from tests.test_jobs_runner import make_job

pytestmark = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)

APPROVE = {"verdict": "approve", "findings": [], "reasons": []}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    source = make_source_repo(tmp_path / "src", {**BASE_FILES, "docs/STANDARDS.md": "Rule: name things plainly.\n"})
    ledger = Ledger(tmp_path / "jobs" / "ledger.db")
    yield {"tmp": tmp_path, "source": source, "ledger": ledger}
    ledger.close()


class AskingExecutor(FakeExecutor):
    """A scripted worker whose `ask` answers the reviewer from a list of results."""

    def __init__(self, scripts, answers):
        super().__init__(scripts)
        self.answers = list(answers)
        self.asked = []

    def ask(self, prompt, cwd, schema, budget, *, model=None, label=None):
        self.asked.append(prompt)
        return self.answers.pop(0)


class ResumingExecutor(AskingExecutor):
    """A worker whose reviewer sessions can be kept and resumed, like the Claude executor's."""

    def ask(self, prompt, cwd, schema, budget, *, model=None, label=None, persist=False, resume=None):
        self.asked.append(prompt)
        self.sessions.append((persist, resume))
        return self.answers.pop(0)

    sessions: list


def resuming(answers):
    executor = ResumingExecutor([lambda w, t: good_change(w)], answers)
    executor.sessions = []
    return executor


NO_VERDICT = ExecResult(0, session_id="rev-1")  # finished, but left no structured verdict
TURN_LIMIT = ExecResult(1, error="error_max_turns", session_id="rev-1")
BLOCK = {"verdict": "block", "findings": [], "reasons": ["it drops the old format with no migration"]}


def test_a_review_with_no_verdict_is_asked_again_in_its_session(env, tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "config"))
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True)
    executor = resuming([NO_VERDICT, ExecResult(0, claim=APPROVE, session_id="rev-1")])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE
    assert executor.sessions == [(True, None), (True, "rev-1")]


def test_reask_attempts_zero_pauses_as_before(env, tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yaml").write_text("jobs:\n  reask_attempts: 0\n")
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(config))
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True)
    executor = resuming([NO_VERDICT] * 4)
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.PAUSED
    assert all(session == (False, None) for session in executor.sessions)
    assert "G7 gave no verdict" in env["ledger"].get_job(job_id).error


def test_an_executor_that_cannot_resume_is_never_asked_to(env, tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "config"))
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True)
    executor = AskingExecutor([lambda w, t: good_change(w)], [NO_VERDICT] * 4)
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.PAUSED


def test_the_reviewer_sees_the_profiles_review_standards(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True,
                      review_standards="docs/STANDARDS.md")
    executor = AskingExecutor([lambda w, t: good_change(w)], [ExecResult(0, claim=APPROVE)])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE
    assert "Rule: name things plainly." in executor.asked[0]
    assert "BEGIN REVIEW STANDARDS" in executor.asked[0]


def test_a_review_cut_off_by_its_turn_limit_pauses_without_asking_again(env, tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "config"))
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True)
    executor = resuming([TURN_LIMIT] * 4)
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.PAUSED
    assert all(resume is None for _, resume in executor.sessions)  # never resumed to "answer now"
    assert "G7 gave no verdict" in env["ledger"].get_job(job_id).error


ONE_WAY = NodeSpec(id="n1", goal="add sub()", door="one-way")


@pytest.mark.parametrize("review", [True, False], ids=["review-on", "review-off"])
def test_an_approval_only_given_when_asked_again_pauses_a_one_way_step(env, tmp_path, monkeypatch, review):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "config"))
    job_id = make_job(env, [ONE_WAY], review=review)
    executor = resuming([NO_VERDICT, ExecResult(0, claim=APPROVE, session_id="rev-1")] * 4)
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.PAUSED
    assert executor.sessions[:2] == [(True, None), (True, "rev-1")]
    assert "only when asked again" in env["ledger"].get_job(job_id).error


def test_a_block_given_when_asked_again_still_fails_a_one_way_step(env, tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "config"))
    job_id = make_job(env, [ONE_WAY], review=True)
    executor = ResumingExecutor([lambda w, t: good_change(w)] * 5,
                                [NO_VERDICT, ExecResult(0, claim=BLOCK, session_id="rev-1")] * 5)
    executor.sessions = []
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.BLOCKED
    attempt = env["ledger"].attempts(job_id, "n1")[0]
    g7 = {r.gate: r for r in env["ledger"].gate_results(attempt.id)}["G7"]
    assert g7.status.value == "fail" and "no migration" in g7.summary


def test_a_one_way_step_is_reviewed_in_a_job_with_review_off(env, tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "config"))
    job_id = make_job(env, [ONE_WAY, NodeSpec(id="n2", goal="write notes")], review=False)
    notes = lambda w, t: (w / "NOTES.md").write_text("notes\n") and None  # noqa: E731
    executor = AskingExecutor([lambda w, t: good_change(w), notes], [ExecResult(0, claim=APPROVE)])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE
    assert len(executor.asked) == 1  # n1 is one-way and reviewed; n2 is not
