"""A crash in Mullet's own checks, or a required review with no verdict, pauses the job.

Every case goes through a real JobRunner (a real clone, real gates in the
sandbox, a real ledger) with one gate, or the reviewer, made to break the way
a changed function signature would. The job must pause, commit nothing, use
no attempt and leave the worker's model free to work on the step: a bug in
Mullet says nothing about the worker's change.
"""

import subprocess
import time

import pytest

from mullet.jobs import gates, machine, runner as runner_module, sandbox
from mullet.jobs.clone import resolve_commit
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.ladder import BudgetState, Ladder, MachineState, PublicCheck, parse_config
from mullet.jobs.ledger import Ledger
from mullet.jobs.night import NightState
from mullet.jobs.profile import Checks, Profile
from mullet.jobs.runner import MAX_EXECUTOR_CRASHES, JobRunner
from mullet.jobs.types import AttemptOutcome, GateResult, GateStatus, JobStatus, NodeSpec, NodeStatus

from tests.jobs_helpers import good_change, make_source_repo

pytestmark = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)

CHECKS = Checks(
    lint="python3 -c \"import ast; ast.parse(open('app.py').read())\"",
    test_full="python3 runtests.py", test_targeted="python3 runtests.py {tests}", junit=True,
)
KIMI = "moonshotai/kimi-k2.7-code"
QWEN = "openrouter/qwen/qwen3-coder"
STEP = NodeSpec(id="n1", goal="add sub()", kind="mechanical")
NODE_GATES = {"G1": "gate_real_change", "G2": "gate_scope", "G3": "gate_static",
              "G6": "gate_test_weakening", "G4": "gate_build", "G5": "gate_tests"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(machine.paths, "state_dir", lambda: tmp_path / "state")
    source = make_source_repo(tmp_path / "src")
    ledger = Ledger(tmp_path / "jobs" / "ledger.db")
    yield {"tmp": tmp_path, "source": source, "ledger": ledger}
    ledger.close()


def make_job(env, nodes=(STEP,), *, review=False, ladder=True) -> str:
    ledger = env["ledger"]
    profile = Profile(repo_id="fixture", checks=CHECKS, env={"PYTHONDONTWRITEBYTECODE": "1"},
                      allowed_executors=["claude-code", "opencode"])
    budget = {"max_turns": 5, "wall_clock_s": 60, "review": review}
    if ladder:
        budget.update(ladder=True, max_spend=2.0)
    job_id = ledger.create_job(str(env["source"]), "improve the app", "main", resolve_commit(env["source"], "HEAD"),
                               profile.model_dump(), budget=budget)
    ledger.set_plan(job_id, [n if isinstance(n, tuple) else (n, None) for n in nodes])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    return job_id


def make_ladder():
    ladder = Ladder(
        config=parse_config({"kimi": KIMI, "openrouter": [QWEN]}),
        usage=lambda: None,
        machine=lambda c, wait_s=0: MachineState(False, "fake machine"),
        public_check=lambda repo, sha, profile: PublicCheck(False, "fake public check", "o/r", "main", time.time()),
        has_key=lambda rung: True,
        night=lambda: NightState(),
    )
    ladder.budget = lambda: BudgetState("medium", 0.6)
    return ladder


def good(work, task):
    good_change(work)


def workers():
    return {"kimi": FakeExecutor([good]), "openrouter": FakeExecutor([good]), "claude": FakeExecutor([])}


def run_ladder_job(env, job_id, fakes, reviewer=None):
    runner = JobRunner(env["ledger"], job_id, ladder=make_ladder(), rung_executors=lambda rung: fakes[rung.name],
                       notify=lambda *args, **kwargs: None, reviewer=reviewer)
    return runner, runner.run()


class Crash:
    """A gate or reviewer that raises the way a changed signature does, counting its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise TypeError("review() takes 1 positional argument but 2 were given")


def head(env, job_id) -> str:
    job = env["ledger"].get_job(job_id)
    return subprocess.run(["git", f"--git-dir={job.git_dir}", "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def assert_paused_and_unjudged(env, job_id, runner, gate):
    ledger = env["ledger"]
    job = ledger.get_job(job_id)
    assert job.status is JobStatus.PAUSED
    assert gate in (job.error or "")
    assert head(env, job_id) == job.base_sha, "nothing may be committed"
    assert ledger.get_node(job_id, "n1").status not in (NodeStatus.PASSED, NodeStatus.UNVERIFIED)
    assert ledger.attempt_count(job_id, "n1") == 0, "the attempt must not be counted"
    assert all(a.outcome is AttemptOutcome.PAUSED for a in ledger.attempts(job_id, "n1"))
    assert not any(t.counted or t.failed for t in runner._tried("n1"))
    # The worker's model isn't held against it: the step's next attempt goes to kimi again.
    pick = runner.executor_for(ledger.get_node(job_id, "n1"))
    assert pick.rung is not None and pick.rung.key == f"kimi:{KIMI}"


@pytest.mark.parametrize("gate", sorted(NODE_GATES))
def test_a_crashing_node_gate_pauses_the_job(env, monkeypatch, gate):
    crash = Crash()
    monkeypatch.setattr(gates, NODE_GATES[gate], crash)
    fakes = workers()
    job_id = make_job(env)
    runner, status = run_ladder_job(env, job_id, fakes)
    assert status is JobStatus.PAUSED
    assert_paused_and_unjudged(env, job_id, runner, gate)
    assert crash.calls == MAX_EXECUTOR_CRASHES  # rerun on the same work before pausing
    assert len(fakes["kimi"].calls) == 1  # the worker ran once; its work wasn't thrown away and redone


def test_a_crashing_reviewer_pauses_the_job(env):
    crash = Crash()
    fakes = workers()
    job_id = make_job(env, review=True)
    runner, status = run_ladder_job(env, job_id, fakes, reviewer=crash)
    assert status is JobStatus.PAUSED
    assert_paused_and_unjudged(env, job_id, runner, "G7")
    assert crash.calls == MAX_EXECUTOR_CRASHES


def no_verdict(ctx):
    return GateResult("G7", GateStatus.NOT_RUN, "the review didn't complete: no verdict")


def test_a_review_with_no_verdict_pauses_the_job(env):
    fakes = workers()
    job_id = make_job(env, review=True)
    runner, status = run_ladder_job(env, job_id, fakes, reviewer=no_verdict)
    assert status is JobStatus.PAUSED
    assert_paused_and_unjudged(env, job_id, runner, "G7")


def test_a_risky_steps_review_with_no_verdict_pauses_the_job(env, monkeypatch):
    # Review is off, but a step that deletes, kills or sends data always gets G7.
    monkeypatch.setattr(JobRunner, "_make_reviewer",
                        lambda self, job, profile, executor, force=False: no_verdict if force else None)
    risky = NodeSpec(id="n1", goal="add sub() and kill stray workers", kind="mechanical")
    fakes = workers()
    job_id = make_job(env, [risky])
    runner, status = run_ladder_job(env, job_id, fakes)
    assert status is JobStatus.PAUSED
    assert_paused_and_unjudged(env, job_id, runner, "G7")


def test_a_crashing_baseline_pauses_the_job(env, monkeypatch):
    crash = Crash()
    monkeypatch.setattr(gates.GateRunner, "baseline", crash)
    job_id = make_job(env)
    fakes = workers()
    runner, status = run_ladder_job(env, job_id, fakes)
    assert status is JobStatus.PAUSED
    job = env["ledger"].get_job(job_id)
    assert "G0" in job.error
    assert env["ledger"].attempts(job_id) == []
    assert head(env, job_id) == job.base_sha
    assert fakes["kimi"].calls == []


def test_resume_reruns_the_checks_on_the_same_work_without_the_worker(env):
    fakes = workers()
    job_id = make_job(env, review=True)
    _, status = run_ladder_job(env, job_id, fakes, reviewer=no_verdict)
    assert status is JobStatus.PAUSED

    reviewed = []

    def approve(ctx):
        reviewed.append(sorted(f.path for f in ctx.snapshot.files))
        return GateResult("G7", GateStatus.PASS, "reviewer approved")

    again = {"kimi": FakeExecutor([]), "openrouter": FakeExecutor([]), "claude": FakeExecutor([])}
    _, status = run_ladder_job(env, job_id, again, reviewer=approve)
    assert status is JobStatus.DONE
    ledger = env["ledger"]
    assert all(f.calls == [] for f in again.values()), "the worker's earlier change is checked, not redone"
    assert reviewed == [["app.py", "tests/test_app.py"]]
    first, second = ledger.attempts(job_id, "n1")
    assert first.outcome is AttemptOutcome.PAUSED and second.outcome is AttemptOutcome.PASSED
    assert second.usage["rung_key"] == f"kimi:{KIMI}" and second.usage["model"] == KIMI
    assert not second.usage.get("tokens") and not second.usage.get("total_cost_usd")  # cost stays on the first
    assert ledger.get_node(job_id, "n1").status is NodeStatus.PASSED
    assert head(env, job_id) != ledger.get_job(job_id).base_sha
    assert ledger.audit(job_id) == []


def test_a_rechecked_change_that_fails_counts_against_its_worker(env):
    fakes = workers()
    job_id = make_job(env, review=True)
    run_ladder_job(env, job_id, fakes, reviewer=no_verdict)

    def block(ctx):
        return GateResult("G7", GateStatus.FAIL, "reviewer blocked: incomplete", {"problems": ["incomplete"]})

    again = {"kimi": FakeExecutor([]), "openrouter": FakeExecutor([good]), "claude": FakeExecutor([])}
    calls = []

    def block_then_approve(ctx):
        calls.append(1)
        return block(ctx) if len(calls) == 1 else GateResult("G7", GateStatus.PASS, "reviewer approved")

    _, status = run_ladder_job(env, job_id, again, reviewer=block_then_approve)
    assert status is JobStatus.DONE
    outcomes = [((a.usage or {}).get("rung"), a.outcome) for a in env["ledger"].attempts(job_id, "n1")]
    assert outcomes == [("kimi", AttemptOutcome.PAUSED), ("kimi", AttemptOutcome.FAILED),
                        ("openrouter", AttemptOutcome.PASSED)]


def test_a_transient_gate_crash_is_rerun_on_the_same_work(env, monkeypatch):
    real = gates.gate_static
    calls = []

    def flaky_gate(ctx):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("temporary failure")
        return real(ctx)

    monkeypatch.setattr(gates, "gate_static", flaky_gate)
    fakes = workers()
    job_id = make_job(env)
    _, status = run_ladder_job(env, job_id, fakes)
    assert status is JobStatus.DONE
    assert len(fakes["kimi"].calls) == 1 and len(calls) == 2
    assert [a.outcome for a in env["ledger"].attempts(job_id, "n1")] == [AttemptOutcome.PASSED]


HELPER_TREE = [NodeSpec(id="feature", goal="helpers and extras"),
               (NodeSpec(id="helper", goal="add a helper"), "feature"),
               (NodeSpec(id="extra", goal="add extra"), "feature")]


def add_helper(work, task):
    (work / "helpers.py").write_text("def double(x):\n    return x * 2\n")
    (work / "tests" / "test_math.py").write_text(
        "from helpers import double\n\ndef test_double():\n    assert double(2) == 4\n")


def break_helper_quietly(work, task):
    (work / "extra.py").write_text("x = 1\n")
    (work / "tests" / "test_extra.py").write_text("def test_extra():\n    assert True\n")
    (work / "helpers.py").write_text("def double(x):\n    return x * 3\n")


def fix_helper(work, task):
    (work / "helpers.py").write_text("def double(x):\n    return x * 2\n")


@pytest.mark.parametrize("name", ["gate_scope", "gate_static", "gate_test_weakening"])
def test_a_crash_checking_a_parents_own_fix_pauses_the_job(env, monkeypatch, name):
    crash = Crash()
    monkeypatch.setattr(runner_module, name, crash)  # the runner's own check of the fix
    job_id = make_job(env, HELPER_TREE, ladder=False)
    executor = FakeExecutor([add_helper, break_helper_quietly, fix_helper])
    status = JobRunner(env["ledger"], job_id, executor, notify=lambda *a, **k: None).run()
    assert status is JobStatus.PAUSED
    ledger = env["ledger"]
    assert crash.calls == MAX_EXECUTOR_CRASHES
    assert ledger.get_node(job_id, "feature").status not in (NodeStatus.PASSED, NodeStatus.UNVERIFIED)
    fixes = [a for a in ledger.attempts(job_id, "feature") if a.executor == "fake"]
    assert [a.outcome for a in fixes] == [AttemptOutcome.PAUSED]


@pytest.mark.parametrize("gate, name", [("G4", "gate_build"), ("G5", "gate_tests")])
def test_a_crash_in_the_final_check_pauses_the_job(env, monkeypatch, gate, name):
    real = getattr(runner_module, name)
    monkeypatch.setattr(runner_module, name, Crash())  # only the final check calls the runner's copy
    job_id = make_job(env, ladder=False)
    status = JobRunner(env["ledger"], job_id, FakeExecutor([good]), notify=lambda *a, **k: None).run()
    assert status is JobStatus.PAUSED
    assert gate in env["ledger"].get_job(job_id).error
    monkeypatch.setattr(runner_module, name, real)
    status = JobRunner(env["ledger"], job_id, FakeExecutor([]), notify=lambda *a, **k: None).run()
    assert status is JobStatus.DONE


def run_noting(env, job_id, reviewer=None):
    sent = []
    runner = JobRunner(env["ledger"], job_id, ladder=make_ladder(), rung_executors=lambda rung: workers()[rung.name],
                       notify=lambda *args, **kwargs: sent.append(args), reviewer=reviewer)
    return runner.run(), sent


def test_a_pause_from_a_crashing_reviewer_sends_a_notification(env):
    job_id = make_job(env, review=True)
    status, sent = run_noting(env, job_id, reviewer=Crash())
    assert status is JobStatus.PAUSED
    assert [args[0] for args in sent] == ["paused"]
    event, notified_job, _goal, reason = sent[0]
    assert notified_job == job_id
    assert reason and "couldn't run" in reason
    assert f"mullet job resume {job_id}" in reason
    assert "review() takes" not in reason, "no output from the checks or the worker in the reason"


def test_a_pause_from_a_review_with_no_verdict_sends_a_notification(env):
    job_id = make_job(env, review=True)
    status, sent = run_noting(env, job_id, reviewer=no_verdict)
    assert status is JobStatus.PAUSED
    assert [args[0] for args in sent] == ["paused"]
    assert "no verdict" not in sent[0][3]


def test_a_pause_from_a_crashing_baseline_sends_a_notification(env, monkeypatch):
    monkeypatch.setattr(gates.GateRunner, "baseline", Crash())
    job_id = make_job(env)
    status, sent = run_noting(env, job_id)
    assert status is JobStatus.PAUSED
    assert [args[0] for args in sent] == ["paused"]
