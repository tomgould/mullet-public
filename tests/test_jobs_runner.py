"""Tests for mullet.jobs.runner: whole jobs driven by a scripted executor.

They use real clones, real gates in the sandbox and a real ledger; only the
executor is fake, so every outcome here comes from the gates.
"""

import fcntl
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from mullet.jobs import children, sandbox
from mullet.jobs.clone import resolve_commit
from mullet.jobs.executors.base import ExecResult
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.gates import GatePaused
from mullet.jobs.ledger import Ledger
from mullet.jobs.profile import Checks, Profile
from mullet.jobs.runner import JobBusy, JobRunner, RunnerOptions, StopRequested, feedback_for, profile_commands
from mullet.jobs.types import AttemptOutcome, GateResult, GateStatus, JobStatus, NodeSpec, NodeStatus

from tests.jobs_helpers import BASE_FILES, FAKE_CLAUDE, git, good_change, make_source_repo

pytestmark = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)

CHECKS = Checks(
    lint="python3 -c \"import ast; ast.parse(open('app.py').read())\"",
    test_full="python3 runtests.py", test_targeted="python3 runtests.py {tests}", junit=True,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    source = make_source_repo(tmp_path / "src")
    ledger = Ledger(tmp_path / "jobs" / "ledger.db")
    yield {"tmp": tmp_path, "source": source, "ledger": ledger}
    ledger.close()


def make_job(env, nodes, checks=CHECKS, source=None, review=False, **profile_kwargs) -> str:
    """Nodes are NodeSpecs (top level) or (NodeSpec, parent id) pairs."""
    source, ledger = source or env["source"], env["ledger"]
    profile = Profile(repo_id="fixture", checks=checks, env={"PYTHONDONTWRITEBYTECODE": "1"}, **profile_kwargs)
    job_id = ledger.create_job(str(source), "improve the app", "main", resolve_commit(source, "HEAD"),
                               profile.model_dump(), budget={"max_turns": 5, "wall_clock_s": 60, "review": review})
    ledger.set_plan(job_id, [n if isinstance(n, tuple) else (n, None) for n in nodes])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    return job_id


def user_repo_state(repo: Path):
    return (git(repo, "for-each-ref", "--format=%(refname) %(objectname)"),
            git(repo, "status", "--porcelain", "--untracked-files=all"),
            git(repo, "stash", "list"))


def add_mul(work: Path, task=None):
    with open(work / "app.py", "a") as fh:
        fh.write("\n\ndef mul(a, b):\n    return a * b\n")
    with open(work / "tests" / "test_app.py", "a") as fh:
        fh.write("\ndef test_mul():\n    from app import mul\n    assert mul(2, 3) == 6\n")


def break_add(work: Path, task=None):
    (work / "app.py").write_text((work / "app.py").read_text().replace("return a + b", "return a - b"))


def add_docs(work: Path, task=None):
    (work / "NOTES.md").write_text("notes\n")


NODES = [
    NodeSpec(id="sub", goal="add sub()"),
    NodeSpec(id="docs", goal="write notes"),
    NodeSpec(id="mul", goal="add mul()"),
]


def test_three_node_job_with_a_retry_runs_to_done(env):
    before = user_repo_state(env["source"])
    seen_feedback = []

    def docs_after_fixing(work, task):
        seen_feedback.extend(task.feedback)
        add_docs(work)

    executor = FakeExecutor([
        lambda w, t: good_change(w),
        lambda w, t: (add_docs(w), break_add(w)) and None,  # planted failure: breaks add()
        docs_after_fixing,
        add_mul,
    ])
    job_id = make_job(env, NODES)
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE

    nodes = {n.node_id: n for n in ledger.nodes(job_id)}
    assert all(n.status is NodeStatus.PASSED for n in nodes.values())
    docs_attempts = ledger.attempts(job_id, "docs")
    assert [a.outcome for a in docs_attempts] == [AttemptOutcome.FAILED, AttemptOutcome.PASSED]
    assert any("G5 failed" in f for f in seen_feedback)
    assert ledger.audit(job_id) == []

    job = ledger.get_job(job_id)
    clone_git = ["git", f"--git-dir={job.git_dir}"]
    log = subprocess.run([*clone_git, "log", "--format=%s", f"{job.base_sha}..HEAD"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert log == ["add mul()", "write notes", "add sub()"]
    final = [e for e in ledger.events(job_id) if e.kind == "final_check"][-1].data
    assert {r["gate"]: r["status"] for r in final["results"]} == {"G4": "pass", "G5": "pass"}
    assert user_repo_state(env["source"]) == before


def test_a_node_that_never_passes_blocks_the_job_and_resume_retries_it(env):
    executor = FakeExecutor([break_add, break_add, break_add])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.BLOCKED
    assert ledger.get_node(job_id, "n1").status is NodeStatus.BLOCKED
    assert "ran out of attempts" in ledger.get_job(job_id).error
    assert JobRunner(ledger, job_id, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.DONE
    assert ledger.attempt_count(job_id, "n1") == 4


def test_claims_never_count(env):
    executor = FakeExecutor([lambda w, t: {"status": "done", "summary": "all tests pass"}] * 3)
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.BLOCKED
    for attempt in env["ledger"].attempts(job_id, "n1"):
        assert attempt.claim == {"status": "done", "summary": "all tests pass"}
        assert attempt.outcome is AttemptOutcome.FAILED


def test_final_check_catches_what_targeted_runs_missed(env):
    def add_helper(work, task):
        (work / "helpers.py").write_text("def double(x):\n    return x * 2\n")
        (work / "tests" / "test_math.py").write_text(
            "from helpers import double\n\ndef test_double():\n    assert double(2) == 4\n")

    def break_helper_quietly(work, task):
        # The targeted run only picks tests/test_extra.py; nothing is named
        # after helpers.py, so only the final full run sees test_math break.
        (work / "extra.py").write_text("x = 1\n")
        (work / "tests" / "test_extra.py").write_text("def test_extra():\n    assert True\n")
        (work / "helpers.py").write_text("def double(x):\n    return x * 3\n")

    job_id = make_job(env, [NodeSpec(id="helper", goal="add a helper"), NodeSpec(id="extra", goal="add extra")])
    executor = FakeExecutor([add_helper, break_helper_quietly])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.BLOCKED
    assert "final check failed" in env["ledger"].get_job(job_id).error


def test_unverified_nodes_still_finish_but_never_pass(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], checks=Checks())
    assert JobRunner(env["ledger"], job_id, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.DONE
    assert env["ledger"].get_node(job_id, "n1").status is NodeStatus.UNVERIFIED


def test_setup_failure_blocks_with_the_command(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], setup=["exit 3"])
    assert JobRunner(env["ledger"], job_id, FakeExecutor([])).run() is JobStatus.BLOCKED
    assert "setup command failed: exit 3" in env["ledger"].get_job(job_id).error


def test_usage_limit_pauses_without_using_an_attempt(env):
    clock = {"now": time.time()}
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock["now"] += seconds

    limited = ExecResult(exit_code=1, paused_until=clock["now"] + 600, error="usage limit reached")
    executor = FakeExecutor([lambda w, t: limited, lambda w, t: good_change(w)])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    runner = JobRunner(env["ledger"], job_id, executor, RunnerOptions(pause_poll_s=100),
                       sleep=sleep, now=lambda: clock["now"])
    assert runner.run() is JobStatus.DONE
    assert env["ledger"].attempt_count(job_id, "n1") == 1
    assert [a.outcome for a in env["ledger"].attempts(job_id, "n1")] == [AttemptOutcome.PAUSED, AttemptOutcome.PASSED]
    assert sum(slept) >= 600
    assert any(e.kind == "waiting_for_limit" for e in env["ledger"].events(job_id))


def test_runner_pauses_before_a_window_runs_out(env):
    clock = {"now": time.time()}
    resets = clock["now"] + 900

    def nearly_out(work, task):
        good_change(work)
        return ExecResult(exit_code=0, claim={"status": "done", "summary": "ok"}, usage={"rate_limit": {
            "status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.97, "resetsAt": resets}}}})

    def sleep(seconds):
        clock["now"] += seconds

    job_id = make_job(env, [NodeSpec(id="sub", goal="add sub()"), NodeSpec(id="mul", goal="add mul()")])
    runner = JobRunner(env["ledger"], job_id, FakeExecutor([nearly_out, add_mul]), RunnerOptions(pause_poll_s=60),
                       sleep=sleep, now=lambda: clock["now"])
    assert runner.run() is JobStatus.DONE
    assert clock["now"] >= resets


def test_stop_request_pauses_and_resume_finishes(env):
    started = threading.Event()

    def slow(work, task):
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        with children.track(proc):
            started.set()
            proc.wait()
        good_change(work)

    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    box = {}

    def run_in_thread():
        # SQLite connections belong to one thread, as in a real runner process.
        with Ledger(env["tmp"] / "jobs" / "ledger.db") as ledger:
            box["runner"] = JobRunner(ledger, job_id, FakeExecutor([slow]))
            box["result"] = box["runner"].run()

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    assert started.wait(30)
    box["runner"].request_stop()
    thread.join(60)
    assert box["result"] is JobStatus.PAUSED
    attempts = env["ledger"].attempts(job_id, "n1")
    assert [a.outcome for a in attempts] == [AttemptOutcome.ABORTED]
    assert env["ledger"].attempt_count(job_id, "n1") == 0
    assert JobRunner(env["ledger"], job_id, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.DONE


def test_the_runner_names_itself_by_a_start_a_clock_step_does_not_move(env, monkeypatch):
    from mullet.cli import job_cli
    from mullet.jobs import reaper

    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    seen = {}
    real = psutil.Process.create_time

    def step(work, task):
        seen["pid"] = (env["tmp"] / "jobs" / job_id / "runner.pid").read_text()
        # NTP or a resume steps the wall clock, which moves psutil's create_time.
        monkeypatch.setattr(psutil.Process, "create_time", lambda self: real(self) + 3600.0)
        seen["state"] = reaper.runner_state(job_id)
        seen["confirmed"] = job_cli._runner_pid(job_id)
        good_change(work)

    assert JobRunner(env["ledger"], job_id, FakeExecutor([step])).run() is JobStatus.DONE
    assert seen["pid"] == f"{os.getpid()} {children.start_ticks(os.getpid())}@{children.boot_id()}"
    assert seen["state"] == "live" and seen["confirmed"] == os.getpid()


def _crashing_runner(jobs_dir: str, db: str, job_id: str, marker: str) -> None:
    os.environ["MULLET_JOBS_DIR"] = jobs_dir

    def hang(work, task):
        orphan = subprocess.Popen(["sleep", "300"], start_new_session=True)
        with children.track(orphan):
            Path(marker).write_text(str(orphan.pid))
            time.sleep(300)

    with Ledger(Path(db)) as ledger:
        JobRunner(ledger, job_id, FakeExecutor([hang])).run()


def test_kill_minus_nine_then_resume_cleans_up_and_finishes(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    marker = env["tmp"] / "orphan.pid"
    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(target=_crashing_runner, args=(
        os.environ["MULLET_JOBS_DIR"], str(env["tmp"] / "jobs" / "ledger.db"), job_id, str(marker)))
    proc.start()
    deadline = time.monotonic() + 60
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert marker.exists()
    orphan_pid = int(marker.read_text())
    os.kill(proc.pid, 9)
    proc.join(10)
    assert psutil.pid_exists(orphan_pid)  # the executor's child outlived the runner

    ledger = env["ledger"]
    assert ledger.get_job(job_id).status is JobStatus.RUNNING
    assert JobRunner(ledger, job_id, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.DONE
    assert not psutil.pid_exists(orphan_pid) or psutil.Process(orphan_pid).status() == psutil.STATUS_ZOMBIE
    outcomes = [a.outcome for a in ledger.attempts(job_id, "n1")]
    assert outcomes == [AttemptOutcome.ABORTED, AttemptOutcome.PASSED]
    assert ledger.audit(job_id) == []


def test_a_second_runner_for_the_same_job_is_refused(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    started = threading.Event()
    release = threading.Event()

    def wait(work, task):
        started.set()
        release.wait(30)
        good_change(work)

    def run_in_thread():
        with Ledger(env["tmp"] / "jobs" / "ledger.db") as ledger:
            JobRunner(ledger, job_id, FakeExecutor([wait])).run()

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    assert started.wait(30)
    with Ledger(env["tmp"] / "jobs" / "ledger.db") as other:
        with pytest.raises(JobBusy):
            JobRunner(other, job_id, FakeExecutor([])).run()
    release.set()
    thread.join(60)


def test_tampering_with_a_dependency_dir_fails_the_checks(env):
    files = dict(BASE_FILES)
    files[".gitignore"] = "__pycache__/\n.venv/\n"
    source = make_source_repo(env["tmp"] / "src-deps", files)
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], source=source,
                      deps_paths=[".venv"], setup=["mkdir -p .venv && echo real > .venv/lib.py"])

    def tamper(work, task):
        good_change(work)
        (work / ".venv" / "lib.py").write_text("planted\n")

    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, FakeExecutor([tamper] * 3)).run() is JobStatus.BLOCKED
    first = ledger.attempts(job_id, "n1")[0]
    g4 = {r.gate: r for r in ledger.gate_results(first.id)}["G4"]
    assert g4.status is GateStatus.FAIL and "changed after setup" in g4.summary


def _crash_right_after_commit(jobs_dir: str, db: str, job_id: str) -> None:
    os.environ["MULLET_JOBS_DIR"] = jobs_dir
    from mullet.jobs import clone as clone_mod
    original = clone_mod.JobClone.commit_snapshot

    def commit_then_die(self, *args, **kwargs):
        original(self, *args, **kwargs)
        os._exit(1)  # like kill -9: no handler runs, nothing is recorded

    clone_mod.JobClone.commit_snapshot = commit_then_die
    with Ledger(Path(db)) as ledger:
        JobRunner(ledger, job_id, FakeExecutor([lambda w, t: good_change(w)])).run()


def test_a_commit_the_ledger_never_recorded_is_dropped_on_resume(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    proc = multiprocessing.get_context("fork").Process(target=_crash_right_after_commit, args=(
        os.environ["MULLET_JOBS_DIR"], str(env["tmp"] / "jobs" / "ledger.db"), job_id))
    proc.start()
    proc.join(120)
    ledger = env["ledger"]
    job = ledger.get_job(job_id)
    head = subprocess.run(["git", f"--git-dir={job.git_dir}", "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head != job.base_sha  # committed, but the ledger never heard

    honest = FakeExecutor([lambda w, t: good_change(w)])
    assert JobRunner(ledger, job_id, honest).run() is JobStatus.DONE
    assert any(e.kind == "head_reset" for e in ledger.events(job_id))
    assert [a.outcome for a in ledger.attempts(job_id, "n1")] == [AttemptOutcome.ABORTED, AttemptOutcome.PASSED]
    assert ledger.audit(job_id) == []


def test_executor_crashes_do_not_use_attempts_but_three_block_the_job(env):
    crash = ExecResult(exit_code=1, aborted=True, error="the CLI ended without a result (exit 1)")
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, FakeExecutor([lambda w, t: crash] * 3)).run() is JobStatus.BLOCKED
    assert "failed 3 times" in ledger.get_job(job_id).error
    assert ledger.attempt_count(job_id, "n1") == 0
    executor = FakeExecutor([lambda w, t: crash, lambda w, t: crash, lambda w, t: good_change(w)])
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE
    assert ledger.attempt_count(job_id, "n1") == 1


def test_stop_while_waiting_for_the_worker_lock_runs_nothing(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    calls = []
    box = {}
    lock_path = Path(os.environ["MULLET_JOBS_DIR"]) / "worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)  # another job is using the worker
    try:
        def run_in_thread():
            with Ledger(env["tmp"] / "jobs" / "ledger.db") as ledger:
                box["runner"] = JobRunner(ledger, job_id, FakeExecutor([lambda w, t: calls.append(1)]),
                                          RunnerOptions(pause_poll_s=0.1))
                box["result"] = box["runner"].run()

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        deadline = time.monotonic() + 30
        while not any(e.kind == "attempt_started" for e in env["ledger"].events(job_id)):
            assert time.monotonic() < deadline
            time.sleep(0.1)
        box["runner"].request_stop()
        thread.join(30)
    finally:
        os.close(fd)
    assert box["result"] is JobStatus.PAUSED
    assert calls == []
    assert env["ledger"].attempt_count(job_id, "n1") == 0


def test_sigterm_to_a_real_runner_process_pauses_the_job(env, tmp_path):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    claude = bin_dir / "claude"
    claude.write_text(FAKE_CLAUDE)
    claude.chmod(0o755)
    (bin_dir / "scenario.json").write_text(json.dumps({"sleep": 120}))
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    child_env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    proc = subprocess.Popen([sys.executable, "-m", "mullet.jobs.runner", job_id], env=child_env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while not (bin_dir / "call.json").exists():
            assert time.monotonic() < deadline, "the runner never started the executor"
            time.sleep(0.2)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == 3  # 3 means paused
    finally:
        if proc.poll() is None:
            proc.kill()
    assert env["ledger"].get_job(job_id).status is JobStatus.PAUSED
    assert [a.outcome for a in env["ledger"].attempts(job_id, "n1")] == [AttemptOutcome.ABORTED]


def test_odd_rate_limit_values_never_crash_the_pause_logic():
    from mullet.jobs.runner import _limit_reset
    now = time.time()
    assert _limit_reset({"unifiedWindows": {"five_hour": {"utilization": None, "resetsAt": None}}}, 0.9, now) is None
    assert _limit_reset({"unifiedWindows": {"x": "junk"}}, 0.9, now) is None
    assert _limit_reset({}, 0.9, now) is None
    assert _limit_reset({"unifiedWindows": {"w": {"utilization": 0.99, "resetsAt": now + 50}}}, 0.9, now) == now + 50


def test_a_parent_passes_on_an_integration_check_of_its_childrens_work(env):
    nodes = [NodeSpec(id="math", goal="add arithmetic"),
             (NodeSpec(id="sub", goal="add sub()"), "math"),
             (NodeSpec(id="mul", goal="add mul()"), "math")]
    job_id = make_job(env, nodes)
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, FakeExecutor([lambda w, t: good_change(w), add_mul])).run() is JobStatus.DONE
    parent_attempts = ledger.attempts(job_id, "math")
    assert [(a.executor, a.outcome) for a in parent_attempts] == [("integration-check", AttemptOutcome.PASSED)]
    g5 = {r.gate: r for r in ledger.gate_results(parent_attempts[0].id)}["G5"]
    assert g5.details["mode"] == "test_full"
    assert ledger.get_node(job_id, "math").end_sha == ledger.get_node(job_id, "mul").end_sha
    assert ledger.audit(job_id) == []


def test_a_failed_integration_check_is_fixed_by_the_executor(env):
    def add_helper(work, task):
        (work / "helpers.py").write_text("def double(x):\n    return x * 2\n")
        (work / "tests" / "test_math.py").write_text(
            "from helpers import double\n\ndef test_double():\n    assert double(2) == 4\n")

    def break_helper_quietly(work, task):
        (work / "extra.py").write_text("x = 1\n")
        (work / "tests" / "test_extra.py").write_text("def test_extra():\n    assert True\n")
        (work / "helpers.py").write_text("def double(x):\n    return x * 3\n")

    fix_feedback = []

    def fix_integration(work, task):
        fix_feedback.extend(task.feedback)
        (work / "helpers.py").write_text("def double(x):\n    return x * 2\n")

    nodes = [NodeSpec(id="feature", goal="helpers and extras"),
             (NodeSpec(id="helper", goal="add a helper"), "feature"),
             (NodeSpec(id="extra", goal="add extra"), "feature")]
    job_id = make_job(env, nodes)
    ledger = env["ledger"]
    executor = FakeExecutor([add_helper, break_helper_quietly, fix_integration])
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE
    attempts = ledger.attempts(job_id, "feature")
    assert [(a.executor, a.outcome) for a in attempts] == [
        ("integration-check", AttemptOutcome.FAILED), ("fake", AttemptOutcome.PASSED)]
    assert any("G5 failed" in f for f in fix_feedback)
    assert ledger.audit(job_id) == []


def test_review_blocks_then_approves(env):
    verdicts = iter([GateStatus.FAIL, GateStatus.PASS])
    reviewed = []

    def reviewer(ctx):
        reviewed.append(ctx.node.id)
        status = next(verdicts)
        return GateResult("G7", status, "reviewer approved" if status is GateStatus.PASS else "reviewer blocked: stub")

    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True)
    ledger = env["ledger"]
    executor = FakeExecutor([lambda w, t: good_change(w)] * 2)
    assert JobRunner(ledger, job_id, executor, reviewer=reviewer).run() is JobStatus.DONE
    outcomes = [a.outcome for a in ledger.attempts(job_id, "n1")]
    assert outcomes == [AttemptOutcome.FAILED, AttemptOutcome.PASSED]
    assert reviewed == ["n1", "n1"]
    assert ledger.audit(job_id) == []


def test_review_is_required_when_the_job_turns_it_on(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True)
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.BLOCKED
    assert "can't run a reviewer" in ledger.get_job(job_id).error


def test_the_built_in_reviewer_runs_in_the_worker_slot(env):
    class AskingExecutor(FakeExecutor):
        def ask(self, prompt, cwd, schema, budget, *, model=None, label=None):
            held = []
            with open(env["tmp"] / "jobs" / "worker.lock", "a") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    held.append(True)
            return ExecResult(0, claim={"verdict": "approve" if held else "block", "reasons": ["no slot"]})

    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True)
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, AskingExecutor([lambda w, t: good_change(w)])).run() is JobStatus.DONE
    assert ledger.audit(job_id) == []


def test_an_incomplete_review_pauses_the_job_and_commits_nothing(env):
    def reviewer(ctx):
        return GateResult("G7", GateStatus.NOT_RUN, "the review didn't complete: timed out")

    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True)
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, FakeExecutor([lambda w, t: good_change(w)]), reviewer=reviewer).run() is JobStatus.PAUSED
    assert ledger.get_node(job_id, "n1").status is not NodeStatus.UNVERIFIED
    assert ledger.attempt_count(job_id, "n1") == 0
    assert "G7 gave no verdict" in ledger.get_job(job_id).error
    assert not any(a.end_sha for a in ledger.attempts(job_id, "n1"))


def _helper_then_break_then(fix):
    def add_helper(work, task):
        (work / "helpers.py").write_text("def double(x):\n    return x * 2\n")
        (work / "tests" / "test_math.py").write_text(
            "from helpers import double\n\ndef test_double():\n    assert double(2) == 4\n")

    def break_helper_quietly(work, task):
        (work / "extra.py").write_text("x = 1\n")
        (work / "tests" / "test_extra.py").write_text("def test_extra():\n    assert True\n")
        (work / "helpers.py").write_text("def double(x):\n    return x * 3\n")

    return FakeExecutor([add_helper, break_helper_quietly, fix, fix, fix])


HELPER_TREE = [NodeSpec(id="feature", goal="helpers and extras"),
               (NodeSpec(id="helper", goal="add a helper"), "feature"),
               (NodeSpec(id="extra", goal="add extra"), "feature")]


def test_a_parent_fix_that_deletes_a_childs_tests_fails(env):
    def delete_the_failing_test(work, task):
        (work / "tests" / "test_math.py").unlink()

    job_id = make_job(env, HELPER_TREE)
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, _helper_then_break_then(delete_the_failing_test)).run() is JobStatus.BLOCKED
    fix_attempt = [a for a in ledger.attempts(job_id, "feature") if a.executor == "fake"][0]
    g6 = {r.gate: r for r in ledger.gate_results(fix_attempt.id)}["G6"]
    assert g6.status is GateStatus.FAIL and g6.summary.startswith("in the fix itself")


def test_a_childs_permissions_carry_up_to_its_parent(env):
    nodes = [NodeSpec(id="deps", goal="bump and use"),
             (NodeSpec(id="bump", goal="bump the lock", allow_protected=["uv.lock"]), "deps"),
             (NodeSpec(id="use", goal="add sub()"), "deps")]
    job_id = make_job(env, nodes)
    executor = FakeExecutor([lambda w, t: (w / "uv.lock").write_text("lock v2\n"), lambda w, t: good_change(w)])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE
    assert env["ledger"].get_node(job_id, "deps").status is NodeStatus.PASSED


def test_a_parent_of_analysis_nodes_passes_without_a_change(env):
    nodes = [NodeSpec(id="survey", goal="look around"),
             (NodeSpec(id="read", goal="read the code", kind="analysis", expects_change=False), "survey")]
    job_id = make_job(env, nodes)
    assert JobRunner(env["ledger"], job_id, FakeExecutor([lambda w, t: None])).run() is JobStatus.DONE
    assert env["ledger"].get_node(job_id, "survey").end_sha == env["ledger"].get_job(job_id).base_sha


def test_a_paused_review_does_not_use_an_attempt(env):
    calls = []

    def reviewer(ctx):
        calls.append(1)
        if len(calls) == 1:
            raise GatePaused("review hit a usage limit", until=time.time() + 5)
        return GateResult("G7", GateStatus.PASS, "reviewer approved")

    clock = {"now": time.time()}
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], review=True)
    runner = JobRunner(env["ledger"], job_id, FakeExecutor([lambda w, t: good_change(w)] * 2),
                       RunnerOptions(pause_poll_s=1), reviewer=reviewer,
                       sleep=lambda s: clock.__setitem__("now", clock["now"] + s), now=lambda: clock["now"])
    assert runner.run() is JobStatus.DONE
    assert [a.outcome for a in env["ledger"].attempts(job_id, "n1")] == [AttemptOutcome.PAUSED, AttemptOutcome.PASSED]
    assert env["ledger"].attempt_count(job_id, "n1") == 1


def test_feedback_never_carries_copy_in_values(env):
    files = dict(BASE_FILES)
    files[".gitignore"] = "__pycache__/\n.env.test\n"
    files["app_env.py"] = "def env_line():\n    return open('.env.test').read().strip()\n"
    files["tests/test_env.py"] = (
        "from app_env import env_line\n\n"
        "def test_env_is_used():\n"
        "    value = env_line()\n"
        "    print('env says', value)\n"
        "    assert value.startswith('DB_')\n")
    source = make_source_repo(env["tmp"] / "src-env", files)
    (source / ".env.test").write_text("DB_PASSWORD=hunter2-very-secret\n")
    job_id = make_job(env, [NodeSpec(id="n1", goal="change env handling")], source=source,
                      copy_in=[".env.test"])
    seen = []

    def break_env(work, task):
        # Passes on base; broken here, so the failing test prints the secret into the output.
        seen.extend(task.feedback)
        (work / "app_env.py").write_text("def env_line():\n    return 'broken ' + open('.env.test').read().strip()\n")

    JobRunner(env["ledger"], job_id, FakeExecutor([break_env] * 3)).run()
    assert seen, "the second attempt should have had feedback"
    assert not any("hunter2-very-secret" in f for f in seen)
    assert any("[redacted]" in f for f in seen)


def test_feedback_names_failed_gates_and_problems():
    results = [
        GateResult("G1", GateStatus.PASS, "1 file changed"),
        GateResult("G2", GateStatus.FAIL, "2 scope problems", {"problems": ["a.py: outside the node's scope"]}),
    ]
    text = feedback_for(results, ExecResult(exit_code=1, error="Reached maximum number of turns (5)"))
    assert "G2 failed: 2 scope problems" in text
    assert "a.py: outside the node's scope" in text
    assert "maximum number of turns" in text
    assert "G1" not in text


def test_profile_commands_become_bash_prefixes():
    profile = Profile(repo_id="x", checks=CHECKS)
    assert profile_commands(profile) == [
        "python3 runtests.py", "python3 -c \"import ast; ast.parse(open('app.py').read())\""]


def test_profile_commands_leave_out_the_junit_argument():
    # Claude Code refuses a Bash rule with a variable in it, so the worker gets
    # the command without the junit file the gates use.
    checks = Checks(test_full='python3 -m pytest tests -q --junitxml="$MULLET_JUNIT"',
                    test_targeted='vendor/bin/phpunit --log-junit "$MULLET_JUNIT" {tests}')
    profile = Profile(repo_id="x", checks=checks)
    assert profile_commands(profile) == ["python3 -m pytest tests -q", "vendor/bin/phpunit"]


def test_jobs_on_the_same_base_and_profile_share_one_baseline(env):
    ledger = env["ledger"]
    first = make_job(env, [NodeSpec(id="sub", goal="add sub()")])
    assert JobRunner(ledger, first, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.DONE
    second = make_job(env, [NodeSpec(id="sub", goal="add sub()")])
    assert JobRunner(ledger, second, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.DONE
    other_checks = CHECKS.model_copy(update={"lint": None})
    third = make_job(env, [NodeSpec(id="sub", goal="add sub()")], checks=other_checks)
    assert JobRunner(ledger, third, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.DONE

    shared = {job: any(e.kind == "baseline_shared" for e in ledger.events(job)) for job in (first, second, third)}
    assert shared == {first: False, second: True, third: False}
    baselines = [next(e.data for e in ledger.events(job) if e.kind == "baseline") for job in (first, second)]
    assert baselines[0] == baselines[1]


def test_a_stale_or_other_commits_shared_baseline_is_not_used(tmp_path):
    from mullet.jobs.runner import BASELINE_SHARE_S, shared_baseline

    path = tmp_path / "b.json"
    path.write_text(json.dumps({"at": 1000.0, "sha": "abc", "runs": {}}))
    assert shared_baseline(path, "abc", now=1000.0 + BASELINE_SHARE_S - 1) is not None
    assert shared_baseline(path, "abc", now=1000.0 + BASELINE_SHARE_S + 1) is None
    assert shared_baseline(path, "def", now=1001.0) is None
    path.write_text("not json")
    assert shared_baseline(path, "abc", now=1001.0) is None


def test_opencode_workers_have_their_own_slots_beside_claudes(env, monkeypatch):
    monkeypatch.setenv("MULLET_API_WORKERS", "2")
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    runner = JobRunner(env["ledger"], job_id, FakeExecutor([]), RunnerOptions(pause_poll_s=0.01))
    claude, api = FakeExecutor([]), FakeExecutor([])
    api.name = "opencode"
    home = Path(os.environ["MULLET_JOBS_DIR"])
    with runner._worker_slot(claude):
        # Claude's one slot is held: two opencode workers still start.
        with runner._worker_slot(api), runner._worker_slot(api):
            assert sorted(p.name for p in home.glob("api-worker-*.lock")) == ["api-worker-0.lock", "api-worker-1.lock"]
            sleeps = []
            runner._sleep = lambda s: (sleeps.append(s), runner._stop.set())  # not request_stop(): that is process-wide
            with pytest.raises(StopRequested):  # a third waits for a slot until the stop arrives
                with runner._worker_slot(api):
                    pass
            assert sleeps


@pytest.mark.parametrize("value, slots", [(None, 3), ("5", 5), ("0", 1), ("lots", 3)])
def test_api_worker_slots_come_from_the_environment(monkeypatch, value, slots):
    from mullet.jobs.runner import api_worker_slots

    if value is None:
        monkeypatch.delenv("MULLET_API_WORKERS", raising=False)
    else:
        monkeypatch.setenv("MULLET_API_WORKERS", value)
    assert api_worker_slots() == slots
