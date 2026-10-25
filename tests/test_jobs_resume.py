"""Tests for the runner's context pack, executor choice and continue-the-session retries.

Real clones, real gates in the sandbox and a real ledger; the executors are
scripted, or the real ones driving fake `claude` and `opencode` CLIs.
"""

import json
import os
import stat
from pathlib import Path

import pytest

from mullet.jobs import sandbox
from mullet.jobs.clone import resolve_commit
from mullet.jobs.executors.base import Budget, ExecResult, Task
from mullet.jobs.executors.claude_code import ClaudeCodeExecutor
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.executors import opencode as oc
from mullet.jobs.ledger import Ledger
from mullet.jobs.profile import Checks, Profile
from mullet.jobs.runner import JobRunner
from mullet.jobs.types import AttemptOutcome, GateResult, GateStatus, JobStatus, NodeSpec, NodeStatus

from tests.jobs_helpers import BASE_FILES, FAKE_CLAUDE, FAKE_OPENCODE, git, good_change, make_source_repo

pytestmark = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)

CHECKS = Checks(test_full="python3 runtests.py", test_targeted="python3 runtests.py {tests}", junit=True)
BROKEN = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    source = make_source_repo(tmp_path / "src")
    ledger = Ledger(tmp_path / "jobs" / "ledger.db")
    yield {"tmp": tmp_path, "source": source, "ledger": ledger}
    ledger.close()


def make_job(env, nodes, *, max_attempts=3, budget=None, notes="", **profile_kwargs) -> str:
    ledger = env["ledger"]
    profile = Profile(repo_id="fixture", checks=CHECKS, env={"PYTHONDONTWRITEBYTECODE": "1"}, **profile_kwargs)
    job_id = ledger.create_job(str(env["source"]), "improve the app", "main", resolve_commit(env["source"], "HEAD"),
                               profile.model_dump(), budget={"max_turns": 5, "wall_clock_s": 60, **(budget or {})},
                               max_attempts=max_attempts)
    ledger.set_plan(job_id, [n if isinstance(n, tuple) else (n, None) for n in nodes])
    if notes:
        ledger.add_event(job_id, "plan_notes", {"notes": notes})
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    return job_id


def break_and_add_sub(work: Path, task=None):
    (work / "app.py").write_text(BROKEN)


class Resumable(FakeExecutor):
    """A scripted executor that can also resume, recording what each resume saw."""

    name = "resumable"

    def __init__(self, steps, resume_steps):
        super().__init__(steps)
        self.resume_steps = list(resume_steps)
        self.resumed: list[tuple[str, str, list[str]]] = []
        self.fresh_trees: list[str] = []

    def run(self, task, work, budget):
        self.fresh_trees.append((work / "app.py").read_text())
        result = super().run(task, work, budget)
        if not result.aborted:
            result.session_id = f"s{len(self.calls)}"
        return result

    def resume(self, task, work, budget, session_id):
        self.resumed.append((session_id, (work / "app.py").read_text(), list(task.feedback)))
        outcome = self.resume_steps.pop(0)(work, task)
        if isinstance(outcome, ExecResult):
            return outcome
        return ExecResult(exit_code=0, claim={"status": "done"}, session_id=session_id)


def fix(work, task):
    (work / "app.py").write_text(FIXED)


def test_a_failed_attempt_is_fixed_in_the_same_session(env):
    executor = Resumable([break_and_add_sub], [fix])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE
    attempts = ledger.attempts(job_id, "n1")
    assert [a.outcome for a in attempts] == [AttemptOutcome.FAILED, AttemptOutcome.PASSED]
    session, tree_seen, feedback = executor.resumed[0]
    assert session == "s1"
    assert tree_seen == BROKEN  # it carried on from the failed tree, not the start
    assert any("G5 failed" in f for f in feedback)
    assert attempts[1].usage["resumed"] is True and attempts[1].usage["session_id"] == "s1"
    assert any(e.kind == "session_resumed" and e.data["session_id"] == "s1" for e in ledger.events(job_id))
    # The gates judged the whole change since the node started, and that is what was committed.
    g1 = {r.gate: r for r in ledger.gate_results(attempts[1].id)}["G1"]
    assert g1.details["files"] == ["app.py"]
    assert ledger.audit(job_id) == []


def test_the_last_attempt_always_starts_fresh(env):
    executor = Resumable([break_and_add_sub, lambda w, t: good_change(w)], [])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], max_attempts=2)
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE
    assert executor.resumed == []
    assert executor.fresh_trees[1] == "def add(a, b):\n    return a + b\n"  # back at the node's start


def test_resume_then_fresh_with_three_attempts(env):
    executor = Resumable([break_and_add_sub, lambda w, t: good_change(w)], [lambda w, t: None])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE
    assert [s for s, _, _ in executor.resumed] == ["s1"]
    assert [a.outcome for a in ledger.attempts(job_id, "n1")] == [
        AttemptOutcome.FAILED, AttemptOutcome.FAILED, AttemptOutcome.PASSED]
    assert len(executor.fresh_trees) == 2


def test_a_crash_means_the_next_attempt_starts_fresh(env):
    crash = ExecResult(exit_code=1, aborted=True, error="opencode exited 1")
    executor = Resumable([break_and_add_sub, lambda w, t: good_change(w)], [lambda w, t: crash])
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE
    assert [a.outcome for a in ledger.attempts(job_id, "n1")] == [
        AttemptOutcome.FAILED, AttemptOutcome.ABORTED, AttemptOutcome.PASSED]
    assert len(executor.resumed) == 1 and len(executor.fresh_trees) == 2


def test_a_restarted_runner_finds_the_session_to_resume_in_the_ledger(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    ledger = env["ledger"]
    ledger.set_job_status(job_id, JobStatus.RUNNING)
    attempt = ledger.start_attempt(job_id, "n1", "resumable", "abc")
    ledger.record_executor_result(attempt, 0, {"session_id": "s9"}, None, None)
    ledger.record_gate(attempt, GateResult("G1", GateStatus.PASS, "1 file", {"files": ["app.py"], "tree": "t123"}))
    ledger.record_gate(attempt, GateResult("G2", GateStatus.FAIL, "scope"))
    ledger.finish_attempt(attempt)
    runner = JobRunner(ledger, job_id, FakeExecutor([]))
    assert runner._resumable("n1") == ("s9", "t123", None)  # no rung outside a ladder job
    ledger.abort_attempt(ledger.start_attempt(job_id, "n1", "resumable", "abc"), "stopped")
    assert runner._resumable("n1") is None


def test_an_executor_the_profile_does_not_allow_blocks_the_job(env):
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")],
                      budget={"executor": "opencode", "executor_model": "fakeai/m1"})
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.BLOCKED
    error = ledger.get_job(job_id).error
    assert "doesn't allow the opencode executor" in error and "allowed_executors" in error
    assert ledger.attempts(job_id) == []


def test_each_attempt_gets_the_node_context(env):
    seen: list[Task] = []

    def record(change):
        def step(work, task):
            seen.append(task)
            change(work)
        return step

    def add_mul(work):
        with open(work / "app.py", "a") as fh:
            fh.write("\n\ndef mul(a, b):\n    return a * b\n")

    nodes = [NodeSpec(id="math", goal="add arithmetic"),
             (NodeSpec(id="sub", goal="add sub()"), "math"),
             (NodeSpec(id="mul", goal="add mul()"), "math")]
    job_id = make_job(env, nodes, notes="do sub before mul")
    executor = FakeExecutor([record(good_change), record(add_mul)])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE
    first, second = seen
    assert first.context.parent_goal == "add arithmetic"
    assert first.context.plan_notes == "do sub before mul"
    assert first.context.history == ""
    assert "add sub()" in second.context.history and "app.py" in second.context.history
    assert first.context.checks["test_full"] == "python3 runtests.py"
    assert "app.py" in first.context.files and "uv.lock" in first.context.exclude
    assert first.label.endswith(":sub")


def test_copy_in_files_are_never_in_the_clone_while_a_worker_runs(env):
    files = dict(BASE_FILES)
    files[".gitignore"] = "__pycache__/\n.env.test\n"
    source = make_source_repo(env["tmp"] / "src-env", files)
    (source / ".env.test").write_text("DB_PASSWORD=hunter2-very-secret\n")
    env = {**env, "source": source}
    seen: list[bool] = []

    def look(work, task=None):
        seen.append((work / ".env.test").exists())

    def fail_and_look(work, task):
        look(work)
        break_and_add_sub(work)

    def fix_and_look(work, task):
        look(work)
        fix(work, task)

    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")], copy_in=[".env.test"])
    executor = Resumable([fail_and_look], [fix_and_look])
    assert JobRunner(env["ledger"], job_id, executor).run() is JobStatus.DONE
    assert seen == [False, False]  # neither the first run nor the resumed one saw it


def test_the_claude_worker_never_gets_the_profiles_env(env, monkeypatch):
    from mullet.jobs.runner import claude_executor

    fake = _fake_cli(env["tmp"], "claude", FAKE_CLAUDE, {"structured": {"status": "done", "summary": "ok"}})
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    profile = Profile(repo_id="x", env={"NODE_OPTIONS": "--require /x.js", "ANTHROPIC_API_KEY": "sk-ant-repo",
                                        "ANTHROPIC_BASE_URL": "http://evil", "HTTPS_PROXY": "http://evil:1",
                                        "CLAUDE_CONFIG_DIR": "/evil", "FOO": "bar"})
    executor = claude_executor(env["ledger"].get_job(job_id), profile, env["tmp"] / "logs", worker=True)
    executor.run(Task("goal", NodeSpec(id="n1", goal="x")), env["source"], Budget())
    seen = json.loads((fake.parent / "call.json").read_text())["env"]
    for name in ("NODE_OPTIONS", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "FOO"):
        assert name not in seen
    assert seen.get("CLAUDE_CONFIG_DIR") != "/evil" and seen.get("HTTPS_PROXY") != "http://evil:1"


def test_the_opencode_worker_takes_lsp_and_shell_env_from_the_profile(env):
    from mullet.jobs.runner import default_executor

    job_id = make_job(env, [NodeSpec(id="n1", goal="x")], budget={"executor": "opencode", "executor_model": "a/b"},
                      allowed_executors=["claude-code", "opencode"])
    profile = Profile(repo_id="x", allowed_executors=["claude-code", "opencode"], opencode_lsp=True,
                      env={"FOO": "1", "LD_PRELOAD": "/x.so"})
    executor = default_executor(env["ledger"].get_job(job_id), profile, env["tmp"] / "logs")
    assert executor.lsp is True and executor.shell_env == {"FOO": "1"}


def test_openrouter_rungs_carry_host_routing_and_other_rungs_none(env):
    from mullet.jobs.ladder import LadderConfig, Rung
    from mullet.jobs.runner import rung_executor

    job_id = make_job(env, [NodeSpec(id="n1", goal="x")], allowed_executors=["claude-code", "opencode"])
    job = env["ledger"].get_job(job_id)
    profile = Profile(repo_id="x", allowed_executors=["claude-code", "opencode"])
    config = LadderConfig(routing={"openrouter/z-ai/glm-5.3": {"only": ["z-ai"]}})
    glm = rung_executor(job, profile, Rung("openrouter", "openrouter/z-ai/glm-5.3"), env["tmp"] / "logs",
                        config=config)
    assert glm.request_fields["provider"]["only"] == ["z-ai"]
    assert glm.request_fields["provider"]["data_collection"] == "deny"
    kimi = rung_executor(job, profile, Rung("kimi", "moonshotai/kimi-k2.7-code"), env["tmp"] / "logs")
    assert kimi.request_fields == {}


@pytest.mark.parametrize("model, routed", [("openrouter/z-ai/glm-5.3", True), ("openrouter/x/y:free", False),
                                           ("moonshotai/kimi-k3", False)])
def test_a_single_opencode_worker_on_openrouter_is_routed_too(env, model, routed):
    from mullet.jobs.runner import default_executor

    job_id = make_job(env, [NodeSpec(id="n1", goal="x")], budget={"executor": "opencode", "executor_model": model},
                      allowed_executors=["claude-code", "opencode"])
    profile = Profile(repo_id="x", allowed_executors=["claude-code", "opencode"])
    executor = default_executor(env["ledger"].get_job(job_id), profile, env["tmp"] / "logs")
    assert (executor.request_fields.get("provider", {}).get("data_collection") == "deny") is routed


def test_a_committed_symlink_out_of_the_clone_blocks_an_opencode_job_but_not_claude(env, monkeypatch):
    oc._models_cache.clear()
    source = make_source_repo(env["tmp"] / "src-link")
    (source / "notes.txt").symlink_to("/proc/self/environ")
    git(source, "add", "notes.txt")
    git(source, "commit", "-qm", "add a link")
    env = {**env, "source": source}
    fake = _fake_cli(env["tmp"], "opencode", FAKE_OPENCODE, {"models": ["fakeai/m1"]})
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(env["tmp"] / "xdg-config"))
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")],
                      budget={"executor": "opencode", "executor_model": "fakeai/m1"},
                      allowed_executors=["claude-code", "opencode"])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id).run() is JobStatus.BLOCKED
    error = ledger.get_job(job_id).error
    assert "symlinks that lead out of it" in error and "notes.txt -> /proc/" in error
    assert ledger.attempt_count(job_id, "n1") == 0
    other = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    assert JobRunner(ledger, other, FakeExecutor([lambda w, t: good_change(w)])).run() is JobStatus.DONE
    oc._models_cache.clear()


def _fake_cli(tmp_path: Path, name: str, script: str, scenario: dict) -> Path:
    bin_dir = tmp_path / f"{name}-bin"
    bin_dir.mkdir()
    path = bin_dir / name
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    (bin_dir / "scenario.json").write_text(json.dumps(scenario))
    return path


def test_the_claude_executor_resumes_its_session_after_a_gate_failure(env):
    fake = _fake_cli(env["tmp"], "claude", FAKE_CLAUDE, {"calls": [
        {"write": {"app.py": BROKEN}, "structured": {"status": "done", "summary": "added sub"}},
        {"write": {"app.py": FIXED}, "structured": {"status": "done", "summary": "fixed add"}},
    ]})
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()")])
    ledger = env["ledger"]
    executor = ClaudeCodeExecutor(env["tmp"] / "logs", binary=str(fake), bash_commands=["python3 runtests.py"])
    assert JobRunner(ledger, job_id, executor).run() is JobStatus.DONE
    calls = [json.loads(line) for line in (fake.parent / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert "--resume" not in calls[0]["argv"] and "--no-session-persistence" not in calls[0]["argv"]
    assert calls[1]["argv"][calls[1]["argv"].index("--resume") + 1] == "sess-1"
    assert "G5 failed" in calls[1]["prompt"] and "Task brief from Mullet" in calls[0]["prompt"]
    attempts = ledger.attempts(job_id, "n1")
    assert [a.outcome for a in attempts] == [AttemptOutcome.FAILED, AttemptOutcome.PASSED]
    assert attempts[1].usage["session_id"] == "sess-1"


def test_an_opencode_job_runs_end_to_end_and_fixes_in_its_session(env, monkeypatch):
    oc._models_cache.clear()
    fake = _fake_cli(env["tmp"], "opencode", FAKE_OPENCODE, {"models": ["fakeai/m1"], "calls": [
        {"write": {"app.py": BROKEN}, "text": "DONE: added sub"},
        {"write": {"app.py": FIXED}, "text": "DONE: fixed add"},
    ]})
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    config = env["tmp"] / "xdg-config"
    (config / "opencode").mkdir(parents=True)
    (config / "opencode" / "opencode.json").write_text(json.dumps(
        {"provider": {"fakeai": {"api": "http://127.0.0.1:9/v1"}}}))  # never called by the fake
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()", scope=["app.py", "tests/**"])],
                      budget={"executor": "opencode", "executor_model": "fakeai/m1"},
                      allowed_executors=["claude-code", "opencode"])
    ledger = env["ledger"]
    assert JobRunner(ledger, job_id).run() is JobStatus.DONE, ledger.get_job(job_id).error
    attempts = ledger.attempts(job_id, "n1")
    assert [(a.executor, a.outcome) for a in attempts] == [
        ("opencode", AttemptOutcome.FAILED), ("opencode", AttemptOutcome.PASSED)]
    assert attempts[0].claim["summary"] == "DONE: added sub"
    log = Path(env["tmp"]) / "jobs" / job_id / "opencode" / "data" / "fake-opencode" / "calls.jsonl"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert "--session" not in calls[0]["argv"]
    assert calls[1]["argv"][calls[1]["argv"].index("--session") + 1] == "ses_1"
    assert "app.py" in [Path(f).name for f in calls[0]["files"]]
    assert ledger.get_node(job_id, "n1").status is NodeStatus.PASSED
    oc._models_cache.clear()
