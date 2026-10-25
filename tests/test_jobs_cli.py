"""Tests for ``mullet job`` (mullet/cli/job_cli.py) up to the point a runner starts.

Running jobs is covered by tests/test_jobs_runner.py; here the CLI's own
behaviour is checked: planning, approval, refusals, reports and handing a
finished branch back.
"""

import fcntl
import json
import os
from pathlib import Path

import pytest
import yaml

from mullet.cli import job_cli
from mullet.cli.job_cli import cmd_job
from mullet.jobs import sandbox
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.ledger import Ledger
from mullet.jobs.profile import Checks, Profile, repo_id, save_profile
from mullet.jobs.runner import JobRunner
from mullet.jobs.types import JobStatus

from tests.jobs_helpers import git, good_change, make_source_repo


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MULLET_PROFILES_DIR", str(tmp_path / "profiles"))
    return make_source_repo(tmp_path / "src")


def store_profile(repo: Path) -> None:
    """A stored profile with the fixture's checks, as a user would write one."""
    save_profile(Profile(repo_id=repo_id(repo), env={"PYTHONDONTWRITEBYTECODE": "1"}, checks=Checks(
        test_full="python3 runtests.py", test_targeted="python3 runtests.py {tests}", junit=True)))


def _job_id(out: str) -> str:
    return next(line.split()[1] for line in out.splitlines() if line.startswith("Job "))


def plan(repo: Path, capsys, *extra: str) -> str:
    assert cmd_job(["plan", "add a sub function", "--repo", str(repo), "--single", "--no-review", *extra]) == 0
    return _job_id(capsys.readouterr().out)


def test_plan_shows_the_plan_and_checks_and_waits_for_approval(repo, capsys, tmp_path):
    assert cmd_job(["plan", "add a sub function", "--repo", str(repo), "--single", "--scope", "app.py",
                    "--scope", "tests/**"]) == 0
    out = capsys.readouterr().out
    assert "add a sub function" in out
    assert "scope: app.py, tests/**" in out
    assert "Approve with: mullet job approve" in out
    assert "Saved a detected profile" in out
    with Ledger() as ledger:
        assert ledger.get_job(_job_id(out)).status is JobStatus.AWAITING_APPROVAL


def test_plan_warns_about_uncommitted_changes(repo, capsys):
    (repo / "scratch.txt").write_text("wip\n")
    plan(repo, capsys)
    # plan() already consumed the output; run once more to see the note.
    assert cmd_job(["plan", "again", "--repo", str(repo), "--single"]) == 0
    assert "uncommitted changes" in capsys.readouterr().out


def test_plan_file_with_several_nodes(repo, capsys, tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump({"nodes": [
        {"id": "sub", "goal": "add sub", "scope": ["app.py", "tests/**"]},
        {"id": "docs", "goal": "document it", "kind": "mechanical", "scope": ["README.md"]},
    ]}))
    job_id = plan(repo, capsys, "--plan-file", str(plan_file))
    with Ledger() as ledger:
        assert [n.node_id for n in ledger.nodes(job_id)] == ["sub", "docs"]


def test_a_bad_plan_file_is_refused_before_any_job_exists(repo, capsys, tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump({"nodes": [
        {"id": "a", "parent": None, "goal": "x", "kind": "design", "scope": ["**"], "max_lines": 10**9}]}))
    assert cmd_job(["plan", "goal", "--repo", str(repo), "--plan-file", str(plan_file)]) == 1
    assert "max_lines" in capsys.readouterr().err
    with Ledger() as ledger:
        assert ledger.list_jobs() == []


def test_run_refuses_an_unapproved_job(repo, capsys):
    job_id = plan(repo, capsys)
    assert cmd_job(["run", job_id]) == 1
    assert "only approved, paused or blocked jobs can run" in capsys.readouterr().err


def test_run_refuses_without_a_sandbox(repo, capsys, monkeypatch):
    job_id = plan(repo, capsys)
    cmd_job(["approve", job_id])
    monkeypatch.setattr(job_cli.sandbox, "check", lambda: sandbox.SandboxCheck(False, "bwrap", "no userns"))
    assert cmd_job(["run", job_id]) == 1
    assert "sandbox is unavailable: no userns" in capsys.readouterr().err


def test_run_refuses_without_socat_on_linux(repo, capsys, monkeypatch):
    job_id = plan(repo, capsys)
    cmd_job(["approve", job_id])
    real_which = job_cli.shutil.which
    monkeypatch.setattr(job_cli.shutil, "which", lambda name: None if name == "socat" else real_which(name))
    monkeypatch.setattr(job_cli.sys, "platform", "linux")
    if not real_which("claude"):
        pytest.skip("claude CLI not installed")
    assert cmd_job(["run", job_id]) == 1
    assert "socat" in capsys.readouterr().err


def test_unknown_job_is_a_clean_error(repo, capsys):
    assert cmd_job(["show", "20990101-000000-abcdef"]) == 1
    assert "mullet job:" in capsys.readouterr().err


@pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)
def test_finished_job_reports_and_merges_into_a_new_branch_only(repo, capsys):
    store_profile(repo)
    job_id = plan(repo, capsys)
    cmd_job(["approve", job_id])
    capsys.readouterr()

    def change_and_claim(work, task):
        good_change(work)
        return {"status": "done", "summary": "added sub() with a test"}

    with Ledger() as ledger:
        assert JobRunner(ledger, job_id, FakeExecutor([change_and_claim])).run() is JobStatus.DONE

    assert cmd_job(["status", job_id]) == 0
    assert "done" in capsys.readouterr().out
    assert cmd_job(["report", job_id]) == 0
    report = capsys.readouterr().out
    assert "n1: passed" in report
    assert "| G5 | pass |" in report
    assert "a claim, not checked" in report
    assert "Clean: every passed node is backed" in report

    head_before = git(repo, "rev-parse", "HEAD")
    assert cmd_job(["merge", job_id, "--yes"]) == 0
    out = capsys.readouterr().out
    assert "app.py" in out and f"mullet/{job_id}" in out
    assert git(repo, "rev-parse", "HEAD") == head_before
    assert git(repo, "branch", "--show-current") == "main"
    assert "def sub" in git(repo, "show", f"mullet/{job_id}:app.py")
    with Ledger() as ledger:
        assert ledger.get_job(job_id).status is JobStatus.MERGED


@pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)
def test_discard_deletes_the_clone(repo, capsys):
    job_id = plan(repo, capsys)
    cmd_job(["approve", job_id])
    with Ledger() as ledger:
        JobRunner(ledger, job_id, FakeExecutor([lambda w, t: good_change(w)])).run()
    clone_root = job_cli.job_dir(job_id) / "clone"
    assert clone_root.exists()
    assert cmd_job(["discard", job_id]) == 0
    assert not clone_root.exists()
    with Ledger() as ledger:
        assert ledger.get_job(job_id).status is JobStatus.DISCARDED


def test_a_job_left_running_by_a_dead_runner_can_be_resumed(repo, capsys, monkeypatch, fake_claude):
    monkeypatch.setattr(job_cli.sandbox, "check", lambda: sandbox.SandboxCheck(True, "bwrap", ""))
    job_id = plan(repo, capsys)
    with Ledger() as ledger:
        ledger.approve(job_id)
        ledger.set_job_status(job_id, JobStatus.RUNNING)  # what a kill -9 leaves behind
    assert not job_cli._runner_alive(job_id)
    job_cli.preflight(JobStatus.RUNNING)  # no error


def _hold_runner_lock(job_id):
    path = job_cli.job_dir(job_id) / "runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_a_foreground_runner_is_seen_by_run_discard_and_stop(repo, capsys):
    job_id = plan(repo, capsys)
    cmd_job(["approve", job_id])
    capsys.readouterr()
    fd = _hold_runner_lock(job_id)  # a runner with no usable pid file, as with --foreground
    try:
        assert job_cli._runner_alive(job_id)
        assert cmd_job(["run", job_id]) == 1
        assert "already working" in capsys.readouterr().err
        assert cmd_job(["discard", job_id]) == 1
        assert "stop the job first" in capsys.readouterr().err
        assert cmd_job(["stop", job_id]) == 1
        assert "pid can't be confirmed" in capsys.readouterr().err
    finally:
        os.close(fd)


def test_stop_ignores_a_recycled_pid(repo, capsys):
    job_id = plan(repo, capsys)
    fd = _hold_runner_lock(job_id)
    try:
        # Our own pid, but with a start time that doesn't match: a reused pid.
        (job_cli.job_dir(job_id) / "runner.pid").write_text(f"{os.getpid()} 12345.0")
        assert job_cli._runner_pid(job_id) is None
    finally:
        os.close(fd)


@pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)
def test_merge_refuses_existing_or_invalid_branch_names(repo, capsys):
    store_profile(repo)
    job_id = plan(repo, capsys)
    cmd_job(["approve", job_id])
    with Ledger() as ledger:
        JobRunner(ledger, job_id, FakeExecutor([lambda w, t: good_change(w)])).run()
    capsys.readouterr()
    git(repo, "branch", "taken")
    assert cmd_job(["merge", job_id, "--yes", "--branch", "taken"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert cmd_job(["merge", job_id, "--yes", "--branch", "bad..name"]) == 1
    assert "not a valid branch name" in capsys.readouterr().err
    assert cmd_job(["merge", job_id, "--yes", "--branch", "main"]) == 1
    with Ledger() as ledger:
        assert ledger.get_job(job_id).status is JobStatus.DONE


def test_plan_asks_the_planner_and_the_plan_round_trips_through_yaml(repo, capsys, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from mullet.jobs.executors.base import ExecResult
    seen = {}

    def ask(prompt, cwd, schema, budget, model=None, label="ask"):
        seen.update(cwd=cwd, label=label, prompt=prompt)
        return ExecResult(exit_code=0, claim={"notes": "two steps", "nodes": [
            {"id": "api", "parent": None, "goal": "add the api", "kind": "design", "scope": ["app.py", "tests/**"]},
            {"id": "api-tests", "parent": "api", "goal": "test it", "kind": "design", "scope": ["tests/**"],
             "acceptance": ["python3 runtests.py"]},
        ]})

    monkeypatch.setattr(job_cli, "claude_executor", lambda job, profile, logs: SimpleNamespace(ask=ask))
    monkeypatch.setattr(job_cli.shutil, "which", lambda name: "/usr/bin/" + name)
    assert cmd_job(["plan", "add an api", "--repo", str(repo)]) == 0
    out = capsys.readouterr().out
    job_id = _job_id(out)
    assert "Planner's notes: two steps" in out
    assert seen["label"] == "plan" and seen["cwd"] == job_cli.job_dir(job_id) / "clone" / "work"
    assert "app.py" in seen["prompt"]

    assert cmd_job(["show", job_id, "--yaml"]) == 0
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(capsys.readouterr().out)
    assert cmd_job(["plan", "add an api", "--repo", str(repo), "--plan-file", str(plan_file)]) == 0
    second = _job_id(capsys.readouterr().out)
    with Ledger() as ledger:
        first_nodes = [(n.spec, n.parent_id) for n in ledger.nodes(job_id)]
        second_nodes = [(n.spec, n.parent_id) for n in ledger.nodes(second)]
        assert first_nodes == second_nodes
        assert ledger.get_job(job_id).budget["review"] is True


def test_a_failed_plan_marks_the_job_failed(repo, capsys, monkeypatch):
    from types import SimpleNamespace

    from mullet.jobs.executors.base import ExecResult

    def ask(prompt, cwd, schema, budget, model=None, label="ask"):
        return ExecResult(exit_code=1, error="Reached maximum number of turns (40)")

    monkeypatch.setattr(job_cli, "claude_executor", lambda job, profile, logs: SimpleNamespace(ask=ask))
    monkeypatch.setattr(job_cli.shutil, "which", lambda name: "/usr/bin/" + name)
    assert cmd_job(["plan", "add an api", "--repo", str(repo)]) == 1
    assert "planning failed" in capsys.readouterr().err
    with Ledger() as ledger:
        assert ledger.list_jobs()[0].status is JobStatus.FAILED


@pytest.fixture
def fake_opencode(tmp_path, monkeypatch):
    """A fake `opencode` on PATH that lists two models, with the user's opencode dirs out of reach."""
    from mullet.jobs.executors import opencode as oc
    from tests.jobs_helpers import FAKE_OPENCODE

    bin_dir = tmp_path / "oc-bin"
    bin_dir.mkdir()
    script = bin_dir / "opencode"
    script.write_text(FAKE_OPENCODE)
    script.chmod(0o755)
    (bin_dir / "scenario.json").write_text(json.dumps({"models": ["fakeai/m1", "openrouter/moonshotai/kimi-k2.7-code"]}))
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    oc._models_cache.clear()
    yield bin_dir
    oc._models_cache.clear()


def allow_opencode(repo: Path) -> None:
    save_profile(Profile(repo_id=repo_id(repo), allowed_executors=["claude-code", "opencode"], checks=Checks(
        test_full="python3 runtests.py", test_targeted="python3 runtests.py {tests}", junit=True)))


def test_plan_refuses_an_executor_the_profile_does_not_allow(repo, capsys, fake_opencode):
    assert cmd_job(["plan", "goal", "--repo", str(repo), "--single", "--executor", "opencode",
                    "--executor-model", "fakeai/m1"]) == 1
    err = capsys.readouterr().err
    assert "doesn't allow the opencode executor" in err and "allowed_executors" in err
    assert str(job_cli.profile_path(repo_id(repo))) in err
    with Ledger() as ledger:
        assert ledger.list_jobs() == []


def test_plan_with_opencode_checks_the_model_before_any_job_exists(repo, capsys, fake_opencode):
    allow_opencode(repo)
    assert cmd_job(["plan", "goal", "--repo", str(repo), "--single", "--executor", "opencode",
                    "--executor-model", "moonshot/kimi-k2.7-code"]) == 1
    err = capsys.readouterr().err
    assert "no model 'moonshot/kimi-k2.7-code'" in err and "openrouter/moonshotai/kimi-k2.7-code" in err
    assert cmd_job(["plan", "goal", "--repo", str(repo), "--single", "--executor", "opencode"]) == 1
    assert "--executor-model" in capsys.readouterr().err
    with Ledger() as ledger:
        assert ledger.list_jobs() == []
    job_id = plan(repo, capsys, "--executor", "opencode", "--executor-model", "fakeai/m1")
    with Ledger() as ledger:
        budget = ledger.get_job(job_id).budget
        assert budget["executor"] == "opencode" and budget["executor_model"] == "fakeai/m1"
        assert "Worker: opencode with fakeai/m1; files pre-loaded" in job_cli.show(ledger, job_id)


@pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)
def test_preflight_checks_the_jobs_own_executor(repo, capsys, fake_opencode, monkeypatch):
    allow_opencode(repo)
    job_id = plan(repo, capsys, "--executor", "opencode", "--executor-model", "fakeai/m1")
    real_which = job_cli.shutil.which
    monkeypatch.setattr(job_cli.shutil, "which", lambda name: None if name == "claude" else real_which(name))
    with Ledger() as ledger:
        ledger.approve(job_id)
        job = ledger.get_job(job_id)
    job_cli.preflight(job.status, job)  # no Claude needed: opencode works and review is off
    monkeypatch.setattr(job_cli.shutil, "which", lambda name: None if name in ("claude", "socat") else real_which(name))
    with pytest.raises(job_cli.CliError, match="socat"):  # opencode reaches its provider through it
        job_cli.preflight(job.status, job)
    monkeypatch.setattr(job_cli.shutil, "which", lambda name: None)
    with pytest.raises(job_cli.CliError, match="opencode is not on PATH"):
        job_cli.preflight(job.status, job)


def test_planner_notes_reach_the_ledger_and_survive_a_plan_file(repo, capsys, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from mullet.jobs.executors.base import ExecResult

    def ask(prompt, cwd, schema, budget, model=None, label="ask"):
        return ExecResult(exit_code=0, claim={"notes": "watch the rounding", "nodes": [
            {"id": "n1", "parent": None, "goal": "do it", "kind": "design", "scope": ["app.py"]}]})

    monkeypatch.setattr(job_cli, "claude_executor", lambda job, profile, logs: SimpleNamespace(ask=ask))
    monkeypatch.setattr(job_cli.shutil, "which", lambda name: "/usr/bin/" + name)
    assert cmd_job(["plan", "goal", "--repo", str(repo)]) == 0
    job_id = _job_id(capsys.readouterr().out)
    with Ledger() as ledger:
        assert [e.data for e in ledger.events(job_id) if e.kind == "plan_notes"] == [{"notes": "watch the rounding"}]
    assert cmd_job(["show", job_id, "--yaml"]) == 0
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(capsys.readouterr().out)
    assert yaml.safe_load(plan_file.read_text())["notes"] == "watch the rounding"
    second = plan(repo, capsys, "--plan-file", str(plan_file))
    with Ledger() as ledger:
        assert [e.data["notes"] for e in ledger.events(second) if e.kind == "plan_notes"] == ["watch the rounding"]


@pytest.fixture
def ladder_user(repo, tmp_path, monkeypatch):
    """A user whose config has a ladder, a Moonshot key, and a stored profile that allows opencode."""
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(
        "ladder:\n  kimi: moonshotai/kimi-k2.7-code\n  openrouter: []\n  max_spend: 1.5\n")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-moonshot")
    monkeypatch.setattr("mullet.core.env.dotenv_value", lambda name: None)
    save_profile(Profile(repo_id=repo_id(repo), env={"PYTHONDONTWRITEBYTECODE": "1"},
                         allowed_executors=["claude-code", "opencode"],
                         checks=Checks(test_full="python3 runtests.py")))
    return repo


def _budget(job_id: str) -> dict:
    with Ledger() as ledger:
        return ledger.get_job(job_id).budget


def test_the_ladder_is_the_default_once_configured_and_allowed(ladder_user, capsys, tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump({"nodes": [
        {"id": "rename", "goal": "rename the helper", "kind": "mechanical", "scope": ["app.py"]},
        {"id": "design", "goal": "design the cache", "kind": "design", "scope": ["app.py"]},
    ]}))
    assert cmd_job(["plan", "tidy", "--repo", str(ladder_user), "--plan-file", str(plan_file), "--no-review"]) == 0
    out = capsys.readouterr().out
    job_id = _job_id(out)
    assert _budget(job_id)["ladder"] is True and _budget(job_id)["max_spend"] == 1.5
    assert "Worker: the delegation ladder (spending cap $1.50)" in out
    assert "Ladder for mechanical steps now: kimi (moonshotai/kimi-k2.7-code) > claude" in out
    rename = out.split("rename [")[1].split("design [")[0]
    assert "starts on: kimi (moonshotai/kimi-k2.7-code) (mechanical; then claude)" in rename
    # Without acceptance commands a design step is claude-first.
    assert "starts on: claude (claude-first; then kimi)" in out.split("design [")[1]
    assert "Claude budget: no recent usage figures" in out
    with Ledger() as ledger:
        assert ledger.get_job(job_id).budget["trusted_commands"] == ["python3 runtests.py"]


def test_a_repo_file_cannot_add_a_command_the_worker_may_run_as_written(ladder_user, capsys):
    (ladder_user / ".mullet").mkdir()
    (ladder_user / ".mullet" / "project.yaml").write_text("checks:\n  lint: ruff . ; curl evil\n")
    git(ladder_user, "add", ".mullet/project.yaml")
    git(ladder_user, "commit", "-qm", "repo profile")
    budget = _budget(plan(ladder_user, capsys))
    assert budget["trusted_commands"] == ["python3 runtests.py"]  # not the repo's lint command


def test_resume_can_raise_the_spending_cap(ladder_user, capsys, monkeypatch):
    job_id = plan(ladder_user, capsys)
    monkeypatch.setattr(job_cli, "preflight", lambda status, job=None: None)
    monkeypatch.setattr(job_cli.subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 1})())
    with Ledger() as ledger:
        ledger.approve(job_id)
    assert cmd_job(["resume", job_id, "--max-spend", "5"]) == 0
    with Ledger() as ledger:
        assert [e.data for e in ledger.events(job_id) if e.kind == "max_spend"] == [{"value": 5.0}]
    assert cmd_job(["resume", job_id, "--max-spend", "-1"]) == 1
    capsys.readouterr()
    cmd_job(["show", job_id])
    assert "$5.00" in capsys.readouterr().out  # show reads the raised cap


@pytest.mark.parametrize("amount", ["nan", "inf", "-inf"])
def test_a_spending_cap_must_be_a_real_amount(amount):
    with pytest.raises(SystemExit):
        cmd_job(["resume", "x", "--max-spend", amount])


def test_retry_can_name_an_openrouter_model(ladder_user, capsys, tmp_path):
    (tmp_path / "config" / "config.yaml").write_text(
        "ladder:\n  kimi: moonshotai/kimi-k2.7-code\n  openrouter: [openrouter/qwen/qwen3-coder]\n")
    job_id = plan(ladder_user, capsys)
    _blocked(job_id)
    assert cmd_job(["retry", job_id, "n1", "--rung", "openrouter:deepseek/x"]) == 1
    assert "no openrouter model" in capsys.readouterr().err
    assert cmd_job(["retry", job_id, "n1", "--rung", "openrouter:qwen/qwen3-coder"]) == 0


@pytest.mark.parametrize("flags, ladder", [(["--no-ladder"], False), (["--executor", "claude"], False),
                                           (["--ladder", "--executor", "claude"], True)])
def test_the_ladder_can_be_turned_off_or_on_per_job(ladder_user, capsys, flags, ladder):
    assert _budget(plan(ladder_user, capsys, *flags))["ladder"] is ladder


def test_max_spend_is_stored_with_the_job(ladder_user, capsys):
    assert _budget(plan(ladder_user, capsys, "--max-spend", "0.75"))["max_spend"] == 0.75


def test_without_a_ladder_config_jobs_stay_single_executor(repo, capsys):
    store_profile(repo)
    assert _budget(plan(repo, capsys))["ladder"] is False


def test_ladder_is_refused_when_the_profile_keeps_code_on_claude(repo, capsys):
    store_profile(repo)  # claude-code only
    assert cmd_job(["plan", "x", "--repo", str(repo), "--single", "--ladder"]) == 1
    assert "allowed_executors" in capsys.readouterr().err
    with Ledger() as ledger:
        assert ledger.list_jobs() == []


def test_ladder_and_a_single_opencode_executor_conflict(ladder_user, capsys):
    assert cmd_job(["plan", "x", "--repo", str(ladder_user), "--single", "--ladder", "--executor", "opencode"]) == 1
    assert "drop --executor opencode" in capsys.readouterr().err


def test_job_ladder_prints_rungs_budget_and_the_repo(ladder_user, capsys, monkeypatch):
    monkeypatch.chdir(ladder_user)
    assert cmd_job(["ladder", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Delegation ladder (from" in out and "config.yaml" in out
    assert "Claude budget: no recent usage figures (treated as medium)" in out
    assert "Spending cap per job on paid rungs: $1.50" in out
    assert "free        not configured" in out and "local       not configured" in out
    kimi = next(line for line in out.splitlines() if line.strip().startswith("kimi"))
    assert "moonshotai/kimi-k2.7-code" in kimi and "key: found" in kimi and "paid" in kimi
    assert any(line.strip().startswith("claude") for line in out.splitlines())
    assert "data class private; allowed executors claude-code, opencode; ladder auto" in out


def test_job_ladder_checks_model_ids_when_online(ladder_user, capsys):
    from mullet.jobs.ladder import Ladder, parse_config

    ladder = Ladder(config=parse_config({"kimi": "moonshotai/kimi-k2.7-code", "openrouter": ["openrouter/x/y"]}),
                    usage=lambda: None, has_key=lambda rung: rung.name == "kimi")
    out = job_cli.ladder_report(ladder, None, model_status=lambda m: "offered by opencode" if "kimi" in m
                                else "not usable: opencode has no model")
    kimi, router = [line for line in out.splitlines() if line.strip().startswith(("kimi", "openrouter"))]
    assert "key: found" in kimi and "offered by opencode" in kimi
    assert "key: missing" in router and "not usable" in router


def _blocked(job_id: str) -> None:
    with Ledger() as ledger:
        ledger.approve(job_id)
        ledger.set_job_status(job_id, JobStatus.RUNNING)
        ledger.block_node(job_id, "n1", "ran out of rungs")
        ledger.set_job_status(job_id, JobStatus.BLOCKED, "node n1 ran out of rungs")


def test_retry_records_fresh_attempts_on_a_named_rung(ladder_user, capsys):
    job_id = plan(ladder_user, capsys)
    assert cmd_job(["retry", job_id, "n1", "--rung", "claude"]) == 1  # not blocked yet
    assert "only a blocked or paused job" in capsys.readouterr().err
    _blocked(job_id)
    assert cmd_job(["retry", job_id, "n1", "--rung", "gemini"]) == 1
    assert cmd_job(["retry", job_id, "n1", "--rung", "claude"]) == 0
    assert f"mullet job resume {job_id}" in capsys.readouterr().out
    with Ledger() as ledger:
        assert [e.data for e in ledger.events(job_id) if e.kind == "node_retry"] == [
            {"node_id": "n1", "rung": "claude", "after_attempt": 0}]


def test_retry_with_a_rung_needs_a_ladder_job(repo, capsys):
    store_profile(repo)
    job_id = plan(repo, capsys)
    _blocked(job_id)
    assert cmd_job(["retry", job_id, "n1", "--rung", "claude"]) == 1
    assert "needs a job planned with the ladder" in capsys.readouterr().err
    assert cmd_job(["retry", job_id, "n1"]) == 0
    assert "one more attempt" in capsys.readouterr().out


def test_list_and_tail(repo, capsys):
    job_id = plan(repo, capsys)
    assert cmd_job(["list"]) == 0
    assert job_id in capsys.readouterr().out
    assert cmd_job(["tail", job_id]) == 0
    events = capsys.readouterr().out
    assert "job_created" in events and "plan_set" in events
