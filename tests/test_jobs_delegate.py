"""Tests for mullet.jobs.delegate.

Real git repositories, a real ledger and injectable ladder config; no models
or network.
"""

import subprocess
import time
from pathlib import Path

import pytest

from mullet.jobs import children
from mullet.jobs import delegate as delegate_mod
from mullet.jobs import machine, sandbox
from mullet.jobs.delegate import create_delegation_job, delegate, restrict_ladder
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.ladder import Ladder, MachineState, PublicCheck, parse_config, plan_rungs, spend_state
from mullet.jobs.ledger import Ledger
from mullet.jobs.profile import Checks, Profile, repo_id, save_profile
from mullet.jobs.types import JobStatus, NodeSpec, NodeStatus

from tests.jobs_helpers import git, good_change, make_source_repo

KIMI = "moonshotai/kimi-k2.7-code"
QWEN = "openrouter/qwen/qwen3-coder"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(machine.paths, "state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr("mullet.jobs.profile.profiles_dir", lambda: tmp_path / "profiles")
    source = make_source_repo(tmp_path / "src")
    ledger = Ledger(tmp_path / "jobs" / "ledger.db")
    yield {"tmp": tmp_path, "source": source, "ledger": ledger}
    ledger.close()


def make_ladder(config=None):
    return Ladder(
        config=parse_config(config if config is not None else {"kimi": KIMI, "openrouter": [QWEN]}),
        usage=lambda: None,
        machine=lambda c: MachineState(False, "fake machine"),
        public_check=lambda repo, sha, profile: PublicCheck(False, "fake public check"),
        has_key=lambda rung: True,
    )


def allow_opencode(source: Path) -> None:
    save_profile(Profile(
        repo_id=repo_id(source), allowed_executors=["claude-code", "opencode"], checks=Checks(),
    ))


def test_job_is_approved_with_one_node_and_given_scope(env):
    job_id = create_delegation_job(
        env["ledger"], task="add a helper", paths=["app.py"], cwd=str(env["source"]),
        timeout_min=20, ladder=make_ladder(),
    )
    job = env["ledger"].get_job(job_id)
    assert job.status.value == "approved"
    nodes = env["ledger"].nodes(job_id)
    assert len(nodes) == 1
    assert nodes[0].spec.id == "n1"
    assert nodes[0].spec.goal == "add a helper"
    assert nodes[0].spec.kind == "mechanical"
    assert nodes[0].spec.scope == ["app.py"]
    assert nodes[0].status.value == "pending"
    events = [e for e in env["ledger"].events(job_id) if e.kind == "mullet_opencode_approved"]
    assert len(events) == 1
    assert events[0].data["source"] == "mullet_opencode"


def test_scope_defaults_to_all_when_no_paths(env):
    job_id = create_delegation_job(
        env["ledger"], task="tidy up", paths=None, cwd=str(env["source"]),
        timeout_min=20, ladder=make_ladder(),
    )
    nodes = env["ledger"].nodes(job_id)
    assert nodes[0].spec.scope == ["**"]


def test_repo_defaults_from_a_file_path(env):
    source = env["source"]
    job_id = create_delegation_job(
        env["ledger"], task="touch app", paths=[str(source / "app.py")],
        cwd=None, timeout_min=20, ladder=make_ladder(),
    )
    job = env["ledger"].get_job(job_id)
    assert Path(job.repo_path).resolve() == source.resolve()
    nodes = env["ledger"].nodes(job_id)
    assert nodes[0].spec.scope == ["app.py"]


def test_path_outside_repo_raises(env):
    outside = env["tmp"] / "outside.txt"
    outside.write_text("nope")
    with pytest.raises(ValueError, match="outside the repo"):
        create_delegation_job(
            env["ledger"], task="touch app", paths=[str(outside)],
            cwd=str(env["source"]), timeout_min=20, ladder=make_ladder(),
        )


def test_review_is_on_for_a_risky_task(env):
    job_id = create_delegation_job(
        env["ledger"], task="delete unused code", paths=["app.py"],
        cwd=str(env["source"]), timeout_min=20, ladder=make_ladder(),
    )
    job = env["ledger"].get_job(job_id)
    assert job.budget["review"] is True


def test_review_is_off_for_a_safe_task(env):
    job_id = create_delegation_job(
        env["ledger"], task="add a helper", paths=["app.py"],
        cwd=str(env["source"]), timeout_min=20, ladder=make_ladder(),
    )
    job = env["ledger"].get_job(job_id)
    assert job.budget["review"] is False


def test_restrict_ladder_keeps_only_named_rungs_below_claude():
    config = parse_config({"kimi": KIMI, "openrouter": [QWEN]})
    restricted = restrict_ladder(config, ["kimi"])
    assert restricted.kimi_model == KIMI
    assert restricted.openrouter_models == []
    names = [r.name for r in restricted.rungs()]
    assert names == ["kimi", "claude"]
    assert config.openrouter_models == [QWEN]


def test_restrict_ladder_unknown_provider_raises():
    config = parse_config({"kimi": KIMI})
    with pytest.raises(ValueError, match="unknown providers"):
        restrict_ladder(config, ["kimi", "bad_one"])


def test_without_claude_provider_plan_rungs_offers_no_claude_rung(env):
    source = env["source"]
    allow_opencode(source)
    ladder = make_ladder()
    job_id = create_delegation_job(
        env["ledger"], task="add a helper", paths=["app.py"],
        cwd=str(source), timeout_min=20, ladder=ladder, providers=["kimi"],
    )
    job = env["ledger"].get_job(job_id)
    assert "claude-code" not in job.profile["allowed_executors"]
    assert job.budget["executor"] == "opencode"
    profile = Profile.model_validate(job.profile)
    config = parse_config({"kimi": KIMI, "openrouter": []})
    node = NodeSpec(id="n1", goal="add a helper", kind="mechanical")
    choice = plan_rungs(
        node, profile, ladder.budget(), ladder.machine(config),
        spend_state([], config.max_spend, config), config=config, public=None,
        has_key=ladder.has_key,
    )
    assert [r.name for r in choice.rungs] == ["kimi"]


def test_timeout_is_clamped(env):
    low_id = create_delegation_job(
        env["ledger"], task="a", paths=None, cwd=str(env["source"]),
        timeout_min=0, ladder=make_ladder(),
    )
    high_id = create_delegation_job(
        env["ledger"], task="b", paths=None, cwd=str(env["source"]),
        timeout_min=90, ladder=make_ladder(),
    )
    assert env["ledger"].get_job(low_id).budget["wall_clock_s"] == 60
    assert env["ledger"].get_job(high_id).budget["wall_clock_s"] == 3600


def test_max_spend_comes_from_ladder_config(env):
    ladder = make_ladder({"kimi": KIMI, "max_spend": 7.5})
    job_id = create_delegation_job(
        env["ledger"], task="a", paths=None, cwd=str(env["source"]),
        timeout_min=20, ladder=ladder,
    )
    assert env["ledger"].get_job(job_id).budget["max_spend"] == 7.5


CHECKS = Checks(test_full="python3 runtests.py", test_targeted="python3 runtests.py {tests}", junit=True)
RESULT_KEYS = {"status", "reason", "job_id", "report_path", "rung", "model", "cost", "gates", "files_changed"}

needs_sandbox = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)


def save_checked_profile(source: Path, checks: Checks = CHECKS) -> None:
    save_profile(Profile(
        repo_id=repo_id(source), allowed_executors=["claude-code", "opencode"], checks=checks,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    ))


def good(work, task):
    good_change(work)


def bad(work, task):
    (work / "app.py").write_text((work / "app.py").read_text().replace("return a + b", "return a - b"))


def snapshot(repo: Path) -> dict:
    files = {
        str(p.relative_to(repo)): p.read_bytes()
        for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.relative_to(repo).parts
    }
    return {"files": files, "head": git(repo, "rev-parse", "HEAD"), "status": git(repo, "status", "--porcelain"),
            "count": git(repo, "rev-list", "--count", "HEAD")}


def run_delegate(env, fakes, **kwargs):
    kwargs.setdefault("task", "add sub()")
    kwargs.setdefault("paths", ["app.py", "tests/test_app.py"])
    kwargs.setdefault("timeout_s", 120)
    return delegate(
        env["ledger"], cwd=str(env["source"]), ladder=make_ladder(),
        rung_executors=lambda rung: fakes[rung.name], check=False, **kwargs,
    )


@needs_sandbox
def test_a_passing_change_is_applied_to_the_caller_tree_without_a_commit(env):
    source = env["source"]
    save_checked_profile(source)
    before = snapshot(source)
    fakes = {"kimi": FakeExecutor([good]), "openrouter": FakeExecutor([]), "claude": FakeExecutor([])}
    result = run_delegate(env, fakes)
    assert result["status"] == "applied", result
    assert result["reason"] is None
    assert "def sub(a, b)" in (source / "app.py").read_text()
    assert "test_sub" in (source / "tests" / "test_app.py").read_text()
    assert git(source, "rev-parse", "HEAD") == before["head"]
    assert git(source, "rev-list", "--count", "HEAD") == before["count"]
    assert git(source, "diff", "--cached", "--name-only") == ""
    assert sorted(result["files_changed"]) == ["app.py", "tests/test_app.py"]
    assert result["rung"] == "kimi" and result["model"] == KIMI
    assert any(line.startswith("G5 passed") for line in result["gates"])
    report = Path(result["report_path"])
    assert report.name == "report.md" and report.read_text().startswith(f"# Job {result['job_id']}")


@needs_sandbox
def test_a_failing_change_leaves_the_caller_tree_byte_identical(env):
    source = env["source"]
    save_checked_profile(source)
    before = snapshot(source)
    fakes = {name: FakeExecutor([bad] * 3) for name in ("kimi", "openrouter", "claude")}
    result = run_delegate(env, fakes)
    assert result["status"] in ("blocked", "failed"), result
    assert result["reason"]
    assert snapshot(source) == before
    assert any(line.startswith("G5 failed") for line in result["gates"])


@needs_sandbox
def test_an_unverified_node_leaves_the_caller_tree_alone(env, monkeypatch):
    source = env["source"]
    save_checked_profile(source)
    before = snapshot(source)
    fakes = {"kimi": FakeExecutor([good]), "openrouter": FakeExecutor([]), "claude": FakeExecutor([])}
    real = delegate_mod._outcome

    def as_unverified(ledger, job_id, caller_repo):
        node = ledger.nodes(job_id)[0]
        assert node.status is NodeStatus.PASSED
        ledger._conn.execute("UPDATE nodes SET status = ? WHERE job_id = ? AND node_id = ?",
                             (NodeStatus.UNVERIFIED.value, job_id, node.node_id))
        return real(ledger, job_id, caller_repo)

    monkeypatch.setattr(delegate_mod, "_outcome", as_unverified)
    result = run_delegate(env, fakes)
    assert result["status"] == "unverified", result
    assert snapshot(source) == before
    assert sorted(result["files_changed"]) == ["app.py", "tests/test_app.py"]


@needs_sandbox
def test_a_dirty_target_file_is_not_applied(env):
    source = env["source"]
    save_checked_profile(source)
    (source / "app.py").write_text("def add(a, b):\n    return a + b  # mine\n")
    before = snapshot(source)
    fakes = {"kimi": FakeExecutor([good]), "openrouter": FakeExecutor([]), "claude": FakeExecutor([])}
    result = run_delegate(env, fakes)
    assert result["status"] == "passed_not_applied", result
    assert "app.py" in result["reason"]
    assert snapshot(source) == before


@needs_sandbox
def test_a_timeout_stops_the_job_and_leaves_the_caller_tree_alone(env):
    source = env["source"]
    save_checked_profile(source)
    before = snapshot(source)

    def slow(work, task):
        time.sleep(3)
        good_change(work)

    fakes = {"kimi": FakeExecutor([slow]), "openrouter": FakeExecutor([good]), "claude": FakeExecutor([good])}
    result = run_delegate(env, fakes, timeout_s=0.5, stop_grace_s=60)
    assert result["status"] == "timeout", result
    assert snapshot(source) == before
    job = env["ledger"].get_job(result["job_id"])
    assert job.status in (JobStatus.PAUSED, JobStatus.BLOCKED)


@needs_sandbox
def test_a_timeout_stops_only_its_own_delegations_processes(env):
    save_checked_profile(env["source"])
    other = subprocess.Popen(["sleep", "60"], start_new_session=True)
    mine: list[subprocess.Popen] = []

    def slow(work, task):
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        mine.append(proc)
        with children.track(proc):
            proc.wait()
        good_change(work)

    fakes = {"kimi": FakeExecutor([slow]), "openrouter": FakeExecutor([good]), "claude": FakeExecutor([good])}
    try:
        with children.track(other):  # tracked by this thread's owner, as another job would be
            result = run_delegate(env, fakes, timeout_s=3, stop_grace_s=60)
            assert result["status"] == "timeout", result
            assert mine and mine[0].poll() is not None, "the delegation's own child is stopped"
            assert other.poll() is None, "another owner's child is left running"
    finally:
        for proc in (other, *mine):
            proc.kill()
            proc.wait()


@needs_sandbox
def test_the_providers_hint_limits_the_rungs(env):
    source = env["source"]
    save_checked_profile(source)
    fakes = {"kimi": FakeExecutor([bad] * 3), "openrouter": FakeExecutor([good]), "claude": FakeExecutor([good])}
    result = run_delegate(env, fakes, providers=["kimi"])
    assert fakes["kimi"].calls
    assert fakes["openrouter"].calls == [] and fakes["claude"].calls == []
    assert result["status"] in ("blocked", "failed")
    assert {(a.usage or {}).get("rung") for a in env["ledger"].attempts(result["job_id"])} == {"kimi"}


@needs_sandbox
def test_the_result_has_exactly_the_documented_keys(env):
    save_checked_profile(env["source"])
    fakes = {"kimi": FakeExecutor([good]), "openrouter": FakeExecutor([]), "claude": FakeExecutor([])}
    result = run_delegate(env, fakes)
    assert set(result) == RESULT_KEYS
    assert isinstance(result["status"], str)
    assert result["reason"] is None or isinstance(result["reason"], str)
    assert isinstance(result["job_id"], str) and isinstance(result["report_path"], str)
    assert isinstance(result["rung"], str) and isinstance(result["model"], str)
    assert isinstance(result["cost"], float)
    assert isinstance(result["gates"], list) and all(isinstance(g, str) for g in result["gates"])
    assert isinstance(result["files_changed"], list) and all(isinstance(f, str) for f in result["files_changed"])


def test_a_failed_preflight_returns_blocked_without_running(env, monkeypatch):
    source = env["source"]
    save_checked_profile(source)
    before = snapshot(source)
    monkeypatch.setattr(delegate_mod, "_preflight", lambda ledger, job_id: "Mullet's sandbox is unavailable: test")
    fakes = {"kimi": FakeExecutor([good]), "openrouter": FakeExecutor([]), "claude": FakeExecutor([])}
    result = delegate(env["ledger"], task="add sub()", paths=["app.py"], cwd=str(source), ladder=make_ladder(),
                      rung_executors=lambda rung: fakes[rung.name], check=True, timeout_s=30)
    assert result["status"] == "blocked"
    assert "sandbox" in result["reason"]
    assert set(result) == RESULT_KEYS
    assert fakes["kimi"].calls == []
    assert snapshot(source) == before


def test_stored_profile_is_saved_when_missing(env):
    source = env["source"]
    create_delegation_job(
        env["ledger"], task="a", paths=None, cwd=str(source),
        timeout_min=20, ladder=make_ladder(),
    )
    assert Path(repo_id(source)).name
    assert (env["tmp"] / "profiles" / f"{repo_id(source)}.yaml").exists()
