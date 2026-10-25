"""Tests for what `mullet job plan` and `mullet job show` put in front of the approver,
and for how the CLI lets the planner be asked again in the same session."""

from types import SimpleNamespace

import pytest
import yaml

from mullet.cli import job_cli
from mullet.cli.job_cli import cmd_job
from mullet.jobs.executors.base import ExecResult
from mullet.jobs.ledger import Ledger

from tests.jobs_helpers import make_source_repo


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MULLET_PROFILES_DIR", str(tmp_path / "profiles"))
    return make_source_repo(tmp_path / "src")


def _job_id(out: str) -> str:
    return next(line.split()[1] for line in out.splitlines() if line.startswith("Job "))


PLAN = {"notes": "two steps", "open_questions": ["Is the v1 endpoint still called by anyone?"], "nodes": [
    {"id": "model", "parent": None, "goal": "add the model", "kind": "design", "scope": ["app.py"]},
    {"id": "rename-api", "parent": None, "goal": "rename the public endpoint", "kind": "design",
     "scope": ["app.py", "tests/test_app.py"], "after": ["model"], "door": "one-way"},
]}


def fake_planner(monkeypatch, *results):
    calls = []

    def ask(prompt, cwd, schema, budget, model=None, label="ask", **kwargs):
        calls.append(kwargs)
        return results[min(len(calls), len(results)) - 1]

    monkeypatch.setattr(job_cli, "claude_executor", lambda job, profile, logs: SimpleNamespace(ask=ask))
    monkeypatch.setattr(job_cli.shutil, "which", lambda name: "/usr/bin/" + name)
    return calls


def test_one_way_steps_and_open_questions_are_shown_under_the_tree(repo, capsys, monkeypatch):
    fake_planner(monkeypatch, ExecResult(exit_code=0, claim=PLAN))
    assert cmd_job(["plan", "rename the api", "--repo", str(repo)]) == 0
    out = capsys.readouterr().out
    tree_end = out.index("Checks Mullet will run")
    shown = out[out.index("Plan:"):tree_end]
    assert "One-way steps (hard to walk back):\n  rename-api: rename the public endpoint" in shown
    assert "Open questions from the planner:\n  - Is the v1 endpoint still called by anyone?" in shown
    assert "after: model" in shown
    job_id = _job_id(out)
    with Ledger() as ledger:
        events = [e.data for e in ledger.events(job_id) if e.kind == "plan_notes"]
    assert events == [{"notes": "two steps", "open_questions": PLAN["open_questions"]}]
    assert cmd_job(["show", job_id]) == 0
    assert "Open questions from the planner:" in capsys.readouterr().out


def test_a_plan_without_doors_or_questions_shows_neither_heading(repo, capsys, monkeypatch):
    claim = {"notes": "", "nodes": [{"id": "n1", "parent": None, "goal": "do it", "kind": "design",
                                     "scope": ["app.py"]}]}
    fake_planner(monkeypatch, ExecResult(exit_code=0, claim=claim))
    assert cmd_job(["plan", "goal", "--repo", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "One-way steps" not in out and "Open questions" not in out and "after:" not in out
    with Ledger() as ledger:
        assert [e for e in ledger.events(_job_id(out)) if e.kind == "plan_notes"] == []


def test_after_door_and_open_questions_round_trip_through_yaml(repo, capsys, monkeypatch, tmp_path):
    fake_planner(monkeypatch, ExecResult(exit_code=0, claim=PLAN))
    assert cmd_job(["plan", "rename the api", "--repo", str(repo)]) == 0
    job_id = _job_id(capsys.readouterr().out)
    assert cmd_job(["show", job_id, "--yaml"]) == 0
    dumped = yaml.safe_load(capsys.readouterr().out)
    assert dumped["open_questions"] == PLAN["open_questions"]
    assert dumped["nodes"][1]["after"] == ["model"] and dumped["nodes"][1]["door"] == "one-way"
    assert "after" not in dumped["nodes"][0] and "door" not in dumped["nodes"][0]
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump(dumped))
    assert cmd_job(["plan", "rename the api", "--repo", str(repo), "--plan-file", str(plan_file)]) == 0
    out = capsys.readouterr().out
    second = _job_id(out)
    assert "Open questions from the planner:" in out
    with Ledger() as ledger:
        assert [n.spec for n in ledger.nodes(second)] == [n.spec for n in ledger.nodes(job_id)]


def test_a_plan_file_with_a_bad_after_is_refused(repo, capsys, tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump({"nodes": [
        {"id": "a", "goal": "x", "scope": ["app.py"], "after": ["b"]},
        {"id": "b", "goal": "y", "scope": ["app.py"]}]}))
    assert cmd_job(["plan", "goal", "--repo", str(repo), "--plan-file", str(plan_file)]) == 1
    assert "list order" in capsys.readouterr().err


def test_the_planner_session_is_kept_and_resumed_once(repo, capsys, monkeypatch):
    calls = fake_planner(monkeypatch, ExecResult(exit_code=0, session_id="s1"),
                         ExecResult(exit_code=0, claim=PLAN, session_id="s1"))
    assert cmd_job(["plan", "rename the api", "--repo", str(repo)]) == 0
    assert calls == [{"persist": True}, {"persist": True, "resume": "s1"}]


def test_reask_attempts_zero_keeps_no_session(repo, capsys, monkeypatch, tmp_path):
    from mullet.jobs.settings import JobSettings

    monkeypatch.setattr(job_cli, "load_job_settings", lambda: JobSettings(reask_attempts=0))
    calls = fake_planner(monkeypatch, ExecResult(exit_code=0, session_id="s1"))
    assert cmd_job(["plan", "rename the api", "--repo", str(repo)]) == 1
    assert calls == [{}]
    assert "no structured answer" in capsys.readouterr().err


def test_ask_can_keep_and_resume_a_read_only_session(tmp_path, monkeypatch):
    from mullet.jobs.executors.base import Budget
    from mullet.jobs.executors.claude_code import ClaudeCodeExecutor

    executor = ClaudeCodeExecutor(tmp_path / "logs")
    seen = {}
    monkeypatch.setattr(executor, "_launch", lambda *a, **kw: seen.update(kw) or ExecResult(exit_code=0))
    executor.ask("p", tmp_path, {}, Budget())
    assert (seen["persist"], seen["resume"]) == (False, None)
    executor.ask("p", tmp_path, {}, Budget(), persist=True, resume="s1")
    assert (seen["persist"], seen["resume"]) == (True, "s1")
    executor.binary = "claude"
    argv = executor.command(tmp_path / "s.json", Budget(), persist=True, resume="s1")
    assert "--no-session-persistence" not in argv and argv[argv.index("--resume") + 1] == "s1"
