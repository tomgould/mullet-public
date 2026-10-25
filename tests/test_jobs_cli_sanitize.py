"""Model text in the approval view, the ladder report and plan questions reaches the terminal sanitised."""

from mullet.cli.job_cli import show
from mullet.jobs.ladder_report import _short
from mullet.jobs.ledger import Ledger
from mullet.jobs.planner import plan_questions
from mullet.jobs.profile import Profile
from mullet.jobs.types import NodeSpec

FAKE_ONE_WAY = "\n\nOne-way steps (hard to walk back):\n  n9: nothing to see"
FAKE_CHECKS = "\n\nChecks Mullet will run (from the repo profile):\n  none"


def test_the_approval_view_cannot_be_spoofed_by_planner_text(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    with Ledger(tmp_path / "jobs" / "ledger.db") as ledger:
        job_id = ledger.create_job(str(tmp_path), "improve the app", "main", "a" * 40,
                                   Profile(repo_id="x").model_dump(), budget={})
        spec = NodeSpec(id="n1", goal="tidy\x1b[2J\x1b[H" + FAKE_ONE_WAY, scope=["app.py\x1b]0;pwned\x07"],
                        acceptance=["make test" + FAKE_CHECKS], after=[], door="one-way")
        ledger.set_plan(job_id, [(spec, None)])
        ledger.add_event(job_id, "plan_notes", {"open_questions": ["why?\x1b[31m" + FAKE_CHECKS]})
        out = show(ledger, job_id)
    assert "\x1b" not in out and "\x07" not in out
    lines = out.splitlines()
    assert lines.count("One-way steps (hard to walk back):") == 1  # the real one, n1's
    assert lines.count("Checks Mullet will run (from the repo profile):") == 1
    assert lines.count("Open questions from the planner:") == 1
    assert not any(line.startswith("  n9:") for line in lines)
    goal_lines = [line for line in lines if "tidy" in line]
    assert len(goal_lines) == 2 and all("One-way steps (hard to walk back): n9" in line for line in goal_lines)


def test_plan_questions_are_one_clean_line_each():
    raw = {"open_questions": ["Is the\nAPI public?\x1b[2J", "  \x1b[0m  ", "ok"]}
    assert plan_questions(raw) == ["Is the API public?", "ok"]


def test_the_ladder_reports_reasons_are_one_clean_line():
    assert _short("tests failed\x1b[31m\n\nRung x: 0 attempts\x07") == "tests failed Rung x: 0 attempts"
