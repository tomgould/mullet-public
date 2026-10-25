"""`mullet job show --attempt` and `mullet job export` on a job with a failed attempt."""

import shutil

import pytest

from mullet.cli.job_cli import cmd_job
from mullet.jobs import sandbox
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.ledger import Ledger
from mullet.jobs.paths import job_dir
from mullet.jobs.runner import JobRunner
from tests.jobs_helpers import git
from tests.test_jobs_cli import plan, repo, store_profile  # noqa: F401  (repo is a fixture)

pytestmark = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)


def _break_add(work, task):
    (work / "app.py").write_text("def add(a, b):\n    return a - b\n")
    return {"status": "done", "summary": "changed add"}


@pytest.fixture
def failed_job(repo, capsys):  # noqa: F811
    store_profile(repo)
    job_id = plan(repo, capsys, "--max-attempts", "1")
    cmd_job(["approve", job_id])
    with Ledger() as ledger:
        JobRunner(ledger, job_id, FakeExecutor([_break_add])).run()
        attempt = ledger.attempts(job_id)[0]
    capsys.readouterr()
    return job_id, attempt.id


def test_show_attempt_prints_outcome_and_gate_results(failed_job, capsys):
    job_id, attempt = failed_job
    assert cmd_job(["show", job_id, "--attempt", str(attempt)]) == 0
    out = capsys.readouterr().out
    assert "failed" in out and "G3" in out and "G1" in out
    assert "return a - b" not in out


def test_show_attempt_diff_prints_the_work_it_left(failed_job, capsys):
    job_id, attempt = failed_job
    assert cmd_job(["show", job_id, "--attempt", str(attempt), "--diff"]) == 0
    out = capsys.readouterr().out
    assert "-    return a + b" in out and "+    return a - b" in out


def test_show_attempt_diff_strips_escape_sequences_on_a_terminal(repo, capsys):  # noqa: F811
    from mullet.cli.job_cli import _show_attempt

    def planted(work, task):
        (work / "app.py").write_text("def add(a, b):\n    return a - b  # \x1b[2J\x1b]0;pwned\x07\n")
        return {"status": "done", "summary": "changed add"}

    store_profile(repo)
    job_id = plan(repo, capsys, "--max-attempts", "1")
    cmd_job(["approve", job_id])
    with Ledger() as ledger:
        JobRunner(ledger, job_id, FakeExecutor([planted])).run()
        attempt = ledger.attempts(job_id)[0].id
        shown = _show_attempt(ledger, job_id, attempt, True, terminal=True)
        raw = _show_attempt(ledger, job_id, attempt, True, terminal=False)
    assert "+    return a - b  # " in shown and "\x1b" not in shown and "\x07" not in shown
    assert "\x1b[2J" in raw  # piped to a file, the diff stays exact, so it can still be applied


def test_show_attempt_that_is_not_in_the_job_lists_the_ones_that_are(failed_job, capsys):
    job_id, attempt = failed_job
    assert cmd_job(["show", job_id, "--attempt", "9999"]) == 1
    err = capsys.readouterr().err
    assert "no attempt 9999" in err and str(attempt) in err


def test_show_attempt_says_plainly_when_the_work_is_gone(failed_job, capsys):
    job_id, attempt = failed_job
    shutil.rmtree(job_dir(job_id) / "clone")
    assert cmd_job(["show", job_id, "--attempt", str(attempt), "--diff"]) == 0
    out = capsys.readouterr().out
    assert "failed" in out and "not kept" in out


def test_export_creates_a_branch_with_the_attempts_work_and_leaves_the_checkout(failed_job, repo, capsys):
    job_id, attempt = failed_job
    head, current = git(repo, "rev-parse", "HEAD"), git(repo, "branch", "--show-current")
    assert cmd_job(["export", job_id, "--attempt", str(attempt), "--branch", "rescue"]) == 0
    assert "rescue" in capsys.readouterr().out
    assert "return a - b" in git(repo, "show", "rescue:app.py")
    assert git(repo, "rev-parse", "rescue~1") == head
    assert git(repo, "rev-parse", "HEAD") == head and git(repo, "branch", "--show-current") == current
    assert git(repo, "status", "--porcelain") == ""


def test_export_refuses_an_existing_branch_unless_forced(failed_job, repo, capsys):
    job_id, attempt = failed_job
    git(repo, "branch", "taken")
    taken = git(repo, "rev-parse", "taken")
    assert cmd_job(["export", job_id, "--attempt", str(attempt), "--branch", "taken"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert git(repo, "rev-parse", "taken") == taken
    assert cmd_job(["export", job_id, "--attempt", str(attempt), "--branch", "taken", "--force"]) == 0
    assert "return a - b" in git(repo, "show", "taken:app.py")


def test_export_never_overwrites_the_checked_out_branch(failed_job, repo, capsys):
    job_id, attempt = failed_job
    head = git(repo, "rev-parse", "HEAD")
    current = git(repo, "branch", "--show-current")
    assert cmd_job(["export", job_id, "--attempt", str(attempt), "--branch", current, "--force"]) == 1
    assert "checked out" in capsys.readouterr().err
    assert git(repo, "rev-parse", "HEAD") == head


def test_export_refuses_a_bad_branch_name(failed_job, capsys):
    job_id, attempt = failed_job
    assert cmd_job(["export", job_id, "--attempt", str(attempt), "--branch", "bad..name"]) == 1
    assert "not a valid plain branch name" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["refs/heads/rescue", "heads/rescue", "refs/rescue", "@{-1}", "-rescue",
                                  "HEAD", "rescue.lock", ""])
def test_export_refuses_a_ref_path_or_anything_but_a_plain_branch_name(failed_job, repo, capsys, name):
    job_id, attempt = failed_job
    before = git(repo, "for-each-ref", "--format=%(refname)")
    assert cmd_job(["export", job_id, "--attempt", str(attempt), f"--branch={name}"]) == 1
    err = capsys.readouterr().err
    assert "plain branch name" in err
    assert git(repo, "for-each-ref", "--format=%(refname)") == before


def test_export_takes_a_nested_branch_name(failed_job, repo, capsys):
    job_id, attempt = failed_job
    assert cmd_job(["export", job_id, "--attempt", str(attempt), "--branch", "rescue/attempt-1"]) == 0
    assert "return a - b" in git(repo, "show", "rescue/attempt-1:app.py")


def test_export_says_plainly_when_the_work_is_gone(failed_job, repo, capsys):
    job_id, attempt = failed_job
    shutil.rmtree(job_dir(job_id) / "clone")
    assert cmd_job(["export", job_id, "--attempt", str(attempt), "--branch", "rescue"]) == 1
    assert "not kept" in capsys.readouterr().err
