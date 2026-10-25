"""Tests for mullet.jobs.gates.

The integration tests run real gates in the sandbox against a tiny fixture
repo whose test runner writes junit XML. Each hostile case edits the clone
the way a careless or dishonest executor might, then checks a gate catches it.
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from mullet.jobs import gates, sandbox
from mullet.jobs.clone import ChangedFile, JobClone, Snapshot, resolve_commit
from mullet.jobs.executors.base import Budget, Task
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.gates import (
    Baseline, CheckRun, GateContext, GateRunner, added_lines_between, is_test_file, is_test_path,
    parse_junit, regressions, run_gates, scan_gitleaks, scan_secrets, select_tests, symlink_escapes,
)
from mullet.jobs.profile import Checks, Profile
from mullet.jobs.types import AttemptOutcome, GateStatus, NodeSpec, judge

from tests.jobs_helpers import BASE_FILES, FAKE_KEY, git, good_change, write

needs_bwrap = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)

@dataclass
class Env:
    source: Path
    clone: JobClone
    runner: GateRunner
    baseline: Baseline
    base_sha: str


def make_env(tmp_path, monkeypatch, files=None, untracked=None, **profile_kwargs) -> Env:
    jobs_home = tmp_path / "jobs"
    jobs_home.mkdir()
    monkeypatch.setenv("MULLET_JOBS_DIR", str(jobs_home))
    source = tmp_path / "src"
    source.mkdir()
    git(source, "init", "-q", "-b", "main")
    write(source, files or BASE_FILES)
    git(source, "add", ".")
    git(source, "commit", "-qm", "init")
    write(source, untracked or {})  # local files such as a test .env, never committed
    base = resolve_commit(source, "HEAD")
    clone = JobClone.create(source, base, jobs_home / "job", "mullet/job", identity=("Dev", "d@e"))
    checks = profile_kwargs.pop("checks", Checks(
        lint="python3 -c \"import ast; ast.parse(open('app.py').read())\"",
        test_full="python3 runtests.py",
        test_targeted="python3 runtests.py {tests}",
        junit=True,
    ))
    profile = Profile(repo_id="fixture", checks=checks, env={"PYTHONDONTWRITEBYTECODE": "1"}, **profile_kwargs)
    runner = GateRunner(clone, profile, tmp_path / "logs")
    baseline = runner.baseline(base, source)
    return Env(source, clone, runner, baseline, base)


def attempt(env: Env, mutate, node: NodeSpec | None = None, full_tests: bool = False):
    node = node or NodeSpec(id="n1", goal="change things")
    mutate(env.clone.work)
    snap = env.clone.snapshot()
    env.clone.materialize(snap.tree, keep=env.runner.keep)
    results = run_gates(GateContext(env.runner, node, snap, env.baseline, full_tests, env.source))
    env.clone.reset_to(env.base_sha, keep=env.runner.keep)
    return {r.gate: r for r in results}, judge(results)


# ---- unit tests, no sandbox needed ----------------------------------------------------------

def test_check_commands_never_get_unsafe_profile_env(tmp_path):
    from types import SimpleNamespace

    profile = Profile(repo_id="x", env={"FOO": "1", "LD_PRELOAD": "/x.so", "HTTPS_PROXY": "http://evil:1",
                                        "NODE_OPTIONS": "--require x", "PIP_CACHE_DIR": "/evil"})
    clone = SimpleNamespace(work=tmp_path / "work", git_dir=tmp_path / "git", root=tmp_path)
    env = gates.GateRunner(clone, profile, tmp_path / "logs", cache_dir=tmp_path / "cache").spec(tmp_path).env
    assert env["FOO"] == "1"
    assert not {"LD_PRELOAD", "HTTPS_PROXY", "NODE_OPTIONS"} & set(env)
    assert env["PIP_CACHE_DIR"] == str(tmp_path / "cache" / "pip")  # Mullet's job caches win


def test_parse_junit(tmp_path):
    report = tmp_path / "j.xml"
    report.write_text(
        '<testsuites><testsuite>'
        '<testcase classname="a" name="ok"/>'
        '<testcase classname="a" name="bad"><failure message="x"/></testcase>'
        '<testcase classname="b" name="err"><error/></testcase>'
        '<testcase classname="b" name="skipped"><skipped/></testcase>'
        '</testsuite></testsuites>'
    )
    parsed = parse_junit(report)
    assert parsed.passed == {"a::ok"}
    assert parsed.failed == {"a::bad", "b::err"}
    assert parsed.skipped == {"b::skipped"}
    (tmp_path / "broken.xml").write_text("<testsuite>")
    assert parse_junit(tmp_path / "broken.xml") is None
    assert parse_junit(tmp_path / "missing.xml") is None


def test_added_lines_between_ignores_diff_syntax_in_content():
    old = "let counter = 0;\nfunction f() {}\n"
    new = "let counter = 0;\n++ counter;\nconst key = 'x';\nfunction f() {}\n"
    assert added_lines_between(old, new) == [(2, "++ counter;"), (3, "const key = 'x';")]
    assert added_lines_between("", "a\nb\n") == [(1, "a"), (2, "b")]


def test_scan_secrets_reports_location_not_value():
    lines = [
        ("config.py", 1, "AWS_KEY = 'AKIAQ7XJ3MZW4PLUTVBN'"),
        ("key.pem", 1, "-----BEGIN OPENSSH PRIVATE KEY-----"),
        ("settings.py", 3, 'api_key = "c8f3a91b2d7e4f60a5b9c3d2e1f0a7b6"'),
        ("app.py", 2, "total = price * quantity"),
        ("tests/test_x.py", 4, 'password = "hunter2"'),
    ]
    findings = scan_secrets(lines)
    assert [(f["file"], f["line"]) for f in findings] == [("config.py", 1), ("key.pem", 1), ("settings.py", 3)]
    assert "AKIAQ7XJ3MZW4PLUTVBN" not in repr(findings)


@pytest.mark.skipif(not gates.shutil.which("gitleaks"), reason="gitleaks not installed")
def test_gitleaks_scans_added_lines_and_ignores_planted_config():
    token = "ghp_" + "aB3dE5fG7hJ9kL1mN2pQ4rS6tU8vW0xY2zA4"  # gitleaks wants real-looking entropy
    lines = [(".gitleaks.toml", 1, "[allowlist]"), ("src/conf.py", 7, f"TOKEN = '{token}'")]
    findings, error = scan_gitleaks(lines)
    assert error is None
    assert any(f["file"] == "src/conf.py" and f["line"] == 7 for f in findings)
    assert token not in repr(findings)


@pytest.mark.parametrize("path, test_file, test_path", [
    ("tests/test_app.py", True, True), ("test_app.py", True, True), ("app_test.go", True, True),
    ("src/button.test.tsx", True, True), ("tests/UserTest.php", True, True),
    ("spec/models/user_spec.rb", True, True), ("tests/fixtures/data.json", False, True),
    ("tests/conftest.py", False, True), ("src/app.py", False, False), ("contest.py", False, False),
])
def test_test_file_detection(path, test_file, test_path):
    assert is_test_file(path) is test_file
    assert is_test_path(path) is test_path


def test_select_tests_maps_sources_to_tests_and_skips_helpers():
    snap = Snapshot(tree="t", base="b", files=[
        ChangedFile("pkg/app.py", "M", "100644", "100644"),
        ChangedFile("tests/test_other.py", "M", "100644", "100644"),
        ChangedFile("tests/fixtures/data.json", "M", "100644", "100644"),
        ChangedFile("tests/test_removed.py", "D", "100644", "000000"),
    ])
    paths = ["pkg/app.py", "tests/test_app.py", "tests/test_other.py", "tests/test_unrelated.py",
             "tests/fixtures/data.json"]
    assert select_tests(snap, paths) == ["tests/test_app.py", "tests/test_other.py"]


def test_symlink_escape_detection_follows_chains():
    assert symlink_escapes("a/link", {"a/link": "/etc/passwd"})
    assert symlink_escapes("a/link", {"a/link": "../../outside"})
    assert not symlink_escapes("a/link", {"a/link": "../inside.txt"})
    assert not symlink_escapes("link", {"link": "a/b"})
    chain = {"x": ".", "y": "x/.."}
    assert not symlink_escapes("x", chain)
    assert symlink_escapes("y", chain)
    assert symlink_escapes("loop", {"loop": "loop2", "loop2": "loop"})


def _run(passed, failed=(), skipped=(), code=0, timed_out=False, junit=True):
    return CheckRun("test_full", "x", None if timed_out else code, timed_out, 1.0, "/l",
                    failing_tests=set(failed) if junit else None,
                    passing_tests=set(passed) if junit else None,
                    skipped_tests=set(skipped) if junit else None)


def test_regressions_catch_tests_switched_off():
    base = _run({"t::a", "t::b"})
    assert regressions(_run({"t::a", "t::b"}), base, full=True) is None
    assert regressions(_run({"t::a"}, skipped={"t::b"}), base, full=False) == {"t::b"}
    assert regressions(_run({"t::a"}), base, full=True) == {"t::b"}
    assert regressions(_run({"t::a"}), base, full=False) is None  # b wasn't selected
    assert regressions(_run(set()), base, full=False) == "no tests ran"
    assert regressions(_run(set(), junit=False), base, full=True) == "the test run produced no junit report"
    assert regressions(_run({"t::a", "t::b"}, code=2), base, full=True).startswith("the tests exited 2")
    assert regressions(_run({"t::a", "t::b"}, timed_out=True), base, full=True) == "the tests timed out"


def test_regressions_ignore_old_failures_and_catch_new_ones():
    base = _run({"t::a"}, failed={"t::old"}, code=1)
    assert regressions(_run({"t::a"}, failed={"t::old"}, code=1), base, full=True) is None
    assert regressions(_run({"t::a", "t::new"}, failed={"t::old", "t::new2"}, code=1), base, full=True) == {"t::new2"}


def test_regressions_without_junit_use_exit_codes():
    base = _run(set(), junit=False)
    assert regressions(_run(set(), junit=False), base, full=True) is None
    assert regressions(_run(set(), code=1, junit=False), base, full=True) == "the tests failed (exit 1)"


def test_check_run_round_trip():
    run = CheckRun("test_full", "x", 1, False, 1.5, "/l", {"a::b"}, {"a::c"}, set(), tail="tail")
    back = CheckRun.from_dict(run.to_dict())
    assert back.failing_tests == {"a::b"} and back.passing_tests == {"a::c"} and back.tail == ""
    base = Baseline("sha", {"test_full": run})
    assert Baseline.from_dict(base.to_dict()).runs["test_full"].exit_code == 1


# ---- integration: real gates in the sandbox ---------------------------------------------------

@needs_bwrap
def test_baseline_records_each_check(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    assert set(env.baseline.runs) == {"lint", "test_full"}
    run = env.baseline.runs["test_full"]
    assert run.ok and run.failing_tests == set()
    assert run.passing_tests == {"test_app::test_add", "test_app::test_add_zero"}


@needs_bwrap
def test_good_change_passes_every_gate(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    results, outcome = attempt(env, good_change)
    assert {g: r.status for g, r in results.items()} == {g: GateStatus.PASS for g in ("G1", "G2", "G3", "G4", "G5", "G6")}
    assert outcome is AttemptOutcome.PASSED
    assert results["G5"].details["mode"] == "test_targeted"
    assert "./tests/test_app.py" in results["G5"].details["runs"][0]["command"]
    assert results["G1"].details["tree"]


@needs_bwrap
def test_claim_of_success_with_no_change_fails(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    executor = FakeExecutor([lambda work, task: {"status": "done", "summary": "implemented and all tests pass"}])
    result = executor.run(Task("goal", NodeSpec(id="n1", goal="add sub")), env.clone.work, Budget())
    assert result.claim["status"] == "done"
    results, outcome = attempt(env, lambda work: None)
    assert results["G1"].status is GateStatus.FAIL
    assert all(results[g].status is GateStatus.SKIPPED for g in ("G2", "G3", "G6", "G4", "G5"))
    assert outcome is AttemptOutcome.FAILED


@needs_bwrap
def test_breaking_a_test_fails_g5_with_the_test_id(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    results, outcome = attempt(env, lambda w: (w / "app.py").write_text("def add(a, b):\n    return a - b\n"))
    assert results["G5"].status is GateStatus.FAIL
    assert "test_app::test_add" in results["G5"].details["regressions"]
    assert len(results["G5"].details["runs"]) == 1 + gates.FLAKY_RERUNS
    assert outcome is AttemptOutcome.FAILED


@needs_bwrap
def test_skipping_a_passing_test_from_app_code_fails_g5(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)

    def skip_from_app(work):
        (work / "app.py").write_text(
            "import unittest\n\ndef add(a, b):\n    raise unittest.SkipTest('later')\n")

    results, _ = attempt(env, skip_from_app)
    assert results["G5"].status is GateStatus.FAIL
    assert "test_app::test_add" in results["G5"].details["regressions"]


@pytest.mark.parametrize("runner_script, reason", [
    ("import os\nopen(os.environ['MULLET_JUNIT'], 'w').write('<testsuite/>')\n", "no tests ran"),
    ("import sys\nsys.exit(0)\n", "no junit report"),
])
@needs_bwrap
def test_rigging_the_test_runner_fails_g5(tmp_path, monkeypatch, runner_script, reason):
    env = make_env(tmp_path, monkeypatch)

    def rig(work):
        good_change(work)
        (work / "runtests.py").write_text(runner_script)

    results, _ = attempt(env, rig)
    assert results["G5"].status is GateStatus.FAIL
    assert reason in results["G5"].summary


@needs_bwrap
def test_deleting_an_assertion_fails_g6(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)

    def cheat(work):
        (work / "app.py").write_text("def add(a, b):\n    return a - b\n")
        (work / "tests" / "test_app.py").write_text(
            "from app import add\n\ndef test_add():\n    add(2, 3)\n\ndef test_add_zero():\n    assert add(0, 0) == 0\n")

    results, _ = attempt(env, cheat)
    assert results["G6"].status is GateStatus.FAIL
    assert "assertions" in results["G6"].summary


@needs_bwrap
def test_non_utf8_test_file_is_still_checked(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)

    def cheat(work):
        (work / "tests" / "test_app.py").write_bytes(
            b"# caf\xe9\nfrom app import add\n\ndef test_add():\n    add(2, 3)\n\ndef test_add_zero():\n    add(0, 0)\n")

    results, _ = attempt(env, cheat)
    assert results["G6"].status is GateStatus.FAIL


@needs_bwrap
def test_adding_a_skip_fails_g6(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)

    def skip(work):
        path = work / "tests" / "test_app.py"
        path.write_text("import pytest\n" + path.read_text().replace("def test_add():", "@pytest.mark.skip\ndef test_add():"))

    results, _ = attempt(env, skip)
    assert results["G6"].status is GateStatus.FAIL
    assert "skip" in results["G6"].summary


@needs_bwrap
def test_deleting_a_test_file_fails_g6(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    results, _ = attempt(env, lambda w: (w / "tests" / "test_app.py").unlink())
    assert results["G6"].status is GateStatus.FAIL


@needs_bwrap
def test_test_weakening_allowed_when_the_node_says_so(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    node = NodeSpec(id="n1", goal="drop a redundant test", allow_test_weakening=True)

    def drop(work):
        (work / "tests" / "test_app.py").write_text("from app import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")

    results, outcome = attempt(env, drop, node, full_tests=False)
    assert results["G6"].status is GateStatus.PASS
    assert outcome is AttemptOutcome.PASSED


@needs_bwrap
def test_tests_an_earlier_step_removed_are_not_held_against_a_later_one(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    # An earlier step, allowed to, removed test_add_zero and was committed.
    (env.clone.work / "tests" / "test_app.py").write_text(
        "from app import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    head = env.clone.commit_snapshot(env.clone.snapshot(), "drop a test")

    def g5(retired: bool, mutate):
        mutate(env.clone.work)
        snap = env.clone.snapshot(head)
        env.clone.materialize(snap.tree, keep=env.runner.keep)
        ctx = GateContext(env.runner, NodeSpec(id="n2", goal="docs"), snap, env.baseline, True, env.source,
                          tests_retired=retired)
        results = {r.gate: r for r in run_gates(ctx)}
        env.clone.reset_to(head, keep=env.runner.keep)
        return results["G5"]

    def docs(work):
        (work / "NOTES.md").write_text("notes\n")

    def touch_tests(work):
        docs(work)
        (work / "tests" / "test_more.py").write_text("def test_more():\n    assert True\n")

    assert g5(False, docs).status is GateStatus.FAIL
    assert g5(True, docs).status is GateStatus.PASS
    # A step that changes tests itself is still held to every test on base.
    assert g5(True, touch_tests).status is GateStatus.FAIL


@needs_bwrap
@pytest.mark.parametrize("path", ["uv.lock", ".github/workflows/ci.yml", ".mcp.json", "docs/CLAUDE.md", ".env",
                                  ".gitattributes", "pytest.ini"])
def test_protected_paths_fail_g2(tmp_path, monkeypatch, path):
    env = make_env(tmp_path, monkeypatch)
    results, _ = attempt(env, lambda w: write(w, {path: "changed\n"}))
    assert results["G2"].status is GateStatus.FAIL
    assert "protected" in repr(results["G2"].details["problems"])


@needs_bwrap
def test_protected_path_allowed_by_the_node(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    node = NodeSpec(id="n1", goal="bump deps", allow_protected=["uv.lock"])
    results, _ = attempt(env, lambda w: (w / "uv.lock").write_text("lock v2\n"), node)
    assert results["G2"].status is GateStatus.PASS


@needs_bwrap
def test_gitattributes_cannot_hide_a_secret(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    node = NodeSpec(id="n1", goal="tune attributes", allow_protected=[".gitattributes"])

    def hide(work):
        (work / ".gitattributes").write_text("* -diff\n")
        (work / "config.py").write_text(f"KEY = '{FAKE_KEY}'\n")

    results, _ = attempt(env, hide, node)
    assert results["G3"].status is GateStatus.FAIL
    assert FAKE_KEY not in repr(results)


@needs_bwrap
def test_nul_bytes_cannot_hide_a_secret(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    results, _ = attempt(env, lambda w: (w / "blob.dat").write_bytes(b"\0\0header\nKEY=" + FAKE_KEY.encode() + b"\n"))
    assert results["G3"].status is GateStatus.FAIL


@needs_bwrap
def test_diff_syntax_in_content_cannot_blind_the_scanner(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    results, _ = attempt(env, lambda w: (w / "app.js").write_text(f"let c = 0;\n++ c;\nconst k = '{FAKE_KEY}';\n"))
    assert results["G3"].status is GateStatus.FAIL


@needs_bwrap
def test_out_of_scope_and_oversized_changes_fail_g2(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    node = NodeSpec(id="n1", goal="touch app only", scope=["app.py", "tests/**"], max_lines=5)

    def sprawl(work):
        (work / "app.py").write_text("def add(a, b):\n    return a + b\n" + "x = 1\n" * 10)
        (work / "other.py").write_text("y = 2\n")

    results, _ = attempt(env, sprawl, node)
    problems = results["G2"].details["problems"]
    assert any("other.py: outside" in p for p in problems)
    assert any("lines changed" in p for p in problems)


@needs_bwrap
def test_huge_file_fails_g2(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    node = NodeSpec(id="n1", goal="add data", max_lines=10_000_000)
    results, _ = attempt(env, lambda w: (w / "data.bin").write_bytes(b"x" * (gates.MAX_FILE_BYTES + 1)), node)
    assert results["G2"].status is GateStatus.FAIL
    assert "bytes" in results["G2"].summary


@needs_bwrap
def test_escaping_symlinks_and_nested_repo_fail_g2(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)

    def plant(work):
        (work / "creds").symlink_to(Path.home() / ".ssh" / "id_ed25519")
        (work / "x").symlink_to(".")
        (work / "y").symlink_to("x/..")
        nested = work / "sub"
        nested.mkdir()
        git(nested, "init", "-q")
        (nested / "f").write_text("f")
        git(nested, "add", ".")
        git(nested, "commit", "-qm", "n")

    results, _ = attempt(env, plant)
    problems = repr(results["G2"].details["problems"])
    assert "creds: symlink points outside" in problems
    assert "y: symlink points outside" in problems
    assert "x: symlink" not in problems
    assert "sub: adds a nested repository" in problems


@needs_bwrap
def test_secret_in_diff_fails_g3_without_echoing_it(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    results, _ = attempt(env, lambda w: (w / "config.py").write_text(f"OPENAI = '{FAKE_KEY}'\n"))
    assert results["G3"].status is GateStatus.FAIL
    assert FAKE_KEY not in repr(results["G3"])


@needs_bwrap
def test_risk_flags_only_block_mechanical_nodes(tmp_path, monkeypatch):
    files = dict(BASE_FILES)
    files["app.py"] = 'GREETING = "hello world"\n\n\ndef add(a, b):\n    return a + b\n'
    env = make_env(tmp_path, monkeypatch, files=files)

    def change_literal(work):
        (work / "app.py").write_text('GREETING = "hello there"\n\n\ndef add(a, b):\n    return a + b\n')

    results, _ = attempt(env, change_literal, NodeSpec(id="n1", goal="tidy"))
    assert results["G3"].status is GateStatus.PASS
    results, _ = attempt(env, change_literal, NodeSpec(id="n1", goal="rename", kind="mechanical"))
    assert results["G3"].status is GateStatus.FAIL


@needs_bwrap
def test_risk_flags_ignore_prose_files(tmp_path, monkeypatch):
    files = dict(BASE_FILES)
    files["README.md"] = 'Run `add(1, 2)` to get "three".\n'
    env = make_env(tmp_path, monkeypatch, files=files)

    def reword(work):
        (work / "README.md").write_text('Run `add(2, 3)` to get "five".\n')

    results, _ = attempt(env, reword, NodeSpec(id="n1", goal="docs", kind="mechanical"))
    assert results["G3"].status is GateStatus.PASS
    assert results["G3"].details["risk_flags"] == []


@needs_bwrap
def test_files_hidden_by_gitignore_are_gone_before_checks_run(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    node = NodeSpec(id="n1", goal="change", acceptance=["test ! -e planted.py"])

    def plant(work):
        good_change(work)
        (work / "planted.py").write_text("print('cheat')\n")
        with open(work / ".gitignore", "a") as fh:
            fh.write("planted.py\n")

    results, outcome = attempt(env, plant, node)
    assert results["G5"].status is GateStatus.PASS
    assert outcome is AttemptOutcome.PASSED


@needs_bwrap
def test_a_check_that_scribbles_on_the_tree_cannot_affect_the_next(tmp_path, monkeypatch):
    checks = Checks(
        lint="echo 'raise SystemExit(3)' >> app.py",
        test_full="python3 runtests.py", test_targeted="python3 runtests.py {tests}", junit=True,
    )
    env = make_env(tmp_path, monkeypatch, checks=checks)
    assert env.baseline.runs["test_full"].ok
    results, outcome = attempt(env, good_change)
    assert results["G5"].status is GateStatus.PASS
    assert outcome is AttemptOutcome.PASSED


@needs_bwrap
def test_breaking_lint_fails_g4(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    results, _ = attempt(env, lambda w: (w / "app.py").write_text("def add(a, b)\n    return a + b\n"))
    assert results["G4"].status is GateStatus.FAIL
    assert results["G5"].status is GateStatus.SKIPPED


@needs_bwrap
def test_failing_acceptance_command_fails_g5(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    node = NodeSpec(id="n1", goal="add sub", acceptance=["python3 -c \"from app import sub\""])
    results, _ = attempt(env, lambda w: (w / "README.md").write_text("docs\n"), node)
    assert results["G5"].status is GateStatus.FAIL
    assert "acceptance" in results["G5"].summary


@needs_bwrap
def test_flaky_test_passes_after_a_rerun(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)

    def flaky(work):
        (work / "tests" / "test_flaky.py").write_text(textwrap.dedent("""\
            from pathlib import Path

            def test_sometimes():
                marker = Path("flaky.marker")
                if not marker.exists():
                    marker.write_text("1")
                    assert False, "first run fails"
                assert True
            """))

    results, outcome = attempt(env, flaky)
    assert results["G5"].status is GateStatus.PASS
    assert "flaky" in results["G5"].summary
    assert outcome is AttemptOutcome.PASSED


@needs_bwrap
def test_existing_failure_on_base_is_not_blamed_on_the_node(tmp_path, monkeypatch):
    files = dict(BASE_FILES)
    files["tests/test_broken.py"] = "def test_already_broken():\n    assert False\n"
    env = make_env(tmp_path, monkeypatch, files=files)
    assert env.baseline.runs["test_full"].failing_tests == {"test_broken::test_already_broken"}
    results, outcome = attempt(env, good_change, full_tests=True)
    assert results["G5"].status is GateStatus.PASS
    assert outcome is AttemptOutcome.PASSED


@needs_bwrap
def test_red_baseline_without_junit_makes_the_node_unverified(tmp_path, monkeypatch):
    files = dict(BASE_FILES)
    files["tests/test_broken.py"] = "def test_already_broken():\n    assert False\n"
    env = make_env(tmp_path, monkeypatch, files=files, checks=Checks(test_full="python3 runtests.py"))
    results, outcome = attempt(env, good_change)
    assert results["G5"].status is GateStatus.NOT_RUN
    assert outcome is AttemptOutcome.UNVERIFIED


@needs_bwrap
def test_no_tests_configured_is_unverified(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch, checks=Checks())
    results, outcome = attempt(env, good_change)
    assert results["G4"].status is GateStatus.PASS
    assert results["G5"].status is GateStatus.NOT_RUN
    assert outcome is AttemptOutcome.UNVERIFIED


@needs_bwrap
def test_analysis_node_must_not_change_files(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    node = NodeSpec(id="n1", goal="report on the code", kind="analysis", expects_change=False)
    results, _ = attempt(env, lambda w: (w / "notes.md").write_text("notes\n"), node)
    assert results["G1"].status is GateStatus.FAIL


@needs_bwrap
def test_copy_in_files_are_visible_to_checks_only(tmp_path, monkeypatch):
    files = dict(BASE_FILES)
    files[".gitignore"] = "__pycache__/\n.env.test\n"
    env = make_env(tmp_path, monkeypatch, files=files, untracked={".env.test": "DB_PASSWORD=local\n"},
                   copy_in=[".env.test"], checks=Checks(test_full="test -f .env.test"))
    assert env.baseline.runs["test_full"].ok
    results, outcome = attempt(env, good_change)
    assert results["G5"].status is GateStatus.PASS
    assert not (env.clone.work / ".env.test").exists()


@needs_bwrap
def test_copy_in_paths_are_protected(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch, untracked={"config/test.ini": "password=local\n"},
                   copy_in=["config/test.ini"], checks=Checks(test_full="true"))

    def plant(work):
        good_change(work)
        (work / "config").mkdir()
        (work / "config" / "test.ini").symlink_to("../app.py")

    results, outcome = attempt(env, plant)
    assert results["G2"].status is GateStatus.FAIL
    assert "config/test.ini: protected path" in results["G2"].details["problems"]
    assert outcome is AttemptOutcome.FAILED


@needs_bwrap
def test_missing_copy_in_source_is_not_run(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch, checks=Checks(test_full="true"))
    env.runner.profile = env.runner.profile.model_copy(update={"copy_in": ["config/absent.ini"]})
    results, outcome = attempt(env, good_change)
    assert results["G4"].status is GateStatus.NOT_RUN
    assert outcome is AttemptOutcome.UNVERIFIED


@needs_bwrap
def test_checks_cannot_write_into_dependency_dirs(tmp_path, monkeypatch):
    files = dict(BASE_FILES)
    files[".gitignore"] = "__pycache__/\nnode_modules/\n"
    env = make_env(tmp_path, monkeypatch, files=files, deps_paths=["node_modules"],
                   checks=Checks(test_full="true"))
    (env.clone.work / "node_modules").mkdir()
    (env.clone.work / "node_modules" / "jest").write_text("real jest\n")
    node = NodeSpec(id="n1", goal="change", acceptance=["echo fake > node_modules/jest"])
    results, _ = attempt(env, good_change, node)
    assert results["G5"].status is GateStatus.FAIL
    assert (env.clone.work / "node_modules" / "jest").read_text() == "real jest\n"


@needs_bwrap
def test_a_dependency_dir_swapped_for_a_symlink_fails(tmp_path, monkeypatch):
    files = dict(BASE_FILES)
    files[".gitignore"] = "__pycache__/\n.venv\n"
    env = make_env(tmp_path, monkeypatch, files=files, deps_paths=[".venv"], checks=Checks(test_full="true"))

    def swap(work):
        good_change(work)
        (work / ".venv").symlink_to(Path.home())

    results, outcome = attempt(env, swap)
    assert results["G4"].status is GateStatus.FAIL
    assert "unsafe work tree" in results["G4"].summary
    assert outcome is AttemptOutcome.FAILED


@needs_bwrap
def test_checks_use_job_caches_not_the_users(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch, checks=Checks(test_full="true"))
    spec = env.runner.spec(tmp_path / "out")
    for var in ("XDG_CACHE_HOME", "npm_config_cache", "PIP_CACHE_DIR", "GOMODCACHE", "CARGO_HOME"):
        assert spec.env[var].startswith(str(env.clone.root))


@needs_bwrap
def test_a_crashing_gate_fails_the_attempt(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)

    def boom(ctx):
        raise RuntimeError("bug in a gate")

    monkeypatch.setattr(gates, "gate_scope", boom)
    results, outcome = attempt(env, good_change)
    assert results["G2"].status is GateStatus.FAIL
    assert "crashed" in results["G2"].summary
    assert outcome is AttemptOutcome.FAILED
