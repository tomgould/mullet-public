"""Tests for mullet.jobs.context: the brief, the repo map and the files picked for pre-loading."""

from pathlib import Path

import pytest

from mullet.jobs import context
from mullet.jobs.clone import JobClone, resolve_commit
from mullet.jobs.context import (
    ContextFile, build_brief, direct_imports, gather, read_text, related_tests, repo_map, scope_seeds,
    select_context_files, symbol_outline,
)
from mullet.jobs.executors.base import NodeContext, Task
from mullet.jobs.profile import Checks, Profile
from mullet.jobs.types import NodeSpec

from tests.jobs_helpers import make_source_repo, write


def make_task(files, scope, goal="change things", **ctx) -> Task:
    return Task("the job goal", NodeSpec(id="n1", goal=goal, scope=scope),
                context=NodeContext(files=sorted(files), **ctx))


def tree(root: Path, files: dict[str, str]) -> Path:
    write(root, files)
    return root


# Imports


def test_python_imports_resolve_absolute_relative_and_submodules():
    tracked = {"pkg/__init__.py", "pkg/a.py", "pkg/b.py", "pkg/sub/__init__.py", "pkg/sub/c.py",
               "src/lib/util.py", "other.py"}
    text = (
        "import os\n"
        "import pkg.a as alias, other\n"
        "from pkg import b\n"
        "from .sub import c\n"
        "from . import a\n"
        "from lib.util import helper\n"
        "from pkg.sub import (\n    c,\n)\n"
    )
    found = direct_imports("pkg/main.py", text, tracked)
    assert found[:3] == ["pkg/a.py", "other.py", "pkg/__init__.py"]
    assert {"pkg/b.py", "pkg/sub/__init__.py", "pkg/sub/c.py", "src/lib/util.py"} <= set(found)
    assert "os" not in "".join(found)
    # Names in parentheses over several lines count too.
    assert direct_imports("x.py", "from pkg.sub import (\n    c,\n)\n", tracked) == [
        "pkg/sub/__init__.py", "pkg/sub/c.py"]


def test_js_and_ts_relative_imports_resolve_extensions_and_index_files():
    tracked = {"src/a.ts", "src/b/index.tsx", "src/c.js", "src/d.ts", "lib/e.mjs"}
    text = (
        "import a from './a'\n"
        "import { B } from \"./b\"\n"
        "const c = require('./c.js')\n"
        "export * from './d.js'\n"
        "import('../lib/e.mjs')\n"
        "import React from 'react'\n"
    )
    assert direct_imports("src/main.ts", text, tracked) == [
        "src/a.ts", "src/b/index.tsx", "src/c.js", "src/d.ts", "lib/e.mjs"]


def test_php_use_and_require_resolve_by_namespace_tail():
    tracked = {"src/Service/Mailer.php", "modules/x/src/Service/Mailer.php", "src/Util.php",
               "web/core/Util.php", "includes/helpers.inc", "lib/Only.php"}
    text = (
        "<?php\nuse App\\Service\\Mailer;\nuse Vendor\\Util;\nuse Some\\Only as O;\n"
        "require_once __DIR__ . '/../includes/helpers.inc';\n"
    )
    found = direct_imports("src/Controller.php", text, tracked)
    assert "modules/x/src/Service/Mailer.php" in found or "src/Service/Mailer.php" in found
    assert "lib/Only.php" in found  # a bare class name that only one file has
    assert "src/Util.php" not in found and "web/core/Util.php" not in found  # ambiguous: skipped
    assert "includes/helpers.inc" in found


def test_other_languages_have_no_imports():
    assert direct_imports("README.md", "import x", {"x.py"}) == []


# Picking files


def test_seeds_come_from_scope_and_broad_scopes_use_the_goal(tmp_path):
    files = [f"pkg/mod{i}.py" for i in range(60)] + ["pkg/opencode_parser.py", "docs/opencode.md"]
    tight = make_task(files, ["pkg/mod1.py", "pkg/mod2.py"])
    assert scope_seeds(tight) == ["pkg/mod1.py", "pkg/mod2.py"]
    broad = make_task(files, ["**"], goal="Fix the opencode parser")
    assert scope_seeds(broad)[:2] == ["pkg/opencode_parser.py", "docs/opencode.md"]
    assert len(scope_seeds(broad)) == 2


def test_selection_order_budget_and_exclusions(tmp_path):
    work = tree(tmp_path / "w", {
        "app/core.py": "from app import helpers\nimport app.big\n",
        "app/helpers.py": "def h():\n    return 1\n",
        "app/big.py": "x = 1\n" * 4000,
        "app/__init__.py": "",
        "tests/test_core.py": "def test_core():\n    assert True\n",
        "tests/test_app_core_extra.py": "def test_more():\n    assert True\n",
        "app/.env": "SECRET=1\n",
        "app/settings.py": "secret = 1\n",
    })
    files = [str(p.relative_to(work)) for p in work.rglob("*") if p.is_file()]
    task = make_task(files, ["app/core.py", "app/.env", "app/settings.py"], exclude=["app/settings.py"])
    picked = select_context_files(task, work, budget_tokens=2000)
    paths = [f.path for f in picked]
    assert paths[0] == "app/core.py"
    assert paths.index("tests/test_core.py") < paths.index("app/helpers.py")
    assert "tests/test_app_core_extra.py" in paths  # named after the package and the file
    assert "app/big.py" not in paths  # over the budget, skipped; smaller ones still go in
    assert "app/.env" not in paths and "app/settings.py" not in paths
    assert sum(f.tokens for f in picked) <= 2000
    reasons = {f.path: f.reason for f in picked}
    assert reasons["app/helpers.py"] == "imported by app/core.py"


def test_symlinks_and_binaries_are_never_read(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "id_rsa").write_text("PRIVATE KEY")
    work = tree(tmp_path / "w", {"real.py": "x = 1\n"})
    (work / "link.py").symlink_to(outside / "id_rsa")
    (work / "dir").symlink_to(outside)
    (work / "blob.py").write_bytes(b"\x00\x01binary")
    assert read_text(work, "real.py") == "x = 1\n"
    for rel in ("link.py", "dir/id_rsa", "blob.py", "../outside/id_rsa", "missing.py"):
        assert read_text(work, rel) is None
    task = make_task(["real.py", "link.py", "dir/id_rsa", "blob.py"], ["**"])
    assert [f.path for f in select_context_files(task, work, 10_000)] == ["real.py"]


def test_related_tests_need_the_name_and_a_directory():
    files = ["tests/test_jobs_runner.py", "tests/test_agent_runner.py", "tests/test_runner.py"]
    assert related_tests(["mullet/jobs/runner.py"], files) == ["tests/test_jobs_runner.py"]


def test_budget_is_a_share_of_the_context_window():
    assert context.context_budget(None, 0.35, context_length=100_000) == 35_000
    assert context.context_budget("openrouter/qwen/qwen3-coder") > 100_000


# The brief


def test_repo_map_groups_by_directory_and_shrinks_to_fit():
    files = [f"d{i}/f{j}.py" for i in range(40) for j in range(30)] + ["hot/a.py"]
    full = repo_map(files, ["hot/a.py"], max_chars=100_000)
    assert full.splitlines()[0] == "hot/ a.py"
    assert "d0/ f0.py" in full and "+15 more" in full
    small = repo_map(files, ["hot/a.py"], max_chars=400)
    assert small.splitlines()[0] == "hot/ a.py"
    assert "d0/ (30 files)" in small and "more directories" in small
    assert len(small) <= 450


def test_symbol_outline_lists_classes_and_functions(tmp_path):
    work = tree(tmp_path / "w", {"m.py": "class A:\n    def run(self, x):\n        pass\n\ndef top(y):\n    pass\n"})
    outline = symbol_outline(work, ["m.py", "notes.txt"])
    assert outline.splitlines() == ["m.py", "  1: class A:", "    2: def run(self, x):", "  5: def top(y):"]


def test_brief_carries_everything_mullet_knows(tmp_path):
    work = tree(tmp_path / "w", {"app.py": "def add(a, b):\n    return a + b\n", "README.md": "hi\n"})
    task = Task(
        "Add subtraction", NodeSpec(id="sub", goal="add sub()", scope=["app.py"], acceptance=["make test"],
                                    allow_protected=["uv.lock"]),
        feedback=["G5 failed: test_app::test_add"],
        context=NodeContext(parent_goal="arithmetic", plan_notes="watch the rounding",
                            history="commit abc123 add mul()\n app.py | 3 +++", files=["README.md", "app.py"],
                            checks={"test_full": "make test"}),
    )
    brief = build_brief(task, work, attached=[ContextFile("app.py", "in scope", 12)], rules="- be good")
    for text in ("Add subtraction", "## This step: sub (design)", "add sub()", "It is part of: arithmetic",
                 "`app.py`", "`uv.lock`", "make test", "## Why earlier attempts failed", "G5 failed",
                 "watch the rounding", "commit abc123 add mul()", "test_full: make test",
                 "- `app.py` (in scope)", "1: def add(a, b):", "## Repository map (2 tracked files)",
                 "- be good"):
        assert text in brief, text


def test_brief_without_context_still_has_the_task(tmp_path):
    brief = build_brief(Task("goal", NodeSpec(id="n1", goal="do x")), tmp_path)
    assert "goal" in brief and "do x" in brief and "Repository map" not in brief


@pytest.fixture
def clone(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    source = make_source_repo(tmp_path / "src", {"app.py": "x = 1\n", "tests/test_app.py": "def test_x():\n    pass\n"})
    base = resolve_commit(source, "HEAD")
    return JobClone.create(source, base, tmp_path / "jobs" / "j1" / "clone", "mullet/j1"), base


def test_gather_reads_history_files_and_checks_from_the_clone(clone):
    job_clone, base = clone
    profile = Profile(repo_id="x", checks=Checks(test_full="python3 runtests.py"), copy_in=["config/test.env"])
    first = gather(job_clone, base, profile, parent_goal="p", plan_notes="n")
    assert first.history == "" and first.files == ["app.py", "tests/test_app.py"]
    assert first.checks == {"test_full": "python3 runtests.py"}
    assert "config/test.env" in first.exclude and ".env" in first.exclude
    (job_clone.work / "b.py").write_text("y = 2\n")
    snap = job_clone.snapshot()
    job_clone.commit_snapshot(snap, "add b")
    later = gather(job_clone, base, profile)
    assert "add b" in later.history and "b.py" in later.history
    assert "b.py" in later.files


def test_gather_trims_long_history(clone):
    job_clone, base = clone
    for i in range(40):
        (job_clone.work / f"f{i}.py").write_text("x\n" * i)
        job_clone.commit_snapshot(job_clone.snapshot(), f"commit number {i} " + "x" * 60)
    assert len(gather(job_clone, base, Profile(repo_id="x")).history) <= context.HISTORY_CHARS + 20
