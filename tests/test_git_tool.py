"""Tests for the git integration tool."""

import json
import subprocess
import pytest
from mullet.core.tools.git import git_tool


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """Create a temporary git repo for testing."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "hello.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, capture_output=True)
    return tmp_path


def test_status_clean(git_repo):
    result = json.loads(git_tool(json.dumps({"action": "status"})))
    assert result["clean"] is True
    assert result["total_changes"] == 0


def test_status_with_changes(git_repo):
    (git_repo / "new_file.py").write_text("x = 1\n")
    result = json.loads(git_tool(json.dumps({"action": "status"})))
    assert result["clean"] is False
    assert result["total_changes"] >= 1
    assert any(c["file"] == "new_file.py" for c in result["changes"])


def test_status_shows_branch(git_repo):
    result = json.loads(git_tool(json.dumps({"action": "status"})))
    assert result["branch"] != ""


def test_log(git_repo):
    result = json.loads(git_tool(json.dumps({"action": "log", "count": 5})))
    assert result["count"] >= 1
    assert result["entries"][0]["message"] == "Initial commit"
    assert len(result["entries"][0]["hash"]) == 8


def test_diff_no_changes(git_repo):
    result = json.loads(git_tool(json.dumps({"action": "diff"})))
    assert result["diff"] == ""


def test_diff_with_changes(git_repo):
    (git_repo / "hello.py").write_text("print('world')\n")
    result = json.loads(git_tool(json.dumps({"action": "diff"})))
    assert "world" in result["diff"]
    assert "hello" in result["diff"]


def test_branches(git_repo):
    result = json.loads(git_tool(json.dumps({"action": "branches"})))
    assert len(result["branches"]) >= 1
    assert result["current"] != ""


def test_show(git_repo):
    result = json.loads(git_tool(json.dumps({"action": "show", "ref": "HEAD"})))
    assert "Initial commit" in result["content"]


def test_blame(git_repo):
    result = json.loads(git_tool(json.dumps({"action": "blame", "path": "hello.py"})))
    assert "blame" in result
    assert "hello" in result["blame"]


def test_blame_missing_path():
    result = json.loads(git_tool(json.dumps({"action": "blame"})))
    assert "error" in result


def test_unknown_action():
    result = json.loads(git_tool(json.dumps({"action": "rebase"})))
    assert "error" in result


def test_invalid_json():
    result = json.loads(git_tool("not json"))
    assert "error" in result


class TestGitMissingLines:
    def test_run_git_timeout(self, monkeypatch):
        def raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired("git", 15)
        monkeypatch.setattr(subprocess, "run", raise_timeout)
        from mullet.core.tools.git import _run_git
        output, code = _run_git(["status"])
        assert "timed out" in output
        assert code == 1

    def test_run_git_not_found(self, monkeypatch):
        def raise_filenotfound(*args, **kwargs):
            raise FileNotFoundError()
        monkeypatch.setattr(subprocess, "run", raise_filenotfound)
        from mullet.core.tools.git import _run_git
        output, code = _run_git(["status"])
        assert "Git not found" in output
        assert code == 1

    def test_git_tool_kwargs_branch(self, git_repo):
        result = json.loads(git_tool(action="status"))
        assert result["clean"] is True

    def test_status_error(self, git_repo, monkeypatch):
        def fake_run_git(args):
            return "fatal: not a git repo", 1
        monkeypatch.setattr("mullet.core.tools.git._run_git", fake_run_git)
        result = json.loads(git_tool(json.dumps({"action": "status"})))
        assert "error" in result

    def test_diff_with_ref(self, git_repo):
        (git_repo / "hello.py").write_text("print('world')\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "second"], cwd=git_repo, capture_output=True)
        result = json.loads(git_tool(json.dumps({"action": "diff", "ref": "HEAD~1"})))
        assert "diff" in result

    def test_diff_with_path(self, git_repo):
        (git_repo / "hello.py").write_text("print('world')\n")
        result = json.loads(git_tool(json.dumps({"action": "diff", "path": "hello.py"})))
        assert "diff" in result

    def test_diff_truncation(self, git_repo, monkeypatch):
        huge = "x" * 40000
        def fake_run_git(args):
            return huge, 0
        monkeypatch.setattr("mullet.core.tools.git._run_git", fake_run_git)
        result = json.loads(git_tool(json.dumps({"action": "diff"})))
        assert result["truncated"] is True
        assert "[... truncated" in result["diff"]

    def test_log_with_path(self, git_repo):
        result = json.loads(git_tool(json.dumps({"action": "log", "path": "hello.py"})))
        assert "entries" in result

    def test_log_error(self, git_repo, monkeypatch):
        def fake_run_git(args):
            return "fatal: bad ref", 1
        monkeypatch.setattr("mullet.core.tools.git._run_git", fake_run_git)
        result = json.loads(git_tool(json.dumps({"action": "log"})))
        assert "error" in result

    def test_blame_error(self, git_repo, monkeypatch):
        def fake_run_git(args):
            return "fatal: no such path", 1
        monkeypatch.setattr("mullet.core.tools.git._run_git", fake_run_git)
        result = json.loads(git_tool(json.dumps({"action": "blame", "path": "nope.py"})))
        assert "error" in result

    def test_blame_truncation(self, git_repo, monkeypatch):
        huge = "x" * 40000
        def fake_run_git(args):
            return huge, 0
        monkeypatch.setattr("mullet.core.tools.git._run_git", fake_run_git)
        result = json.loads(git_tool(json.dumps({"action": "blame", "path": "hello.py"})))
        assert "[truncated]" in result["blame"]

    def test_branches_error(self, git_repo, monkeypatch):
        def fake_run_git(args):
            return "fatal: not a git repo", 1
        monkeypatch.setattr("mullet.core.tools.git._run_git", fake_run_git)
        result = json.loads(git_tool(json.dumps({"action": "branches"})))
        assert "error" in result

    def test_show_with_path(self, git_repo):
        result = json.loads(git_tool(json.dumps({"action": "show", "ref": "HEAD", "path": "hello.py"})))
        assert "content" in result

    def test_show_truncation(self, git_repo, monkeypatch):
        huge = "x" * 40000
        def fake_run_git(args):
            return huge, 0
        monkeypatch.setattr("mullet.core.tools.git._run_git", fake_run_git)
        result = json.loads(git_tool(json.dumps({"action": "show", "ref": "HEAD"})))
        assert "[truncated]" in result["content"]

    def test_register(self):
        class FakeRegistry:
            def __init__(self):
                self.calls = []
            def register(self, **kwargs):
                self.calls.append(kwargs)
        from mullet.core.tools.git import register, GIT_SCHEMA, git_tool
        reg = FakeRegistry()
        register(reg)
        assert len(reg.calls) == 1
        assert reg.calls[0]["name"] == "git"
        assert reg.calls[0]["func"] == git_tool
        assert reg.calls[0]["schema"] == GIT_SCHEMA
