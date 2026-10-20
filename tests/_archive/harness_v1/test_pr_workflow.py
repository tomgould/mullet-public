"""Tests for the PR creation workflow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mullet.harness.workflows.pr_workflow import PRResult, PRWorkflow


class TestSlugify:
    def test_slugify_basic(self):
        wf = PRWorkflow()
        assert wf._slugify("Add Pydantic validation") == "add-pydantic-validation"

    def test_slugify_truncates_long_strings(self):
        wf = PRWorkflow()
        long_task = "Implement a very long feature description that exceeds the maximum allowed length"
        slug = wf._slugify(long_task, max_len=40)
        assert len(slug) <= 40
        assert slug.startswith("implement-a-very-long-feature")
        assert not slug.endswith("-")


class TestCommitMessage:
    def test_generate_commit_message_format(self):
        wf = PRWorkflow()
        task = "Add user authentication\n\nThis is the body of the task."
        files = ["src/auth.py", "tests/test_auth.py"]
        msg = wf._generate_commit_message(task, files)
        lines = msg.split("\n")
        assert lines[0] == "Add user authentication"
        assert "Files changed:" in lines
        assert "  - src/auth.py" in lines
        assert "  - tests/test_auth.py" in lines


class TestPreCheck:
    def test_pre_check_fails_on_dirty_tree(self):
        wf = PRWorkflow()
        with patch.object(wf, "_git") as mock_git, patch.object(wf, "_gh") as mock_gh:
            mock_git.return_value = MagicMock(
                stdout=" M src/main.py\n", stderr="", returncode=0
            )
            mock_gh.return_value = MagicMock(
                stdout="gh version 2.0.0\n", stderr="", returncode=0
            )
            error = wf._pre_check()
            assert error is not None
            assert "not clean" in error

    def test_pre_check_fails_when_gh_not_installed(self):
        wf = PRWorkflow()
        with patch.object(wf, "_git") as mock_git, patch.object(wf, "_gh") as mock_gh:
            mock_git.return_value = MagicMock(stdout="", stderr="", returncode=0)
            mock_gh.return_value = MagicMock(
                stdout="", stderr="command not found", returncode=127
            )
            error = wf._pre_check()
            assert error is not None
            assert "not installed" in error


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_creates_branch(self):
        engine = MagicMock()
        engine.arun = AsyncMock(return_value={
            "session_id": "sess-123",
            "response": "Done",
        })

        wf = PRWorkflow(engine_factory=lambda: engine)

        git_calls = [
            # 0. pre-check status
            MagicMock(stdout="", stderr="", returncode=0),
            # 1. current branch
            MagicMock(stdout="main\n", stderr="", returncode=0),
            # 2. base branch rev-parse
            MagicMock(stdout="main\n", stderr="", returncode=0),
            # 3. checkout -b
            MagicMock(stdout="", stderr="", returncode=0),
            # 4. post-agent status
            MagicMock(stdout=" M src/auth.py\n", stderr="", returncode=0),
            # 5. add
            MagicMock(stdout="", stderr="", returncode=0),
            # 6. commit (variable args, caught by index)
            MagicMock(stdout="", stderr="", returncode=0),
            # 7. rev-parse HEAD
            MagicMock(stdout="abc123\n", stderr="", returncode=0),
            # 8. push
            MagicMock(stdout="", stderr="", returncode=0),
        ]

        gh_calls = [
            # 0. version
            MagicMock(stdout="gh version 2.0.0\n", stderr="", returncode=0),
            # 1. auth
            MagicMock(stdout="ok\n", stderr="", returncode=0),
            # 2. pr create (variable args, caught by index)
            MagicMock(
                stdout="https://github.com/org/repo/pull/42\n", stderr="", returncode=0
            ),
        ]

        with patch.object(wf, "_git", side_effect=git_calls), patch.object(
            wf, "_gh", side_effect=gh_calls
        ):
            result = await wf.execute("add auth")

        assert result.branch == "feat/mullet-add-auth"
        assert result.success is True
        assert result.commit_sha == "abc123"
        assert result.pr_url == "https://github.com/org/repo/pull/42"
        assert result.files_changed == ["src/auth.py"]

    @pytest.mark.asyncio
    async def test_execute_returns_pr_result_on_success(self):
        engine = MagicMock()
        engine.arun = AsyncMock(return_value={
            "session_id": "sess-456",
            "response": "Implemented",
        })

        wf = PRWorkflow(engine_factory=lambda: engine)

        git_calls = [
            MagicMock(stdout="", stderr="", returncode=0),  # pre-check status
            MagicMock(stdout="main\n", stderr="", returncode=0),  # current branch
            MagicMock(stdout="main\n", stderr="", returncode=0),  # base branch
            MagicMock(stdout="", stderr="", returncode=0),  # checkout -b
            MagicMock(stdout=" M src/foo.py\n", stderr="", returncode=0),  # post-agent status
            MagicMock(stdout="", stderr="", returncode=0),  # add
            MagicMock(stdout="", stderr="", returncode=0),  # commit
            MagicMock(stdout="defabc\n", stderr="", returncode=0),  # rev-parse HEAD
            MagicMock(stdout="", stderr="", returncode=0),  # push
        ]

        gh_calls = [
            MagicMock(stdout="gh version 2.0.0\n", stderr="", returncode=0),  # version
            MagicMock(stdout="ok\n", stderr="", returncode=0),  # auth
            MagicMock(
                stdout="https://github.com/org/repo/pull/99\n", stderr="", returncode=0
            ),  # pr create
        ]

        with patch.object(wf, "_git", side_effect=git_calls), patch.object(
            wf, "_gh", side_effect=gh_calls
        ):
            result = await wf.execute("fix bug")

        assert isinstance(result, PRResult)
        assert result.success is True
        assert result.pr_url == "https://github.com/org/repo/pull/99"
        assert result.commit_sha == "defabc"
        assert result.files_changed == ["src/foo.py"]
        assert result.error is None

    @pytest.mark.asyncio
    async def test_dry_run_skips_push_and_pr_create(self):
        engine = MagicMock()
        engine.arun = AsyncMock(return_value={
            "session_id": "sess-789",
            "response": "Done",
        })

        wf = PRWorkflow(engine_factory=lambda: engine)

        git_calls = [
            MagicMock(stdout="", stderr="", returncode=0),  # pre-check status
            MagicMock(stdout="main\n", stderr="", returncode=0),  # current branch
            MagicMock(stdout="main\n", stderr="", returncode=0),  # base branch
            MagicMock(stdout="", stderr="", returncode=0),  # checkout -b
            MagicMock(stdout=" M src/bar.py\n", stderr="", returncode=0),  # post-agent status
            MagicMock(stdout="", stderr="", returncode=0),  # checkout original
        ]

        gh_calls = [
            MagicMock(stdout="gh version 2.0.0\n", stderr="", returncode=0),  # version
            MagicMock(stdout="ok\n", stderr="", returncode=0),  # auth
        ]

        with patch.object(wf, "_git", side_effect=git_calls) as mock_git, patch.object(
            wf, "_gh", side_effect=gh_calls
        ) as mock_gh:
            result = await wf.execute("update bar", dry_run=True)

            # Ensure push and pr create were never called
            git_call_args = [call.args for call in mock_git.call_args_list]
            assert not any(args[0] == "push" for args in git_call_args)
            gh_call_args = [call.args for call in mock_gh.call_args_list]
            assert not any(args[0] == "pr" for args in gh_call_args)

        assert result.success is True
        assert result.pr_url is None
        assert result.commit_sha is None
        assert result.files_changed == ["src/bar.py"]
