"""Coverage-gap tests for mullet.core.orchestration.ralph."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from mullet.core.agent.types import AgentResult, TokenUsage
from mullet.core.orchestration.ralph import (
    PRD,
    RalphMode,
    RalphResult,
    StoryTask,
    UserStory,
    VerificationState,
    _extract_learnings,
    _git_commits_since,
    _git_head_sha,
)
from mullet.core.orchestration.ralphthon import (
    RalphthonPRD,
    RalphthonState,
    HardeningTask,
)


# --- _git_head_sha gaps: lines 39-40 ---

class TestGitHeadShaGaps:
    def test_git_head_sha_exception_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("git not found")),
        )
        assert _git_head_sha() is None


# --- _git_commits_since gaps: lines 47-57 ---

class TestGitCommitsSinceGaps:
    def test_commits_since_none_sha_returns_empty(self):
        assert _git_commits_since(None) == []
        assert _git_commits_since("") == []

    def test_commits_since_subprocess_error_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("git boom")),
        )
        assert _git_commits_since("abc1234") == []

    def test_commits_since_nonzero_rc_returns_empty(self, monkeypatch):
        mock_result = MagicMock(returncode=1, stdout="bad ref")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)
        assert _git_commits_since("abc1234") == []

    def test_commits_since_success(self, monkeypatch):
        mock_result = MagicMock(returncode=0, stdout="abc1234 feat: add x\ndef5678 test: add y\n")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)
        commits = _git_commits_since("HEAD~2")
        assert len(commits) == 2
        assert "feat: add x" in commits[0]


# --- _extract_learnings gaps: lines 117, 129 ---

class TestExtractLearningsGaps:
    def test_empty_response_returns_empty(self):
        assert _extract_learnings("") == []
        assert _extract_learnings(None) == []

    def test_match_appends_learning(self):
        text = "During implementation I learned that the API rate limit is 100 requests per minute."
        learnings = _extract_learnings(text)
        assert len(learnings) >= 1
        assert "API rate limit" in learnings[0]

    def test_note_pattern_match(self):
        text = "Note: the database connection should be pooled for better performance under load."
        learnings = _extract_learnings(text)
        assert len(learnings) >= 1
        assert "database connection" in learnings[0]


# --- PRD.from_dict gaps: lines 237-239 ---

class TestPRDFromDictGaps:
    def test_from_dict_git_exception_uses_defaults(self, monkeypatch):
        """When git subprocess raises, project/branch fall back to 'unknown'."""
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("git boom")),
        )
        data = {
            "task": "test",
            "stories": [{"id": "US-001", "title": "Story"}],
        }
        prd = PRD.from_dict(data)
        assert prd.project == "unknown"
        assert prd.branchName == "unknown"


# --- execute() gaps: lines 356-360, 378, 429-432, 451-452, 480 ---

class TestExecuteGaps:
    @pytest.fixture(autouse=True)
    def clean_state(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def test_execute_resumes_from_saved_state(self):
        """Cover lines 356-360: resume branch when saved state has prd."""
        saved_prd = PRD(
            task="resumed task",
            stories=[
                UserStory(id="US-001", title="Resume story", acceptance_criteria=["c1"], passes=False, priority=1),
            ],
        )
        ralph = RalphMode(max_iterations=5)
        ralph._state.save({"prd": saved_prd.to_dict(), "task": "resumed task"})

        agent = MagicMock()
        agent.allowed_providers = frozenset({"test"})
        agent.budget = MagicMock(max_total=100)
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="Implemented.",
            tokens_used=TokenUsage(),
        )

        with patch("mullet.core.orchestration.verifier.verify_criteria") as mock_verify, \
             patch.object(ralph, "_architect_verify") as mock_arch:
            mock_verify.return_value = MagicMock(
                passed=True,
                summary="PASS: 1/1 criteria met (attempt 1)",
                tokens_used=TokenUsage(),
                issues=[],
            )
            mock_arch.return_value = RalphResult(
                completed=True,
                stories_completed=1,
                stories_total=1,
                iterations_used=1,
                total_tokens=TokenUsage(),
                summary="Architect approved.",
                verification_state=VerificationState(architect_approved=True),
            )
            result = ralph.execute("", agent)

        assert result.stories_completed == 1
        assert result.stories_total == 1
        assert result.completed is True

    def test_execute_no_next_story_break(self):
        """Cover line 378: break when next_story is None despite all_pass being False."""
        ralph = RalphMode(max_iterations=5)
        prd = PRD(
            task="test",
            stories=[
                UserStory(id="US-001", title="Story", acceptance_criteria=["c1"], passes=False, priority=1),
            ],
        )

        # Force impossible state: all_pass False but next_story None
        with patch.object(PRD, "all_pass", new_callable=PropertyMock, return_value=False), \
             patch.object(PRD, "next_story", new_callable=PropertyMock, return_value=None):
            agent = MagicMock()
            agent.allowed_providers = frozenset({"test"})
            agent.budget = MagicMock(max_total=100)
            agent.run.return_value = AgentResult(
                completed=True, final_response="ok", tokens_used=TokenUsage()
            )
            with patch.object(ralph, "_generate_prd", return_value=prd):
                result = ralph.execute("test", agent)

        assert result.completed is False
        assert result.iterations_used == 0

    def test_execute_tdd_violations_logged(self):
        """Cover lines 429-432: TDD violations branch."""
        ralph = RalphMode(max_iterations=5)
        prd = PRD(
            task="test",
            stories=[
                UserStory(id="US-001", title="Story", acceptance_criteria=["c1"], passes=False, priority=1),
            ],
        )

        agent = MagicMock()
        agent.allowed_providers = frozenset({"test"})
        agent.budget = MagicMock(max_total=100)
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="Implemented.",
            tokens_used=TokenUsage(),
        )

        with patch("mullet.core.orchestration.verifier.verify_criteria") as mock_verify, \
             patch("mullet.core.orchestration.ralph._git_commits_since") as mock_commits, \
             patch("mullet.core.orchestration.ralph._git_head_sha", return_value="abc1234"), \
             patch.object(ralph, "_architect_verify") as mock_arch:
            mock_verify.return_value = MagicMock(
                passed=True,
                summary="PASS: 1/1 criteria met (attempt 1)",
                tokens_used=TokenUsage(),
                issues=[],
            )
            # feat without test triggers TDD advisory
            mock_commits.return_value = ["abc1234 feat: add login form"]
            mock_arch.return_value = RalphResult(
                completed=True,
                stories_completed=1,
                stories_total=1,
                iterations_used=1,
                total_tokens=TokenUsage(),
                summary="Approved.",
                verification_state=VerificationState(architect_approved=True),
            )

            with patch.object(ralph, "_generate_prd", return_value=prd):
                result = ralph.execute("test", agent)

        assert result.stories_completed == 1
        story = prd.stories[0]
        assert "[TDD:" in story.notes

    def test_execute_no_acceptance_criteria_trusts_impl(self):
        """Cover lines 451-452: story with no acceptance criteria passes automatically."""
        ralph = RalphMode(max_iterations=5)
        prd = PRD(
            task="test",
            stories=[
                UserStory(id="US-001", title="Story", acceptance_criteria=[], passes=False, priority=1),
            ],
        )

        agent = MagicMock()
        agent.allowed_providers = frozenset({"test"})
        agent.budget = MagicMock(max_total=100)
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="Implemented.",
            tokens_used=TokenUsage(),
        )

        with patch.object(ralph, "_generate_prd", return_value=prd), \
             patch.object(ralph, "_architect_verify") as mock_arch:
            mock_arch.return_value = RalphResult(
                completed=True,
                stories_completed=1,
                stories_total=1,
                iterations_used=1,
                total_tokens=TokenUsage(),
                summary="Approved.",
                verification_state=VerificationState(architect_approved=True),
            )
            result = ralph.execute("test", agent)

        assert result.stories_completed == 1
        assert result.completed is True

    def test_execute_tdd_advisory_summary_appended(self):
        """Cover line 480: TDD advisory summary appended to final text."""
        ralph = RalphMode(max_iterations=5)
        prd = PRD(
            task="test",
            stories=[
                UserStory(
                    id="US-001", title="Story",
                    acceptance_criteria=["c1"],
                    passes=True, priority=1,
                    notes="PASS [TDD: advisory here]",
                ),
            ],
        )

        agent = MagicMock()
        agent.allowed_providers = frozenset({"test"})
        agent.budget = MagicMock(max_total=100)
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="APPROVAL.",
            tokens_used=TokenUsage(),
        )

        with patch.object(ralph, "_generate_prd", return_value=prd):
            result = ralph.execute("test", agent)

        assert "TDD advisory" in result.summary
        assert "story lacked" in result.summary or "stories lacked" in result.summary


# --- _generate_prd gaps: lines 520, 532-533 ---

class TestGeneratePRDGaps:
    def test_generate_prd_agent_not_completed(self):
        """Cover line 520: agent.run returns not completed."""
        ralph = RalphMode()
        agent = MagicMock()
        agent.run.return_value = AgentResult(
            completed=False, final_response="", tokens_used=TokenUsage()
        )
        prd = ralph._generate_prd("some task", agent)
        assert prd.task == "some task"
        assert prd.stories == []

    def test_generate_prd_json_decode_error(self):
        """Cover lines 532-533: JSON decode error."""
        ralph = RalphMode()
        agent = MagicMock()
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="this is not {valid json",
            tokens_used=TokenUsage(),
        )
        prd = ralph._generate_prd("some task", agent)
        assert prd.task == "some task"
        assert prd.stories == []

    def test_generate_prd_key_error(self):
        """Cover lines 532-533: KeyError from missing story keys."""
        ralph = RalphMode()
        agent = MagicMock()
        agent.run.return_value = AgentResult(
            completed=True,
            final_response='{"stories": [{"title": "missing id"}]}',
            tokens_used=TokenUsage(),
        )
        prd = ralph._generate_prd("some task", agent)
        assert prd.task == "some task"
        assert prd.stories == []


# --- _implement_story gaps: lines 554-559 ---

class TestImplementStoryGaps:
    def test_implement_story_with_completed_stories_notes(self):
        """Cover lines 554-559: completed stories with notes appear in context."""
        ralph = RalphMode()
        story = UserStory(
            id="US-002", title="Second",
            acceptance_criteria=["c1"],
            tasks=[StoryTask(id="T-1", title="sub", description="desc")],
        )
        completed = [
            UserStory(id="US-001", title="First", notes="Some notes here"),
        ]
        agent = MagicMock()
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="Done.",
            tokens_used=TokenUsage(),
        )
        result = ralph._implement_story(
            story, "task", agent,
            agent_profile="default", max_tokens=100,
            completed_stories=completed,
            prior_learnings=[],
        )
        prompt = agent.run.call_args[0][0]
        assert "Previously completed stories" in prompt
        assert "Some notes here" in prompt
        assert "Sub-tasks" in prompt
        assert story.tasks[0].completed is True


# --- _verify_story gaps: lines 599-608 ---

class TestVerifyStoryGaps:
    def test_verify_story_builds_prompt_and_calls_agent(self):
        """Cover lines 599-608: _verify_story builds criteria prompt."""
        ralph = RalphMode()
        story = UserStory(
            id="US-001", title="Story",
            acceptance_criteria=["criterion one", "criterion two"],
        )
        agent = MagicMock()
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="ALL CRITERIA PASS",
            tokens_used=TokenUsage(),
        )
        result = ralph._verify_story(story, agent)
        prompt = agent.run.call_args[0][0]
        assert "Verify each acceptance criterion" in prompt
        assert "criterion one" in prompt
        assert "criterion two" in prompt
        assert result.final_response == "ALL CRITERIA PASS"


# --- _architect_verify gaps: lines 660-663 ---

class TestArchitectVerifyGaps:
    def test_architect_verify_max_attempts_zero(self):
        """Cover lines 660-663: max attempts reached without approval."""
        ralph = RalphMode()
        prd = PRD(
            task="test",
            stories=[UserStory(id="US-001", title="Story", passes=True)],
        )
        agent = MagicMock()
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="NEEDS WORK",
            tokens_used=TokenUsage(),
        )
        # Patch VerificationState so max_verification_attempts=0, causing the
        # while loop body to never run and hitting lines 659-663.
        zero_attempt_state = VerificationState(
            original_task=prd.task, max_verification_attempts=0
        )
        with patch(
            "mullet.core.orchestration.ralph.VerificationState",
            return_value=zero_attempt_state,
        ):
            result = ralph._architect_verify(prd, agent, TokenUsage())
        assert result.completed is False
        assert result.verification_state.architect_approved is False
        assert "Max verification attempts reached" in result.verification_state.architect_feedback


# --- _final_summary gaps: lines 677-685 ---

class TestFinalSummaryGaps:
    def test_final_summary_builds_prompt(self):
        """Cover lines 677-685: _final_summary builds prompt and calls agent."""
        ralph = RalphMode()
        prd = PRD(
            task="build feature",
            stories=[
                UserStory(id="US-001", title="Auth", passes=True),
                UserStory(id="US-002", title="DB", passes=True),
            ],
        )
        agent = MagicMock()
        agent.run.return_value = AgentResult(
            completed=True,
            final_response="Summary of work.",
            tokens_used=TokenUsage(),
        )
        result = ralph._final_summary(prd, agent)
        prompt = agent.run.call_args[0][0]
        assert "All stories" in prompt
        assert "Auth" in prompt
        assert "DB" in prompt
        assert result.final_response == "Summary of work."


# --- ralphthon.py gaps ---

class TestRalphthonStateGaps:
    def test_get_next_task_complete_phase_returns_none(self):
        """Cover line 128: get_next_task returns None when phase is not execution/hardening."""
        prd = RalphthonPRD(project="test", description="test")
        state = RalphthonState(prd)
        state.phase = "complete"
        assert state.get_next_task() is None

    def test_find_task_hardening_path(self):
        """Cover lines 171-172: _find_task searches hardening tasks."""
        prd = RalphthonPRD(
            project="test",
            description="test",
            hardening=[HardeningTask(id="H1", title="Harden 1", description="harden")],
        )
        state = RalphthonState(prd)
        result = state._find_task("H1")
        assert result is not None
        assert result.id == "H1"
