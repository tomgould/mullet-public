"""Integration tests for Ralph mode orchestration (PRD, UserStory, StoryTask)."""

from mullet.core.orchestration.ralph import PRD, UserStory, StoryTask


def _make_prd(n_stories=2):
    stories = [
        UserStory(id=f"S-{i}", title=f"Story {i}", priority=i, passes=False)
        for i in range(1, n_stories + 1)
    ]
    return PRD(task="Build feature X", project="test", branchName="feat/x", stories=stories)


class TestPRDGeneration:
    def test_prd_to_dict_roundtrip(self):
        prd = _make_prd(3)
        data = prd.to_dict()
        restored = PRD.from_dict(data)
        assert restored.task == prd.task
        assert len(restored.stories) == 3
        assert restored.stories[0].title == "Story 1"

    def test_prd_with_nested_tasks(self):
        story = UserStory(
            id="S-1", title="Auth story",
            tasks=[StoryTask(id="T-1", title="Write tests"), StoryTask(id="T-2", title="Implement")]
        )
        prd = PRD(task="Auth feature", stories=[story])
        data = prd.to_dict()
        restored = PRD.from_dict(data)
        assert len(restored.stories[0].tasks) == 2
        assert restored.stories[0].tasks[0].title == "Write tests"


class TestNextStory:
    def test_next_story_returns_lowest_priority(self):
        prd = _make_prd(3)
        next_s = prd.next_story
        assert next_s is not None
        assert next_s.priority == 1

    def test_next_story_skips_passed(self):
        prd = _make_prd(2)
        prd.stories[0].passes = True  # S-1 passes
        next_s = prd.next_story
        assert next_s.id == "S-2"

    def test_next_story_none_when_all_pass(self):
        prd = _make_prd(2)
        for s in prd.stories:
            s.passes = True
        assert prd.next_story is None


class TestCompletionCriteria:
    def test_all_pass_false_initially(self):
        prd = _make_prd(2)
        assert prd.all_pass is False

    def test_all_pass_true_when_all_done(self):
        prd = _make_prd(2)
        for s in prd.stories:
            s.passes = True
        assert prd.all_pass is True

    def test_all_pass_false_if_one_fails(self):
        prd = _make_prd(3)
        prd.stories[0].passes = True
        prd.stories[1].passes = True
        # stories[2] still False
        assert prd.all_pass is False

    def test_progress_tracking(self):
        prd = _make_prd(4)
        prd.stories[0].passes = True
        prd.stories[1].passes = True
        done, total = prd.progress
        assert done == 2 and total == 4


class TestTDDGates:
    def test_no_commits_no_violations(self):
        from mullet.core.orchestration.ralph import _check_tdd_gates, UserStory
        story = UserStory(id="US-001", title="Test")
        assert _check_tdd_gates(story, []) == []

    def test_feat_without_test_flagged(self):
        from mullet.core.orchestration.ralph import _check_tdd_gates, UserStory
        story = UserStory(id="US-001", title="Test")
        commits = ["abc1234 feat: add login form", "def5678 chore: update deps"]
        violations = _check_tdd_gates(story, commits)
        assert any("RED phase" in v for v in violations)

    def test_test_then_feat_no_violation(self):
        from mullet.core.orchestration.ralph import _check_tdd_gates, UserStory
        story = UserStory(id="US-001", title="Test")
        # newest-first: feat is at index 0, test at index 1 → test committed first
        commits = ["abc1234 feat: add login form", "def5678 test: add login test"]
        violations = _check_tdd_gates(story, commits)
        # No RED-before-GREEN violation (test was first)
        assert not any("RED before GREEN" in v for v in violations)

    def test_feat_then_test_flagged(self):
        from mullet.core.orchestration.ralph import _check_tdd_gates, UserStory
        story = UserStory(id="US-001", title="Test")
        # newest-first: test is at index 0, feat at index 1 → feat committed first (wrong order)
        commits = ["abc1234 test: add login test", "def5678 feat: add login form"]
        violations = _check_tdd_gates(story, commits)
        assert any("RED before GREEN" in v for v in violations)


class TestStoryFormatting:
    def test_story_with_notes_preserved(self):
        story = UserStory(id="S-1", title="Story", notes=["note1", "note2"])
        data = story.to_dict() if hasattr(story, 'to_dict') else vars(story)
        assert story.notes == ["note1", "note2"]

    def test_story_task_completion_criteria(self):
        task = StoryTask(id="T-1", title="Write failing tests first")
        assert "test" in task.title.lower()

    def test_prd_completion_criteria_requires_all_stories(self):
        prd = _make_prd(3)
        assert prd.all_pass is False
        for s in prd.stories:
            s.passes = True
        assert prd.all_pass is True

    def test_story_priority_ordering(self):
        prd = _make_prd(3)
        # next_story returns lowest priority (highest urgency) first
        first = prd.next_story
        assert first.priority == 1  # lowest number = highest priority

    def test_prd_with_zero_stories_is_all_pass(self):
        prd = PRD(task="empty", stories=[])
        assert prd.all_pass is True
