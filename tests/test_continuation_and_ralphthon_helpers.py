"""Tests for continuation pure helpers and ralphthon dataclass/state helpers."""

import pytest

from mullet.core.agent.continuation import (
    should_continue,
    format_todo_reminder,
    CONTINUATION_REMINDERS,
    _STOP_PHRASES,
    _COMPLETION_PHRASES,
)
from mullet.core.orchestration.ralphthon import (
    RalphthonTask,
    RalphthonStory,
    HardeningTask,
    RalphthonConfig,
    RalphthonPRD,
    RalphthonState,
)


# ---------------------------------------------------------------------------
# CONTINUATION_REMINDERS / constants
# ---------------------------------------------------------------------------

class TestContinuationConstants:
    def test_reminders_nonempty(self):
        assert len(CONTINUATION_REMINDERS) > 0

    def test_reminders_are_strings(self):
        assert all(isinstance(r, str) for r in CONTINUATION_REMINDERS)

    def test_stop_phrases_nonempty(self):
        assert len(_STOP_PHRASES) > 0

    def test_completion_phrases_nonempty(self):
        assert len(_COMPLETION_PHRASES) > 0


# ---------------------------------------------------------------------------
# should_continue
# ---------------------------------------------------------------------------

class TestShouldContinueReturnType:
    def test_returns_tuple(self):
        result = should_continue("some text", 3, [])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_bool(self):
        flag, _ = should_continue("some text", 3, [])
        assert isinstance(flag, bool)


class TestShouldContinueDoNotContinue:
    def test_empty_response_no_continue(self):
        flag, nudge = should_continue("", 0)
        assert flag is False
        assert nudge is None

    def test_completion_phrase_no_continue(self):
        flag, _ = should_continue("all tests pass, moving on", 5, [])
        assert flag is False

    def test_all_items_completed_no_continue(self):
        flag, _ = should_continue("all items completed successfully", 3, [])
        assert flag is False

    def test_no_tools_short_ack_no_continue(self):
        flag, _ = should_continue("okay understood", 0, [{"status": "pending"}])
        assert flag is False

    def test_stop_phrase_no_pending_no_continue(self):
        flag, _ = should_continue("let me know if you need anything", 3, [])
        assert flag is False

    def test_stop_phrase_all_done_todos_no_continue(self):
        todos = [{"status": "done", "content": "task 1"}]
        flag, _ = should_continue("let me know if you need anything", 3, todos)
        assert flag is False


class TestShouldContinueShouldContinue:
    def _pending_todo(self):
        return [{"status": "pending", "content": "task"}]

    def test_stop_phrase_with_pending_continues(self):
        flag, _ = should_continue("let me know if you need anything", 3, self._pending_todo())
        assert flag is True

    def test_nudge_message_returned(self):
        _, nudge = should_continue("i am done, let me know if you need more", 3, self._pending_todo())
        assert nudge is not None
        assert isinstance(nudge, str)

    def test_nudge_contains_pending_count(self):
        todos = [{"status": "pending"}, {"status": "in_progress"}]
        _, nudge = should_continue("all done, let me know", 3, todos)
        assert "2" in nudge

    def test_iteration_rotates_reminder(self):
        todos = self._pending_todo()
        _, nudge0 = should_continue("all done, let me know", 3, todos, iteration=0)
        _, nudge1 = should_continue("all done, let me know", 3, todos, iteration=1)
        assert nudge0 != nudge1

    def test_iteration_wraps_around(self):
        todos = self._pending_todo()
        n = len(CONTINUATION_REMINDERS)
        _, nudge_0 = should_continue("all done, let me know", 3, todos, iteration=0)
        _, nudge_n = should_continue("all done, let me know", 3, todos, iteration=n)
        # Should use the same reminder (wrapped)
        assert nudge_0 == nudge_n

    def test_finished_phrase_triggers(self):
        flag, _ = should_continue("finished, let me know", 3, self._pending_todo())
        assert flag is True

    def test_in_progress_todos_count_as_pending(self):
        todos = [{"status": "in_progress", "content": "still going"}]
        flag, _ = should_continue("i'm done, let me know", 3, todos)
        assert flag is True


# ---------------------------------------------------------------------------
# format_todo_reminder
# ---------------------------------------------------------------------------

class TestFormatTodoReminder:
    def test_empty_list_returns_none(self):
        assert format_todo_reminder([]) is None

    def test_none_returns_none(self):
        assert format_todo_reminder(None) is None

    def test_all_done_returns_none(self):
        todos = [{"status": "done", "id": "1", "content": "task"}]
        assert format_todo_reminder(todos) is None

    def test_pending_returns_string(self):
        todos = [{"status": "pending", "id": "1", "content": "fix bug"}]
        result = format_todo_reminder(todos)
        assert isinstance(result, str)

    def test_pending_shows_count(self):
        todos = [{"status": "pending", "id": "1", "content": "fix bug"}]
        result = format_todo_reminder(todos)
        assert "1" in result

    def test_pending_marker(self):
        todos = [{"status": "pending", "id": "1", "content": "fix bug"}]
        result = format_todo_reminder(todos)
        assert "[ ]" in result

    def test_in_progress_marker(self):
        todos = [{"status": "in_progress", "id": "1", "content": "running tests"}]
        result = format_todo_reminder(todos)
        assert "[>]" in result

    def test_content_in_output(self):
        todos = [{"status": "pending", "id": "1", "content": "fix auth bug"}]
        result = format_todo_reminder(todos)
        assert "fix auth bug" in result

    def test_multiple_active_counted(self):
        todos = [
            {"status": "pending", "id": "1", "content": "task A"},
            {"status": "in_progress", "id": "2", "content": "task B"},
            {"status": "done", "id": "3", "content": "task C"},
        ]
        result = format_todo_reminder(todos)
        assert "2" in result
        assert "task A" in result
        assert "task B" in result
        assert "task C" not in result


# ---------------------------------------------------------------------------
# RalphthonTask
# ---------------------------------------------------------------------------

class TestRalphthonTask:
    def test_defaults(self):
        t = RalphthonTask(id="t1", title="Task", description="Do it")
        assert t.status == "pending"
        assert t.retries == 0
        assert t.notes == ""

    def test_fields_stored(self):
        t = RalphthonTask(id="t1", title="My Task", description="Details")
        assert t.id == "t1"
        assert t.title == "My Task"
        assert t.description == "Details"


# ---------------------------------------------------------------------------
# RalphthonStory
# ---------------------------------------------------------------------------

class TestRalphthonStoryProgress:
    def test_no_tasks_progress_zero(self):
        s = RalphthonStory(id="s1", title="Story", description="desc")
        assert s.progress == 0.0

    def test_half_done_progress_point_five(self):
        s = RalphthonStory(id="s1", title="Story", description="desc", tasks=[
            RalphthonTask("t1", "T1", "d", status="done"),
            RalphthonTask("t2", "T2", "d", status="pending"),
        ])
        assert s.progress == pytest.approx(0.5)

    def test_all_done_progress_one(self):
        s = RalphthonStory(id="s1", title="Story", description="desc", tasks=[
            RalphthonTask("t1", "T1", "d", status="done"),
            RalphthonTask("t2", "T2", "d", status="done"),
        ])
        assert s.progress == pytest.approx(1.0)

    def test_skipped_counts_as_done(self):
        s = RalphthonStory(id="s1", title="S", description="d", tasks=[
            RalphthonTask("t1", "T1", "d", status="skipped"),
        ])
        assert s.progress == pytest.approx(1.0)

    def test_mixed_statuses(self):
        s = RalphthonStory(id="s1", title="S", description="d", tasks=[
            RalphthonTask("t1", "T1", "d", status="done"),
            RalphthonTask("t2", "T2", "d", status="skipped"),
            RalphthonTask("t3", "T3", "d", status="pending"),
            RalphthonTask("t4", "T4", "d", status="failed"),
        ])
        assert s.progress == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RalphthonConfig
# ---------------------------------------------------------------------------

class TestRalphthonConfig:
    def test_default_max_waves(self):
        assert RalphthonConfig().max_waves == 10

    def test_default_clean_waves(self):
        assert RalphthonConfig().clean_waves_for_termination == 3

    def test_default_max_retries(self):
        assert RalphthonConfig().max_retries == 3


# ---------------------------------------------------------------------------
# RalphthonPRD
# ---------------------------------------------------------------------------

class TestRalphthonPRDToDict:
    def test_returns_dict(self):
        prd = RalphthonPRD(project="P", description="D")
        assert isinstance(prd.to_dict(), dict)

    def test_project_in_dict(self):
        prd = RalphthonPRD(project="MyProject", description="D")
        assert prd.to_dict()["project"] == "MyProject"

    def test_empty_stories_in_dict(self):
        prd = RalphthonPRD(project="P", description="D")
        assert prd.to_dict()["stories"] == []


class TestRalphthonPRDFromDict:
    def test_roundtrip_project(self):
        prd = RalphthonPRD(project="MyApp", description="A description")
        restored = RalphthonPRD.from_dict(prd.to_dict())
        assert restored.project == "MyApp"

    def test_roundtrip_description(self):
        prd = RalphthonPRD(project="P", description="Test description")
        restored = RalphthonPRD.from_dict(prd.to_dict())
        assert restored.description == "Test description"

    def test_roundtrip_config(self):
        prd = RalphthonPRD(project="P", description="D", config=RalphthonConfig(max_waves=5))
        restored = RalphthonPRD.from_dict(prd.to_dict())
        assert restored.config.max_waves == 5

    def test_roundtrip_stories(self):
        prd = RalphthonPRD(
            project="P", description="D",
            stories=[RalphthonStory("s1", "Story 1", "desc")]
        )
        restored = RalphthonPRD.from_dict(prd.to_dict())
        assert len(restored.stories) == 1
        assert restored.stories[0].id == "s1"

    def test_roundtrip_tasks_in_story(self):
        story = RalphthonStory("s1", "S1", "desc", tasks=[
            RalphthonTask("t1", "Task 1", "do it")
        ])
        prd = RalphthonPRD(project="P", description="D", stories=[story])
        restored = RalphthonPRD.from_dict(prd.to_dict())
        assert restored.stories[0].tasks[0].id == "t1"


# ---------------------------------------------------------------------------
# RalphthonState — basic
# ---------------------------------------------------------------------------

def _make_state_with_tasks():
    prd = RalphthonPRD(project="P", description="D", stories=[
        RalphthonStory("s1", "Story 1", "desc", tasks=[
            RalphthonTask("t1", "Task 1", "do it"),
            RalphthonTask("t2", "Task 2", "also do it"),
        ])
    ])
    return RalphthonState(prd)


class TestRalphthonStateInit:
    def test_initial_phase_execution(self):
        state = _make_state_with_tasks()
        assert state.phase == "execution"

    def test_not_complete_initially(self):
        state = _make_state_with_tasks()
        assert state.is_complete is False

    def test_started_at_set(self):
        state = _make_state_with_tasks()
        assert state.started_at != ""


class TestRalphthonStateGetNextTask:
    def test_returns_first_pending(self):
        state = _make_state_with_tasks()
        task = state.get_next_task()
        assert task.id == "t1"

    def test_returns_second_after_first_done(self):
        state = _make_state_with_tasks()
        state.complete_task("t1", success=True)
        task = state.get_next_task()
        assert task.id == "t2"

    def test_transitions_to_hardening_when_all_done(self):
        state = _make_state_with_tasks()
        state.complete_task("t1", success=True)
        state.complete_task("t2", success=True)
        state.get_next_task()
        assert state.phase == "hardening"

    def test_returns_none_when_no_hardening_tasks(self):
        state = _make_state_with_tasks()
        state.complete_task("t1", success=True)
        state.complete_task("t2", success=True)
        task = state.get_next_task()
        assert task is None


class TestRalphthonStateCompleteTask:
    def test_complete_success_increments_counter(self):
        state = _make_state_with_tasks()
        state.complete_task("t1", success=True)
        assert state.tasks_completed == 1

    def test_complete_success_marks_done(self):
        state = _make_state_with_tasks()
        state.complete_task("t1", success=True)
        assert state.prd.stories[0].tasks[0].status == "done"

    def test_fail_once_still_pending(self):
        state = _make_state_with_tasks()
        state.complete_task("t1", success=False)
        assert state.prd.stories[0].tasks[0].status == "pending"

    def test_fail_increments_retries(self):
        state = _make_state_with_tasks()
        state.complete_task("t1", success=False)
        assert state.prd.stories[0].tasks[0].retries == 1

    def test_fail_max_retries_skips(self):
        state = _make_state_with_tasks()
        for _ in range(3):
            state.complete_task("t1", success=False)
        assert state.prd.stories[0].tasks[0].status == "skipped"

    def test_skip_increments_skipped_counter(self):
        state = _make_state_with_tasks()
        for _ in range(3):
            state.complete_task("t1", success=False)
        assert state.tasks_skipped == 1

    def test_nonexistent_task_no_error(self):
        state = _make_state_with_tasks()
        state.complete_task("nonexistent", success=True)  # should not raise


# ---------------------------------------------------------------------------
# RalphthonState — hardening waves
# ---------------------------------------------------------------------------

class TestRalphthonStateHardeningWaves:
    def _hardening_state(self):
        state = _make_state_with_tasks()
        state.phase = "hardening"
        return state

    def test_first_wave_number_is_one(self):
        state = self._hardening_state()
        wave = state.start_hardening_wave([HardeningTask("h1", "H1", "desc")])
        assert wave == 1

    def test_wave_number_increments(self):
        state = self._hardening_state()
        state.start_hardening_wave([HardeningTask("h1", "H1", "desc")])
        wave = state.start_hardening_wave([HardeningTask("h2", "H2", "desc")])
        assert wave == 2

    def test_clean_wave_increments_counter(self):
        state = self._hardening_state()
        state.start_hardening_wave([])
        assert state.consecutive_clean_waves == 1

    def test_dirty_wave_resets_counter(self):
        state = self._hardening_state()
        state.start_hardening_wave([])
        state.start_hardening_wave([HardeningTask("h1", "H1", "d")])
        assert state.consecutive_clean_waves == 0

    def test_three_clean_waves_completes(self):
        state = self._hardening_state()
        state.start_hardening_wave([])
        state.start_hardening_wave([])
        state.start_hardening_wave([])
        assert state.phase == "complete"
        assert state.is_complete is True

    def test_max_waves_exceeded_completes(self):
        state = self._hardening_state()
        state.prd.config.max_waves = 2
        state.current_wave = 2
        state.start_hardening_wave([HardeningTask("h1", "H1", "d")])
        assert state.phase == "complete"

    def test_new_tasks_added_to_prd(self):
        state = self._hardening_state()
        state.start_hardening_wave([HardeningTask("h1", "H1", "desc")])
        assert len(state.prd.hardening) == 1


# ---------------------------------------------------------------------------
# RalphthonState.is_complete
# ---------------------------------------------------------------------------

class TestRalphthonStateIsComplete:
    def test_execution_phase_not_complete(self):
        state = _make_state_with_tasks()
        state.phase = "execution"
        assert state.is_complete is False

    def test_hardening_phase_not_complete(self):
        state = _make_state_with_tasks()
        state.phase = "hardening"
        assert state.is_complete is False

    def test_complete_phase_is_complete(self):
        state = _make_state_with_tasks()
        state.phase = "complete"
        assert state.is_complete is True

    def test_failed_phase_is_complete(self):
        state = _make_state_with_tasks()
        state.phase = "failed"
        assert state.is_complete is True


# ---------------------------------------------------------------------------
# RalphthonState — to_dict / save / load
# ---------------------------------------------------------------------------

class TestRalphthonStateSaveLoad:
    def _make_prd(self):
        return RalphthonPRD(project="TestProject", description="Test desc", stories=[
            RalphthonStory("s1", "Story 1", "desc", tasks=[
                RalphthonTask("t1", "Task 1", "do it")
            ])
        ])

    def test_to_dict_returns_dict(self):
        state = RalphthonState(self._make_prd())
        assert isinstance(state.to_dict(), dict)

    def test_to_dict_has_phase(self):
        state = RalphthonState(self._make_prd())
        assert "phase" in state.to_dict()

    def test_to_dict_has_prd(self):
        state = RalphthonState(self._make_prd())
        assert "prd" in state.to_dict()

    def test_save_creates_file(self, tmp_path):
        state = RalphthonState(self._make_prd())
        path = str(tmp_path / "state.json")
        state.save(path)
        assert (tmp_path / "state.json").exists()

    def test_load_project_preserved(self, tmp_path):
        state = RalphthonState(self._make_prd())
        path = str(tmp_path / "state.json")
        state.save(path)
        loaded = RalphthonState.load(path)
        assert loaded.prd.project == "TestProject"

    def test_load_story_count_preserved(self, tmp_path):
        state = RalphthonState(self._make_prd())
        path = str(tmp_path / "state.json")
        state.save(path)
        loaded = RalphthonState.load(path)
        assert len(loaded.prd.stories) == 1

    def test_load_phase_preserved(self, tmp_path):
        state = RalphthonState(self._make_prd())
        state.phase = "hardening"
        path = str(tmp_path / "state.json")
        state.save(path)
        loaded = RalphthonState.load(path)
        assert loaded.phase == "hardening"

    def test_load_tasks_completed_preserved(self, tmp_path):
        state = RalphthonState(self._make_prd())
        state.tasks_completed = 7
        path = str(tmp_path / "state.json")
        state.save(path)
        loaded = RalphthonState.load(path)
        assert loaded.tasks_completed == 7
