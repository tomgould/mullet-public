"""Tests for mullet.core.orchestration.ralphthon — dataclasses and state machine."""

from pathlib import Path

from mullet.core.orchestration.ralphthon import (
    RalphthonTask,
    RalphthonStory,
    HardeningTask,
    RalphthonConfig,
    RalphthonPRD,
    RalphthonState,
)


def make_task(id="t1", title="Task", desc="Do something", status="pending"):
    return RalphthonTask(id=id, title=title, description=desc, status=status)


def make_story(id="s1", tasks=None):
    return RalphthonStory(
        id=id, title="Story", description="Story desc",
        tasks=tasks or [],
    )


def make_prd(stories=None, hardening=None):
    return RalphthonPRD(
        project="Test Project",
        description="Test description",
        stories=stories or [],
        hardening=hardening or [],
    )


def make_state(stories=None, hardening=None):
    prd = make_prd(stories=stories, hardening=hardening)
    return RalphthonState(prd)


# ---------------------------------------------------------------------------
# RalphthonTask
# ---------------------------------------------------------------------------

class TestRalphthonTask:
    def test_defaults(self):
        t = RalphthonTask(id="t1", title="Title", description="Desc")
        assert t.status == "pending"
        assert t.retries == 0
        assert t.notes == ""

    def test_custom_status(self):
        t = RalphthonTask(id="t1", title="T", description="D", status="done")
        assert t.status == "done"


# ---------------------------------------------------------------------------
# RalphthonStory.progress
# ---------------------------------------------------------------------------

class TestRalphthonStoryProgress:
    def test_no_tasks_is_zero(self):
        story = make_story()
        assert story.progress == 0.0

    def test_all_done_is_one(self):
        tasks = [make_task(id=f"t{i}", status="done") for i in range(3)]
        story = make_story(tasks=tasks)
        assert story.progress == 1.0

    def test_all_pending_is_zero(self):
        tasks = [make_task(id=f"t{i}", status="pending") for i in range(3)]
        story = make_story(tasks=tasks)
        assert story.progress == 0.0

    def test_half_done(self):
        tasks = [
            make_task("t1", status="done"),
            make_task("t2", status="pending"),
        ]
        story = make_story(tasks=tasks)
        assert story.progress == 0.5

    def test_skipped_counts_as_done(self):
        tasks = [make_task("t1", status="skipped")]
        story = make_story(tasks=tasks)
        assert story.progress == 1.0

    def test_failed_does_not_count(self):
        tasks = [make_task("t1", status="failed")]
        story = make_story(tasks=tasks)
        assert story.progress == 0.0


# ---------------------------------------------------------------------------
# RalphthonConfig
# ---------------------------------------------------------------------------

class TestRalphthonConfig:
    def test_defaults(self):
        c = RalphthonConfig()
        assert c.max_waves == 10
        assert c.clean_waves_for_termination == 3
        assert c.max_retries == 3


# ---------------------------------------------------------------------------
# RalphthonPRD.to_dict / from_dict
# ---------------------------------------------------------------------------

class TestRalphthonPRDRoundTrip:
    def test_to_dict_has_project(self):
        prd = make_prd()
        assert prd.to_dict()["project"] == "Test Project"

    def test_from_dict_restores_project(self):
        prd = make_prd()
        prd2 = RalphthonPRD.from_dict(prd.to_dict())
        assert prd2.project == "Test Project"

    def test_round_trip_with_story_and_task(self):
        task = make_task()
        story = make_story(tasks=[task])
        prd = make_prd(stories=[story])
        prd2 = RalphthonPRD.from_dict(prd.to_dict())
        assert len(prd2.stories) == 1
        assert prd2.stories[0].tasks[0].id == "t1"

    def test_round_trip_with_hardening(self):
        ht = HardeningTask(id="h1", title="Harden", description="Do hardening")
        prd = make_prd(hardening=[ht])
        prd2 = RalphthonPRD.from_dict(prd.to_dict())
        assert len(prd2.hardening) == 1
        assert prd2.hardening[0].id == "h1"

    def test_config_round_tripped(self):
        prd = make_prd()
        prd.config.max_waves = 7
        prd2 = RalphthonPRD.from_dict(prd.to_dict())
        assert prd2.config.max_waves == 7


# ---------------------------------------------------------------------------
# RalphthonState — initial state
# ---------------------------------------------------------------------------

class TestRalphthonStateInit:
    def test_initial_phase_execution(self):
        state = make_state()
        assert state.phase == "execution"

    def test_initial_wave_zero(self):
        state = make_state()
        assert state.current_wave == 0

    def test_not_complete_initially(self):
        state = make_state()
        assert state.is_complete is False


# ---------------------------------------------------------------------------
# RalphthonState.get_next_task
# ---------------------------------------------------------------------------

class TestGetNextTask:
    def test_no_stories_returns_none_after_transition(self):
        state = make_state()
        task = state.get_next_task()
        assert task is None

    def test_returns_pending_execution_task(self):
        task = make_task("t1", status="pending")
        story = make_story(tasks=[task])
        state = make_state(stories=[story])
        result = state.get_next_task()
        assert result is not None
        assert result.id == "t1"

    def test_skips_done_tasks(self):
        t1 = make_task("t1", status="done")
        t2 = make_task("t2", status="pending")
        story = make_story(tasks=[t1, t2])
        state = make_state(stories=[story])
        result = state.get_next_task()
        assert result.id == "t2"

    def test_transitions_to_hardening_when_execution_done(self):
        task = make_task("t1", status="done")
        story = make_story(tasks=[task])
        state = make_state(stories=[story])
        state.get_next_task()  # Triggers transition
        assert state.phase == "hardening"

    def test_hardening_task_returned_in_hardening_phase(self):
        ht = HardeningTask(id="h1", title="H", description="D", wave=1)
        task = make_task("t1", status="done")
        story = make_story(tasks=[task])
        state = make_state(stories=[story], hardening=[ht])
        state.get_next_task()  # Transition to hardening
        state.current_wave = 1
        result = state.get_next_task()
        assert result is not None
        assert result.id == "h1"


# ---------------------------------------------------------------------------
# RalphthonState.complete_task
# ---------------------------------------------------------------------------

class TestCompleteTask:
    def test_success_marks_done(self):
        task = make_task("t1")
        story = make_story(tasks=[task])
        state = make_state(stories=[story])
        state.complete_task("t1", success=True)
        assert task.status == "done"

    def test_success_increments_completed(self):
        task = make_task("t1")
        story = make_story(tasks=[task])
        state = make_state(stories=[story])
        state.complete_task("t1", success=True)
        assert state.tasks_completed == 1

    def test_failure_increments_retries(self):
        task = make_task("t1")
        story = make_story(tasks=[task])
        state = make_state(stories=[story])
        state.complete_task("t1", success=False)
        assert task.retries == 1

    def test_max_retries_skips_task(self):
        task = make_task("t1")
        task.retries = 2
        story = make_story(tasks=[task])
        prd = make_prd(stories=[story])
        prd.config.max_retries = 3
        state = RalphthonState(prd)
        state.complete_task("t1", success=False)
        assert task.status == "skipped"
        assert state.tasks_skipped == 1

    def test_unknown_task_id_no_crash(self):
        state = make_state()
        state.complete_task("nonexistent", success=True)  # Should not raise


# ---------------------------------------------------------------------------
# RalphthonState.start_hardening_wave
# ---------------------------------------------------------------------------

class TestStartHardeningWave:
    def test_increments_wave(self):
        state = make_state()
        state.start_hardening_wave([])
        assert state.current_wave == 1

    def test_clean_wave_increments_consecutive(self):
        state = make_state()
        state.start_hardening_wave([])
        assert state.consecutive_clean_waves == 1

    def test_new_tasks_resets_consecutive(self):
        state = make_state()
        state.consecutive_clean_waves = 2
        ht = HardeningTask(id="h1", title="H", description="D")
        state.start_hardening_wave([ht])
        assert state.consecutive_clean_waves == 0

    def test_enough_clean_waves_completes(self):
        state = make_state()
        state.prd.config.clean_waves_for_termination = 2
        state.start_hardening_wave([])
        state.start_hardening_wave([])
        assert state.phase == "complete"
        assert state.is_complete is True

    def test_max_waves_completes(self):
        state = make_state()
        state.prd.config.max_waves = 2
        state.current_wave = 2
        state.start_hardening_wave([])
        assert state.phase == "complete"


# ---------------------------------------------------------------------------
# RalphthonState save / load
# ---------------------------------------------------------------------------

class TestRalphthonStateSaveLoad:
    def test_save_creates_file(self, tmp_path):
        state = make_state()
        path = str(tmp_path / "state.json")
        state.save(path)
        assert Path(path).exists()

    def test_load_restores_phase(self, tmp_path):
        state = make_state()
        state.phase = "hardening"
        path = str(tmp_path / "state.json")
        state.save(path)
        loaded = RalphthonState.load(path)
        assert loaded.phase == "hardening"

    def test_load_restores_counters(self, tmp_path):
        state = make_state()
        state.tasks_completed = 5
        state.tasks_skipped = 2
        path = str(tmp_path / "state.json")
        state.save(path)
        loaded = RalphthonState.load(path)
        assert loaded.tasks_completed == 5
        assert loaded.tasks_skipped == 2
