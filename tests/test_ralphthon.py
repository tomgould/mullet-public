"""Tests for mullet.core.orchestration.ralphthon."""

import pytest
from mullet.core.orchestration.ralphthon import (
    RalphthonPRD, RalphthonState, RalphthonStory, RalphthonTask,
    HardeningTask,
)


@pytest.fixture
def prd():
    return RalphthonPRD(
        project="test-project",
        description="Test",
        stories=[
            RalphthonStory(id="US-1", title="Story 1", description="Do stuff", tasks=[
                RalphthonTask(id="T-1", title="Task 1", description="First"),
                RalphthonTask(id="T-2", title="Task 2", description="Second"),
            ]),
            RalphthonStory(id="US-2", title="Story 2", description="More stuff", tasks=[
                RalphthonTask(id="T-3", title="Task 3", description="Third"),
            ]),
        ],
    )


def test_prd_creation(prd):
    assert prd.project == "test-project"
    assert len(prd.stories) == 2


def test_story_progress(prd):
    assert prd.stories[0].progress == 0.0
    prd.stories[0].tasks[0].status = "done"
    assert prd.stories[0].progress == 0.5


def test_state_get_next_task(prd):
    state = RalphthonState(prd)
    t = state.get_next_task()
    assert t.id == "T-1"


def test_state_complete_tasks(prd):
    state = RalphthonState(prd)
    state.complete_task("T-1")
    assert state.tasks_completed == 1
    t = state.get_next_task()
    assert t.id == "T-2"


def test_state_transitions_to_hardening(prd):
    state = RalphthonState(prd)
    for tid in ["T-1", "T-2", "T-3"]:
        state.complete_task(tid)
    state.get_next_task()  # triggers transition
    assert state.phase == "hardening"


def test_state_task_retry(prd):
    state = RalphthonState(prd)
    state.complete_task("T-1", success=False, notes="error")
    t = state._find_task("T-1")
    assert t.retries == 1
    assert t.status == "pending"


def test_state_task_skip_after_max_retries(prd):
    prd.config.max_retries = 2
    state = RalphthonState(prd)
    state.complete_task("T-1", success=False)
    state.complete_task("T-1", success=False)
    t = state._find_task("T-1")
    assert t.status == "skipped"
    assert state.tasks_skipped == 1


def test_hardening_waves(prd):
    state = RalphthonState(prd)
    for tid in ["T-1", "T-2", "T-3"]:
        state.complete_task(tid)
    state.get_next_task()
    
    state.start_hardening_wave([HardeningTask(id="H-1", title="Edge case", description="test")])
    assert state.consecutive_clean_waves == 0
    
    state.start_hardening_wave([])
    state.start_hardening_wave([])
    state.start_hardening_wave([])
    assert state.is_complete


def test_hardening_max_waves(prd):
    prd.config.max_waves = 2
    state = RalphthonState(prd)
    for tid in ["T-1", "T-2", "T-3"]:
        state.complete_task(tid)
    state.get_next_task()
    state.start_hardening_wave([HardeningTask(id="H-1", title="t", description="d")])
    state.start_hardening_wave([HardeningTask(id="H-2", title="t", description="d")])
    state.start_hardening_wave([])  # wave 3 > max_waves=2
    assert state.is_complete


def test_prd_serialization():
    prd = RalphthonPRD(project="p", description="d", stories=[
        RalphthonStory(id="US-1", title="S", description="D", tasks=[
            RalphthonTask(id="T-1", title="T", description="DD"),
        ]),
    ])
    d = prd.to_dict()
    prd2 = RalphthonPRD.from_dict(d)
    assert prd2.project == "p"
    assert len(prd2.stories) == 1
    assert prd2.stories[0].tasks[0].id == "T-1"


def test_state_save_load(tmp_path, prd):
    state = RalphthonState(prd)
    state.complete_task("T-1")
    path = str(tmp_path / "state.json")
    state.save(path)

    state2 = RalphthonState.load(path)
    assert state2.tasks_completed == 1
    assert state2.prd.project == "test-project"
