"""Tests for orchestration mode definitions and selection."""

from mullet.core.orchestration.modes import Mode, MODE_HIERARCHY, MODE_TYPES, select_mode


def test_mode_enum():
    assert Mode.STANDARD.value == "standard"
    assert Mode.ULTRAWORK.value == "ultrawork"
    assert Mode.RALPH.value == "ralph"
    assert Mode.AUTOPILOT.value == "autopilot"
    assert Mode.TEAM.value == "team"


def test_mode_hierarchy():
    assert Mode.RALPH in MODE_HIERARCHY[Mode.AUTOPILOT]
    assert Mode.ULTRAWORK in MODE_HIERARCHY[Mode.AUTOPILOT]
    assert Mode.ULTRAWORK in MODE_HIERARCHY[Mode.RALPH]
    assert MODE_HIERARCHY[Mode.STANDARD] == []


def test_mode_types():
    assert MODE_TYPES[Mode.AUTOPILOT] == "standalone"
    assert MODE_TYPES[Mode.TEAM] == "standalone"
    assert MODE_TYPES[Mode.RALPH] == "wrapper"
    assert MODE_TYPES[Mode.ULTRAWORK] == "component"


def test_select_autopilot():
    assert select_mode("run autopilot on this") == Mode.AUTOPILOT
    assert select_mode("do this autonomously") == Mode.AUTOPILOT


def test_select_ralph():
    assert select_mode("use ralph mode") == Mode.RALPH
    assert select_mode("don't stop until it's done") == Mode.RALPH


def test_select_ultrawork():
    assert select_mode("run in parallel") == Mode.ULTRAWORK


def test_select_standard():
    assert select_mode("fix the bug") == Mode.STANDARD
    assert select_mode("what is this?") == Mode.STANDARD
