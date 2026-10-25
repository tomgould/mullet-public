"""Tests for one-way doors in the ladder: the planner can raise a step's risk, never lower it."""

from mullet.jobs import ladder as lad
from mullet.jobs.ladder import LadderConfig
from mullet.jobs.types import NodeSpec

CONFIG = LadderConfig()
CHECKABLE = NodeSpec(id="d", goal="add a retry helper", kind="design", scope=["app/retry.py", "tests/test_retry.py"],
                     acceptance=["python3 -m pytest tests/test_retry.py"])


def test_a_one_way_door_is_risky_without_any_risky_word():
    assert not lad.is_risky(NodeSpec(id="x", goal="rename the config key"), CONFIG)
    assert lad.is_risky(NodeSpec(id="x", goal="rename the config key", door="one-way"), CONFIG)


def test_a_two_way_door_never_lowers_the_keyword_floor():
    assert lad.is_risky(NodeSpec(id="x", goal="delete old clones", door="two-way"), CONFIG)


def test_a_one_way_step_is_claude_first_whatever_its_kind():
    for spec in (NodeSpec(id="m", goal="rename a helper", kind="mechanical", door="one-way"),
                 CHECKABLE.model_copy(update={"door": "one-way"})):
        group, why = lad.step_group(spec, CONFIG)
        assert group == lad.CLAUDE_FIRST and "one-way" in why


def test_two_way_steps_keep_their_groups():
    assert lad.step_group(NodeSpec(id="m", goal="rename a helper", kind="mechanical"), CONFIG)[0] == lad.MECHANICAL
    assert lad.step_group(CHECKABLE, CONFIG)[0] == lad.CHECKABLE
