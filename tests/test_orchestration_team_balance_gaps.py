"""Coverage-gap tests for mullet.core.orchestration.team_balance."""

from mullet.core.orchestration.team_balance import TeamLoadBalancer


def test_new_agent_initialized_and_considered():
    """New agents are initialized to load 0 and considered for assignment."""
    lb = TeamLoadBalancer()
    lb.assign_task(["a", "b"], "t1")  # a picked (both at 0, a first)
    lb.assign_task(["a", "b", "c"], "t2")  # b and c at 0, b is first min
    assert lb.agent_load["a"] == 1
    assert lb.agent_load["b"] == 1
    assert lb.agent_load["c"] == 0  # initialized but not chosen


def test_assign_least_loaded_from_available_only():
    """Ensure min only considers agents in the current agents list."""
    lb = TeamLoadBalancer()
    lb.assign_task(["a", "b"], "t1")  # a picked
    lb.assign_task(["a", "b"], "t2")  # b picked (least loaded)
    lb.assign_task(["a"], "t3")  # only a available, must pick a
    assert lb.agent_load["a"] == 2
    assert lb.agent_load["b"] == 1
