"""Tests for TeamLoadBalancer."""
from mullet.core.orchestration.team_balance import TeamLoadBalancer


def test_assign_to_least_loaded():
    lb = TeamLoadBalancer()
    a1 = lb.assign_task(["a", "b"], "task1")
    assert a1 in ("a", "b")
    a2 = lb.assign_task(["a", "b"], "task2")
    assert a2 != a1  # Should pick the other (less loaded)


def test_assign_empty_agents():
    lb = TeamLoadBalancer()
    assert lb.assign_task([], "task") is None


def test_complete_reduces_load():
    lb = TeamLoadBalancer()
    lb.assign_task(["a"], "t1")
    lb.assign_task(["a"], "t2")
    assert lb.agent_load["a"] == 2
    lb.complete_task("a")
    assert lb.agent_load["a"] == 1


def test_complete_nonexistent():
    lb = TeamLoadBalancer()
    lb.complete_task("x")  # Should not crash
