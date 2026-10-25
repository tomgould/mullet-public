"""The job report's ladder summary keeps unverified steps out of the passed count."""

from __future__ import annotations

from types import SimpleNamespace

from mullet.jobs.ladder import LadderConfig
from mullet.jobs.report import ladder_summary
from mullet.jobs.types import AttemptOutcome, NodeStatus


def _node(node_id: str, status: NodeStatus) -> SimpleNamespace:
    return SimpleNamespace(node_id=node_id, parent_id=None, status=status)


def _attempt(node_id: str, outcome: AttemptOutcome, rung: str = "kimi") -> SimpleNamespace:
    return SimpleNamespace(node_id=node_id, outcome=outcome, usage={"rung": rung})


def test_unverified_steps_are_not_counted_as_passed() -> None:
    nodes = [_node("a", NodeStatus.PASSED), _node("b", NodeStatus.UNVERIFIED), _node("c", NodeStatus.UNVERIFIED)]
    attempts = [_attempt("a", AttemptOutcome.PASSED), _attempt("b", AttemptOutcome.UNVERIFIED),
                _attempt("c", AttemptOutcome.UNVERIFIED)]
    summary = ladder_summary(nodes, attempts, LadderConfig())
    assert summary.startswith("1 of 3 steps passed below Claude, 2 unverified; ")


def test_summary_without_unverified_steps_is_unchanged() -> None:
    nodes = [_node("a", NodeStatus.PASSED)]
    summary = ladder_summary(nodes, [_attempt("a", AttemptOutcome.PASSED)], LadderConfig())
    assert summary.startswith("1 of 1 steps passed below Claude; ")
