"""Tests for swarm task deduplication and diversity gate."""

from mullet.core.orchestration.swarm import _dedupe_tasks
from mullet.core.orchestration.swarm_types import SwarmTask


def _make(instruction: str, task_id: str = "t") -> SwarmTask:
    return SwarmTask(id=task_id, instruction=instruction)


def test_dedupe_exact_duplicates():
    tasks = [_make("Summarise X", "a"), _make("Summarise X", "b"), _make("Summarise X", "c")]
    out = _dedupe_tasks(tasks)
    assert len(out) == 1
    assert out[0].id == "a"  # first wins


def test_dedupe_case_and_whitespace_normalized():
    tasks = [
        _make("  Summarise foo  ", "a"),
        _make("summarise FOO", "b"),
        _make("SUMMARISE  foo", "c"),
    ]
    out = _dedupe_tasks(tasks)
    assert len(out) == 1
    assert out[0].id == "a"


def test_dedupe_preserves_unique():
    tasks = [_make("Task A", "a"), _make("Task B", "b"), _make("Task A", "c")]
    out = _dedupe_tasks(tasks)
    assert len(out) == 2
    assert [t.id for t in out] == ["a", "b"]


def test_dedupe_empty_list():
    assert _dedupe_tasks([]) == []


def test_dedupe_all_unique_passthrough():
    tasks = [_make("A", "1"), _make("B", "2"), _make("C", "3")]
    out = _dedupe_tasks(tasks)
    assert len(out) == 3
    assert [t.id for t in out] == ["1", "2", "3"]
