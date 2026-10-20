"""Tests for mullet.core.orchestration.scheduler — interval-based task scheduling."""

import pytest
from datetime import datetime, timedelta
from mullet.core.orchestration.scheduler import TaskScheduler, parse_interval


def test_parse_interval():
    assert parse_interval("5m") == 300
    assert parse_interval("1h") == 3600
    assert parse_interval("30s") == 30
    assert parse_interval("2d") == 172800


def test_parse_interval_invalid():
    with pytest.raises(ValueError):
        parse_interval("bad")


@pytest.fixture
def sched(tmp_path):
    return TaskScheduler(str(tmp_path / "sched.json"))


def test_add_task(sched):
    t = sched.add("test-job", "10m", "echo hello")
    assert t.name == "test-job"
    assert t.schedule == "10m"
    assert t.command == "echo hello"
    assert t.enabled is True
    assert t.run_count == 0
    assert t.next_run is not None


def test_add_duplicate(sched):
    sched.add("dup", "1m", "cmd")
    with pytest.raises(ValueError, match="already exists"):
        sched.add("dup", "2m", "cmd2")


def test_remove(sched):
    sched.add("rm-me", "1m", "cmd")
    assert sched.remove("rm-me") is True
    assert sched.remove("rm-me") is False
    assert len(sched.list_tasks()) == 0


def test_pause_resume(sched):
    sched.add("pr", "1m", "cmd")
    sched.pause("pr")
    assert sched.list_tasks()[0].enabled is False
    sched.resume("pr")
    assert sched.list_tasks()[0].enabled is True


def test_list_tasks(sched):
    sched.add("a", "1m", "cmd1")
    sched.add("b", "2m", "cmd2")
    assert len(sched.list_tasks()) == 2


def test_mark_run(sched):
    sched.add("counter", "1h", "cmd")
    sched.mark_run("counter")
    t = sched.list_tasks()[0]
    assert t.run_count == 1
    assert t.last_run is not None


def test_get_due_tasks(sched):
    t = sched.add("due-soon", "1s", "cmd")
    # Force next_run to be in the past
    t.next_run = (datetime.now() - timedelta(seconds=5)).isoformat()
    due = sched.get_due_tasks()
    assert len(due) == 1
    assert due[0].name == "due-soon"


def test_paused_not_due(sched):
    t = sched.add("paused", "1s", "cmd")
    t.next_run = (datetime.now() - timedelta(seconds=5)).isoformat()
    sched.pause("paused")
    assert len(sched.get_due_tasks()) == 0


def test_persistence(tmp_path):
    path = str(tmp_path / "persist.json")
    s1 = TaskScheduler(path)
    s1.add("persisted", "5m", "cmd")
    s1.mark_run("persisted")

    s2 = TaskScheduler(path)
    tasks = s2.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].name == "persisted"
    assert tasks[0].run_count == 1
