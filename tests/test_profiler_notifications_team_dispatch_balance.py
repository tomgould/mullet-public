"""Tests for Profiler, NotificationManager, TeamDispatchManager, TeamLoadBalancer."""

import time
import pytest

from mullet.core.agent.profiler import Profiler
from mullet.core.orchestration.notifications import Notification, NotificationManager
from mullet.core.orchestration.team_dispatch import TeamDispatchManager
from mullet.core.orchestration.team_balance import TeamLoadBalancer


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class TestProfilerInit:
    def test_timers_empty_initially(self):
        p = Profiler()
        assert p._timers == {}

    def test_durations_empty_initially(self):
        p = Profiler()
        assert p._durations == {}

    def test_report_empty_initially(self):
        p = Profiler()
        assert p.report() == {}


class TestProfilerStartStop:
    def test_start_stop_records_duration(self):
        p = Profiler()
        p.start_timer("task")
        time.sleep(0.01)
        p.stop_timer("task")
        assert "task" in p._durations
        assert p._durations["task"] > 0

    def test_start_twice_raises(self):
        p = Profiler()
        p.start_timer("task")
        with pytest.raises(ValueError):
            p.start_timer("task")

    def test_stop_without_start_raises(self):
        p = Profiler()
        with pytest.raises(ValueError):
            p.stop_timer("task")

    def test_timer_deleted_after_stop(self):
        p = Profiler()
        p.start_timer("task")
        p.stop_timer("task")
        assert "task" not in p._timers

    def test_multiple_timers(self):
        p = Profiler()
        p.start_timer("a")
        p.start_timer("b")
        p.stop_timer("a")
        p.stop_timer("b")
        assert "a" in p._durations
        assert "b" in p._durations

    def test_timer_can_be_reused_after_stop(self):
        p = Profiler()
        p.start_timer("task")
        p.stop_timer("task")
        p.start_timer("task")
        p.stop_timer("task")
        # Duration should have accumulated
        assert p._durations["task"] > 0

    def test_duration_accumulates(self):
        p = Profiler()
        p.start_timer("task")
        p.stop_timer("task")
        first = p._durations["task"]
        p.start_timer("task")
        p.stop_timer("task")
        assert p._durations["task"] > first


class TestProfilerReport:
    def test_report_includes_completed_timers(self):
        p = Profiler()
        p.start_timer("task")
        p.stop_timer("task")
        report = p.report()
        assert "task" in report

    def test_report_includes_running_timers(self):
        p = Profiler()
        p.start_timer("running")
        report = p.report()
        assert "running" in report

    def test_report_clears_running_timers(self):
        p = Profiler()
        p.start_timer("running")
        p.report()
        assert p._timers == {}

    def test_report_returns_dict(self):
        p = Profiler()
        assert isinstance(p.report(), dict)

    def test_duration_is_positive(self):
        p = Profiler()
        p.start_timer("task")
        time.sleep(0.01)
        p.stop_timer("task")
        assert p.report()["task"] > 0


class TestProfilerReset:
    def test_reset_clears_timers(self):
        p = Profiler()
        p.start_timer("task")
        p.reset()
        assert p._timers == {}

    def test_reset_clears_durations(self):
        p = Profiler()
        p.start_timer("task")
        p.stop_timer("task")
        p.reset()
        assert p._durations == {}


# ---------------------------------------------------------------------------
# Notification and NotificationManager
# ---------------------------------------------------------------------------

class TestNotification:
    def test_message_stored(self):
        n = Notification(message="hello")
        assert n.message == "hello"

    def test_default_level_info(self):
        n = Notification(message="test")
        assert n.level == "info"

    def test_default_source_system(self):
        n = Notification(message="test")
        assert n.source == "system"

    def test_timestamp_auto_set(self):
        n = Notification(message="test")
        assert n.timestamp != ""

    def test_custom_level(self):
        n = Notification(message="err", level="error")
        assert n.level == "error"


class TestNotificationManager:
    def test_unread_count_zero_initially(self):
        nm = NotificationManager()
        assert nm.unread_count == 0

    def test_notify_increments_count(self):
        nm = NotificationManager()
        nm.notify("hello")
        assert nm.unread_count == 1

    def test_get_recent_returns_list(self):
        nm = NotificationManager()
        nm.notify("hello")
        assert isinstance(nm.get_recent(), list)

    def test_get_recent_contains_notification(self):
        nm = NotificationManager()
        nm.notify("test message")
        recent = nm.get_recent()
        assert any(n.message == "test message" for n in recent)

    def test_get_recent_limit(self):
        nm = NotificationManager()
        for i in range(20):
            nm.notify(f"message {i}")
        recent = nm.get_recent(5)
        assert len(recent) == 5

    def test_get_recent_returns_last_n(self):
        nm = NotificationManager()
        for i in range(10):
            nm.notify(f"msg{i}")
        recent = nm.get_recent(3)
        assert recent[-1].message == "msg9"

    def test_clear_empties_notifications(self):
        nm = NotificationManager()
        nm.notify("hello")
        nm.clear()
        assert nm.unread_count == 0

    def test_clear_get_recent_empty(self):
        nm = NotificationManager()
        nm.notify("hello")
        nm.clear()
        assert nm.get_recent() == []

    def test_source_stored(self):
        nm = NotificationManager()
        nm.notify("msg", source="mymodule")
        recent = nm.get_recent()
        assert recent[0].source == "mymodule"

    def test_level_stored(self):
        nm = NotificationManager()
        nm.notify("err", level="warning")
        recent = nm.get_recent()
        assert recent[0].level == "warning"


# ---------------------------------------------------------------------------
# TeamDispatchManager
# ---------------------------------------------------------------------------

class TestTeamDispatchManagerInit:
    def test_last_dispatch_none_initially(self):
        d = TeamDispatchManager()
        assert d.last_dispatch_time is None

    def test_dispatch_counts_empty(self):
        d = TeamDispatchManager()
        assert d.issue_dispatch_counts == {}

    def test_can_dispatch_first_time(self):
        d = TeamDispatchManager()
        assert d.can_dispatch("issue-1") is True

    def test_custom_max_retries(self):
        d = TeamDispatchManager(max_retries=5)
        assert d.max_retries == 5


class TestCanDispatch:
    def test_after_max_retries_cannot_dispatch(self):
        d = TeamDispatchManager(max_retries=2, issue_cooldown=0.0, trigger_cooldown=0.0)
        d.record_dispatch("issue-1")
        d.record_dispatch("issue-1")
        assert d.can_dispatch("issue-1") is False

    def test_different_issue_can_dispatch(self):
        d = TeamDispatchManager(max_retries=2, issue_cooldown=0.0, trigger_cooldown=0.0)
        d.record_dispatch("issue-1")
        d.record_dispatch("issue-1")
        assert d.can_dispatch("issue-2") is True

    def test_trigger_cooldown_blocks(self):
        d = TeamDispatchManager(trigger_cooldown=3600.0)
        d.record_dispatch("issue-1")
        assert d.can_dispatch("issue-2") is False

    def test_zero_cooldowns_can_dispatch_repeatedly(self):
        d = TeamDispatchManager(issue_cooldown=0.0, trigger_cooldown=0.0, max_retries=100)
        d.record_dispatch("issue-1")
        assert d.can_dispatch("issue-1") is True


class TestRecordDispatch:
    def test_record_increments_count(self):
        d = TeamDispatchManager()
        d.record_dispatch("issue-1")
        assert d.issue_dispatch_counts["issue-1"] == 1

    def test_record_twice_increments_to_two(self):
        d = TeamDispatchManager()
        d.record_dispatch("issue-1")
        d.record_dispatch("issue-1")
        assert d.issue_dispatch_counts["issue-1"] == 2

    def test_sets_last_dispatch_time(self):
        d = TeamDispatchManager()
        d.record_dispatch("issue-1")
        assert d.last_dispatch_time is not None

    def test_sets_issue_last_dispatch_time(self):
        d = TeamDispatchManager()
        d.record_dispatch("issue-1")
        assert "issue-1" in d.issue_last_dispatch_time


# ---------------------------------------------------------------------------
# TeamLoadBalancer
# ---------------------------------------------------------------------------

class TestTeamLoadBalancerInit:
    def test_agent_load_empty(self):
        lb = TeamLoadBalancer()
        assert lb.agent_load == {}


class TestAssignTask:
    def test_empty_agents_returns_none(self):
        lb = TeamLoadBalancer()
        assert lb.assign_task([], "task") is None

    def test_single_agent_returns_that_agent(self):
        lb = TeamLoadBalancer()
        result = lb.assign_task(["agent1"], "task")
        assert result == "agent1"

    def test_assignment_increments_load(self):
        lb = TeamLoadBalancer()
        lb.assign_task(["agent1"], "task")
        assert lb.agent_load["agent1"] == 1

    def test_least_loaded_agent_chosen(self):
        lb = TeamLoadBalancer()
        lb.agent_load["busy"] = 5
        lb.agent_load["free"] = 0
        result = lb.assign_task(["busy", "free"], "task")
        assert result == "free"

    def test_new_agent_in_result_set(self):
        lb = TeamLoadBalancer()
        lb.agent_load["existing"] = 0
        result = lb.assign_task(["existing", "new_agent"], "task")
        assert result in ("existing", "new_agent")

    def test_multiple_assignments_balanced(self):
        lb = TeamLoadBalancer()
        agents = ["a", "b"]
        lb.assign_task(agents, "t1")
        lb.assign_task(agents, "t2")
        lb.assign_task(agents, "t3")
        lb.assign_task(agents, "t4")
        # Both should have some load
        assert lb.agent_load["a"] > 0
        assert lb.agent_load["b"] > 0


class TestCompleteTask:
    def test_complete_decrements_load(self):
        lb = TeamLoadBalancer()
        lb.assign_task(["agent1"], "task")
        lb.complete_task("agent1")
        assert lb.agent_load["agent1"] == 0

    def test_complete_unknown_agent_no_error(self):
        lb = TeamLoadBalancer()
        lb.complete_task("nonexistent")  # should not raise

    def test_complete_zero_load_no_negative(self):
        lb = TeamLoadBalancer()
        lb.agent_load["agent1"] = 0
        lb.complete_task("agent1")
        assert lb.agent_load["agent1"] == 0
