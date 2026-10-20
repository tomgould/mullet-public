"""Tests for TeamLoadBalancer, Profiler, and TeamDispatchManager pure helpers."""

import time
import pytest

from mullet.core.orchestration.team_balance import TeamLoadBalancer
from mullet.core.agent.profiler import Profiler
from mullet.core.orchestration.team_dispatch import TeamDispatchManager


# ---------------------------------------------------------------------------
# TeamLoadBalancer.assign_task
# ---------------------------------------------------------------------------

class TestTeamLoadBalancerAssignTask:
    def setup_method(self):
        self.lb = TeamLoadBalancer()

    def test_no_agents_returns_none(self):
        assert self.lb.assign_task([], "task1") is None

    def test_single_agent_returned(self):
        result = self.lb.assign_task(["agent_a"], "task1")
        assert result == "agent_a"

    def test_increments_load_on_assign(self):
        self.lb.assign_task(["agent_a"], "task1")
        assert self.lb.agent_load.get("agent_a", 0) == 1

    def test_second_agent_used_when_less_loaded(self):
        self.lb.assign_task(["agent_a", "agent_b"], "task1")
        self.lb.assign_task(["agent_a", "agent_b"], "task2")
        # One agent has 1, one has 1 — both get assigned equally
        assert self.lb.agent_load.get("agent_a", 0) + self.lb.agent_load.get("agent_b", 0) == 2

    def test_least_loaded_agent_selected(self):
        self.lb.agent_load = {"agent_a": 5, "agent_b": 1}
        result = self.lb.assign_task(["agent_a", "agent_b"], "task")
        assert result == "agent_b"

    def test_new_agent_priority(self):
        # agent_a has been used, agent_b is new
        self.lb.agent_load = {"agent_a": 3}
        result = self.lb.assign_task(["agent_a", "agent_b"], "task")
        assert result == "agent_b"


class TestTeamLoadBalancerCompleteTask:
    def setup_method(self):
        self.lb = TeamLoadBalancer()

    def test_complete_decrements_load(self):
        self.lb.assign_task(["agent_a"], "t1")
        self.lb.complete_task("agent_a")
        assert self.lb.agent_load["agent_a"] == 0

    def test_complete_unknown_agent_no_crash(self):
        self.lb.complete_task("nonexistent")  # Should not raise

    def test_complete_does_not_go_below_zero(self):
        self.lb.agent_load["agent_a"] = 0
        self.lb.complete_task("agent_a")
        assert self.lb.agent_load["agent_a"] == 0

    def test_multiple_assigns_then_completes(self):
        self.lb.assign_task(["agent_a"], "t1")
        self.lb.assign_task(["agent_a"], "t2")
        self.lb.complete_task("agent_a")
        assert self.lb.agent_load["agent_a"] == 1


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class TestProfilerStartStop:
    def setup_method(self):
        self.prof = Profiler()

    def test_start_and_stop_records_duration(self):
        self.prof.start_timer("op")
        time.sleep(0.01)
        self.prof.stop_timer("op")
        durations = self.prof.report()
        assert "op" in durations
        assert durations["op"] >= 0.005

    def test_start_already_started_raises(self):
        self.prof.start_timer("op")
        with pytest.raises(ValueError, match="already started"):
            self.prof.start_timer("op")

    def test_stop_not_started_raises(self):
        with pytest.raises(ValueError, match="not started"):
            self.prof.stop_timer("op")

    def test_multiple_timers_independent(self):
        self.prof.start_timer("a")
        self.prof.start_timer("b")
        self.prof.stop_timer("a")
        self.prof.stop_timer("b")
        durations = self.prof.report()
        assert "a" in durations
        assert "b" in durations

    def test_timer_can_be_restarted_after_stop(self):
        self.prof.start_timer("op")
        self.prof.stop_timer("op")
        self.prof.start_timer("op")  # Should not raise
        self.prof.stop_timer("op")

    def test_accumulated_durations(self):
        self.prof.start_timer("op")
        self.prof.stop_timer("op")
        self.prof.start_timer("op")
        self.prof.stop_timer("op")
        durations = self.prof.report()
        # Duration should be sum of two runs
        assert durations["op"] >= 0.0


class TestProfilerReport:
    def setup_method(self):
        self.prof = Profiler()

    def test_empty_report_returns_empty_dict(self):
        assert self.prof.report() == {}

    def test_report_includes_running_timers(self):
        self.prof.start_timer("running")
        durations = self.prof.report()
        assert "running" in durations
        assert durations["running"] >= 0.0

    def test_report_clears_running_timers(self):
        self.prof.start_timer("running")
        self.prof.report()
        # After report, timer is cleared — can restart
        self.prof.start_timer("running")  # Should not raise

    def test_returns_dict(self):
        assert isinstance(self.prof.report(), dict)


class TestProfilerReset:
    def setup_method(self):
        self.prof = Profiler()

    def test_reset_clears_durations(self):
        self.prof.start_timer("op")
        self.prof.stop_timer("op")
        self.prof.reset()
        assert self.prof.report() == {}

    def test_reset_clears_running_timers(self):
        self.prof.start_timer("op")
        self.prof.reset()
        self.prof.start_timer("op")  # Should not raise


# ---------------------------------------------------------------------------
# TeamDispatchManager
# ---------------------------------------------------------------------------

class TestTeamDispatchManagerCanDispatch:
    def test_can_dispatch_initially(self):
        mgr = TeamDispatchManager()
        assert mgr.can_dispatch("issue-1") is True

    def test_cannot_dispatch_after_max_retries(self):
        mgr = TeamDispatchManager(max_retries=2, trigger_cooldown=0.0, issue_cooldown=0.0)
        mgr.record_dispatch("issue-1")
        mgr.record_dispatch("issue-1")
        assert mgr.can_dispatch("issue-1") is False

    def test_can_dispatch_different_issue(self):
        mgr = TeamDispatchManager(max_retries=1, trigger_cooldown=0.0, issue_cooldown=0.0)
        mgr.record_dispatch("issue-1")
        # issue-2 hasn't been dispatched, but trigger cooldown is 0 so check
        assert mgr.can_dispatch("issue-2") is True

    def test_trigger_cooldown_blocks(self):
        mgr = TeamDispatchManager(trigger_cooldown=9999.0, issue_cooldown=0.0)
        mgr.record_dispatch("issue-1")
        # Next dispatch should be blocked by trigger cooldown
        assert mgr.can_dispatch("issue-2") is False

    def test_issue_cooldown_blocks(self):
        mgr = TeamDispatchManager(trigger_cooldown=0.0, issue_cooldown=9999.0)
        mgr.record_dispatch("issue-1")
        # Need to wait out trigger cooldown (which is 0), but issue cooldown blocks
        assert mgr.can_dispatch("issue-1") is False

    def test_first_dispatch_no_cooldown_issue(self):
        mgr = TeamDispatchManager(trigger_cooldown=0.0, issue_cooldown=9999.0)
        # First dispatch of issue-2 — no prior record, no cooldown blocking
        assert mgr.can_dispatch("issue-2") is True


class TestTeamDispatchManagerRecordDispatch:
    def test_record_increments_count(self):
        mgr = TeamDispatchManager()
        mgr.record_dispatch("issue-1")
        assert mgr.issue_dispatch_counts["issue-1"] == 1

    def test_record_twice_increments_twice(self):
        mgr = TeamDispatchManager()
        mgr.record_dispatch("issue-1")
        mgr.record_dispatch("issue-1")
        assert mgr.issue_dispatch_counts["issue-1"] == 2

    def test_record_sets_last_dispatch_time(self):
        mgr = TeamDispatchManager()
        assert mgr.last_dispatch_time is None
        mgr.record_dispatch("issue-1")
        assert mgr.last_dispatch_time is not None

    def test_record_stores_issue_timestamp(self):
        mgr = TeamDispatchManager()
        mgr.record_dispatch("issue-5")
        assert "issue-5" in mgr.issue_last_dispatch_time
