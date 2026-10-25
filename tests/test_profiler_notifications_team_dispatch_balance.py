"""Tests for Profiler."""

import time
import pytest

from mullet.core.agent.profiler import Profiler


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
