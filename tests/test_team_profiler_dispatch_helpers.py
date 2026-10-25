"""Tests for Profiler pure helpers."""

import time
import pytest

from mullet.core.agent.profiler import Profiler


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
