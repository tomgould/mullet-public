import pytest
import time
from mullet.core.agent.profiler import Profiler

def test_start_and_stop_timer():
    profiler = Profiler()
    profiler.start_timer("task_a")
    time.sleep(0.01) # Simulate work
    profiler.stop_timer("task_a")
    report = profiler.report()
    assert "task_a" in report
    assert report["task_a"] > 0

def test_restarting_timer():
    profiler = Profiler()
    profiler.start_timer("task_c")
    time.sleep(0.01)
    profiler.stop_timer("task_c")
    time.sleep(0.01)
    profiler.start_timer("task_c")
    time.sleep(0.01)
    profiler.stop_timer("task_c")
    report = profiler.report()
    assert "task_c" in report
    assert report["task_c"] > 0.02 # Sum of two durations

def test_timer_not_started_raises_error_on_stop():
    profiler = Profiler()
    with pytest.raises(ValueError, match="Timer 'non_existent_timer' not started."):
        profiler.stop_timer("non_existent_timer")

def test_timer_already_started_raises_error_on_start():
    profiler = Profiler()
    profiler.start_timer("task_d")
    with pytest.raises(ValueError, match="Timer 'task_d' already started."):
        profiler.start_timer("task_d")

def test_report_with_running_timers():
    profiler = Profiler()
    profiler.start_timer("task_e")
    time.sleep(0.01)
    report = profiler.report()
    assert "task_e" in report
    assert report["task_e"] > 0
    assert not profiler._timers # Should be cleared after report

def test_reset_profiler():
    profiler = Profiler()
    profiler.start_timer("task_f")
    time.sleep(0.01)
    profiler.stop_timer("task_f")
    profiler.reset()
    report = profiler.report()
    assert not report # Report should be empty after reset
    assert not profiler._timers
    assert not profiler._durations

def test_empty_profiler_report():
    profiler = Profiler()
    report = profiler.report()
    assert report == {}

