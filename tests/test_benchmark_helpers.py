"""Tests for the benchmark dataclass and summary."""

import sys
import io

from mullet.cli.benchmark import BenchmarkResult, BENCHMARK_TASKS, print_benchmark_summary


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_name_stored(self):
        r = BenchmarkResult(name="my_task", passed=True)
        assert r.name == "my_task"

    def test_passed_stored(self):
        r = BenchmarkResult(name="test", passed=False)
        assert r.passed is False

    def test_duration_default_zero(self):
        r = BenchmarkResult(name="test", passed=True)
        assert r.duration == 0.0

    def test_tool_calls_default_zero(self):
        r = BenchmarkResult(name="test", passed=True)
        assert r.tool_calls == 0

    def test_tokens_default_zero(self):
        r = BenchmarkResult(name="test", passed=True)
        assert r.tokens == 0

    def test_error_default_empty(self):
        r = BenchmarkResult(name="test", passed=True)
        assert r.error == ""

    def test_details_default_empty(self):
        r = BenchmarkResult(name="test", passed=True)
        assert r.details == ""

    def test_custom_duration(self):
        r = BenchmarkResult(name="test", passed=True, duration=3.14)
        assert r.duration == 3.14

    def test_custom_tokens(self):
        r = BenchmarkResult(name="test", passed=True, tokens=1234)
        assert r.tokens == 1234

    def test_custom_error(self):
        r = BenchmarkResult(name="test", passed=False, error="timeout")
        assert r.error == "timeout"


# ---------------------------------------------------------------------------
# BENCHMARK_TASKS constant
# ---------------------------------------------------------------------------

class TestBenchmarkTasks:
    def test_nonempty(self):
        assert len(BENCHMARK_TASKS) > 0

    def test_each_task_has_name(self):
        for task in BENCHMARK_TASKS:
            assert "name" in task

    def test_each_task_has_prompt(self):
        for task in BENCHMARK_TASKS:
            assert "prompt" in task

    def test_tasks_is_list(self):
        assert isinstance(BENCHMARK_TASKS, list)

    def test_complexity_check_tasks_have_expected_tier(self):
        for task in BENCHMARK_TASKS:
            if task.get("complexity_check"):
                assert "expected_tier" in task


# ---------------------------------------------------------------------------
# print_benchmark_summary (side-effect via stdout capture)
# ---------------------------------------------------------------------------

class TestPrintBenchmarkSummary:
    def _capture(self, results):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        print_benchmark_summary(results)
        sys.stdout = old
        return buf.getvalue()

    def test_shows_passed_count(self):
        results = [
            BenchmarkResult(name="t1", passed=True),
            BenchmarkResult(name="t2", passed=False),
        ]
        output = self._capture(results)
        assert "1/2" in output

    def test_shows_total_time(self):
        results = [
            BenchmarkResult(name="t1", passed=True, duration=2.5),
        ]
        output = self._capture(results)
        assert "2.5" in output

    def test_shows_total_tokens(self):
        results = [BenchmarkResult(name="t1", passed=True, tokens=500)]
        output = self._capture(results)
        assert "500" in output

    def test_pass_icon_shown(self):
        results = [BenchmarkResult(name="t1", passed=True)]
        output = self._capture(results)
        assert "✓" in output

    def test_fail_icon_shown(self):
        results = [BenchmarkResult(name="t1", passed=False)]
        output = self._capture(results)
        assert "✗" in output

    def test_empty_results_no_crash(self):
        self._capture([])

    def test_task_name_in_output(self):
        results = [BenchmarkResult(name="my_special_task", passed=True)]
        output = self._capture(results)
        assert "my_special_task" in output
