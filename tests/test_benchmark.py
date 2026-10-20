"""Tests for the benchmark system."""

from mullet.cli.benchmark import BENCHMARK_TASKS, BenchmarkResult, run_benchmark, print_benchmark_summary


def test_benchmark_tasks_defined():
    assert len(BENCHMARK_TASKS) >= 6
    for task in BENCHMARK_TASKS:
        assert "name" in task
        assert "prompt" in task


def test_benchmark_result_defaults():
    r = BenchmarkResult(name="test", passed=True)
    assert r.duration == 0.0
    assert r.tokens == 0
    assert r.error == ""


def test_routing_benchmarks_no_provider():
    """Routing benchmarks should work without an LLM provider."""
    results = run_benchmark(verbose=False)
    # At least the routing tests should pass (they don't need a provider)
    routing_results = [r for r in results if "routing" in r.name]
    assert len(routing_results) >= 2
    assert all(r.passed for r in routing_results)


def test_benchmark_task_names_unique():
    names = [t["name"] for t in BENCHMARK_TASKS]
    assert len(names) == len(set(names))


def test_benchmark_has_complexity_and_agent_tasks():
    complexity_tasks = [t for t in BENCHMARK_TASKS if t.get("complexity_check")]
    agent_tasks = [t for t in BENCHMARK_TASKS if not t.get("complexity_check")]
    assert len(complexity_tasks) >= 2
    assert len(agent_tasks) >= 3


def test_print_benchmark_summary_passed_count(capsys):
    results = [
        BenchmarkResult(name="t1", passed=True, duration=1.0, tokens=100),
        BenchmarkResult(name="t2", passed=False, duration=2.0, tokens=200, error="fail"),
    ]
    print_benchmark_summary(results)
    captured = capsys.readouterr().out
    assert "1/2 passed" in captured


def test_print_benchmark_summary_shows_timing(capsys):
    results = [BenchmarkResult(name="t1", passed=True, duration=3.5)]
    print_benchmark_summary(results)
    captured = capsys.readouterr().out
    assert "3.5s" in captured


def test_print_benchmark_summary_empty(capsys):
    print_benchmark_summary([])
    captured = capsys.readouterr().out
    assert "0/0 passed" in captured


def test_benchmark_result_with_details():
    r = BenchmarkResult(name="x", passed=True, details="tier=LOW")
    assert r.details == "tier=LOW"


# ---------------------------------------------------------------------------
# Missing coverage: verbose complexity output, _resolve_named_provider, agent task
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock
from mullet.cli.benchmark import _resolve_named_provider


def test_run_benchmark_verbose_complexity_output(capsys):
    """Lines 158-160: run_benchmark(verbose=True) → PASS/FAIL printed for complexity tasks."""
    results = run_benchmark(verbose=True)
    out = capsys.readouterr().out
    # At least one complexity task line should be printed
    assert "PASS" in out or "FAIL" in out


def test_resolve_named_provider_returns_none_on_exception():
    """Lines 112-114: Exception other than ValueError → log warning → return None."""
    with patch("mullet.cli.main.resolve_provider", side_effect=RuntimeError("no key")):
        result = _resolve_named_provider("kimi", None)
    assert result is None


def test_resolve_named_provider_reraises_value_error():
    """Lines 109-111: ValueError → re-raised to caller."""
    with patch("mullet.cli.main.resolve_provider", side_effect=ValueError("conflict")):
        import pytest
        with pytest.raises(ValueError, match="conflict"):
            _resolve_named_provider("kimi", frozenset(["kimi", "haiku"]))


def test_run_benchmark_verbose_agent_task_exception(capsys):
    """Lines 231-237: AgentRunner.run raises → BenchmarkResult(passed=False, error=...) + verbose print."""
    mock_provider = MagicMock()
    mock_provider.name = "kimi"

    mock_result = MagicMock()
    mock_result.completed = True
    mock_result.final_response = "done"
    mock_result.tool_calls_made = 0
    mock_result.tokens_used.total_tokens = 100

    with patch("mullet.cli.benchmark._resolve_named_provider", return_value=mock_provider), \
         patch("mullet.cli.main.build_tool_registry", return_value=MagicMock()), \
         patch("mullet.core.agent.prompts.build_base_prompt", return_value="sys"), \
         patch("mullet.core.agent.runner.AgentRunner") as mock_runner_cls:
        # First call succeeds (for verbose), second raises (for exception branch)
        mock_runner_cls.return_value.run.side_effect = [mock_result, RuntimeError("agent down")]
        results = run_benchmark(verbose=True, provider_name="kimi")

    out = capsys.readouterr().out
    assert any(r.error for r in results)
    assert "FAIL" in out


def test_run_benchmark_verbose_agent_task_success(capsys):
    """Lines 227-229: AgentRunner.run succeeds with verbose=True → PASS/FAIL line printed."""
    mock_provider = MagicMock()
    mock_provider.name = "kimi"

    mock_result = MagicMock()
    mock_result.completed = True
    mock_result.final_response = "done"
    mock_result.tool_calls_made = 2
    mock_result.tokens_used.total_tokens = 150

    with patch("mullet.cli.benchmark._resolve_named_provider", return_value=mock_provider), \
         patch("mullet.cli.main.build_tool_registry", return_value=MagicMock()), \
         patch("mullet.core.agent.prompts.build_base_prompt", return_value="sys"), \
         patch("mullet.core.agent.runner.AgentRunner") as mock_runner_cls:
        mock_runner_cls.return_value.run.return_value = mock_result
        results = run_benchmark(verbose=True, provider_name="kimi")

    out = capsys.readouterr().out
    # Some task should produce PASS/FAIL verbose output
    assert "PASS" in out or "FAIL" in out


def test_run_benchmark_provider_not_in_allowlist():
    """Lines 171-173: resolved provider not in allowlist → raises ValueError."""
    import pytest
    mock_provider = MagicMock()
    mock_provider.name = "local"

    with patch("mullet.cli.benchmark._resolve_named_provider", return_value=mock_provider):
        with pytest.raises(ValueError, match="not in allowlist"):
            run_benchmark(provider_name="local", allowed_providers=frozenset(["kimi"]))


def test_run_benchmark_no_provider_fills_error_results():
    """Lines 179-189: no provider → BenchmarkResult(passed=False) for all agent tasks."""
    # detect_local_provider is imported inside run_benchmark() so patch the source module
    with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None):
        results = run_benchmark()

    agent_tasks = [t for t in BENCHMARK_TASKS if not t.get("complexity_check")]
    error_results = [r for r in results if not r.passed and r.error]
    assert len(error_results) == len(agent_tasks)
    assert "No LLM provider available" in error_results[0].error


def test_run_benchmark_should_use_tools_zero_calls():
    """Line 215: task with should_use_tools=True but agent returns 0 tool calls → passed=False."""
    mock_provider = MagicMock()
    mock_provider.name = "kimi"

    mock_result = MagicMock()
    mock_result.completed = True
    mock_result.final_response = "done"
    mock_result.tool_calls_made = 0  # <-- zero tool calls despite should_use_tools=True
    mock_result.tokens_used.total_tokens = 50

    with patch("mullet.cli.benchmark._resolve_named_provider", return_value=mock_provider), \
         patch("mullet.cli.main.build_tool_registry", return_value=MagicMock()), \
         patch("mullet.core.agent.prompts.build_base_prompt", return_value="sys"), \
         patch("mullet.core.agent.runner.AgentRunner") as mock_runner_cls:
        mock_runner_cls.return_value.run.return_value = mock_result
        results = run_benchmark(provider_name="kimi")

    tool_tasks = [t for t in BENCHMARK_TASKS if t.get("should_use_tools") and not t.get("complexity_check")]
    assert len(tool_tasks) >= 1
    # All should_use_tools tasks returned 0 tool_calls → all should be failed
    failed = [r for r in results if not r.passed and not r.error]
    assert len(failed) >= 1
