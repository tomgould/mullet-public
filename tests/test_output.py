"""Tests for execution tracking and output formatting."""

from mullet.core.agent.output import ExecutionSummary, ExecutionTracker
from mullet.core.agent.types import AgentResult, TokenUsage


def test_execution_summary_defaults():
    s = ExecutionSummary(task="test")
    assert s.task == "test"
    assert s.duration_seconds == 0
    assert s.completed is False


def test_summary_verbose():
    s = ExecutionSummary(
        task="test task",
        provider="local",
        model="mistral",
        tier="LOW",
        tool_calls=["terminal", "read_file", "terminal"],
        duration_seconds=5.2,
        tokens=TokenUsage(total_tokens=1500),
        iterations=3,
        completed=True,
    )
    output = s.format_verbose()
    assert "local" in output
    assert "LOW" in output
    assert "3" in output
    assert "1,500" in output or "1500" in output
    assert "5.2" in output


def test_summary_json():
    s = ExecutionSummary(
        task="test",
        provider="local",
        tool_calls=["terminal", "terminal"],
        tokens=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        completed=True,
    )
    data = s.format_json()
    assert data["task"] == "test"
    assert data["completed"] is True
    assert data["tokens"]["total"] == 150
    assert data["unique_tools"] == ["terminal"]
    assert len(data["tool_calls"]) == 2


def test_execution_tracker():
    tracker = ExecutionTracker(task="build feature")
    tracker.on_tool_call("terminal", "ls")
    tracker.on_tool_call("read_file", "main.py")

    result = AgentResult(
        completed=True,
        final_response="Done",
        tool_calls_made=2,
        iterations_used=3,
        tokens_used=TokenUsage(total_tokens=500),
        provider_used="local",
    )

    summary = tracker.build_summary(result)
    assert summary.task == "build feature"
    assert len(summary.tool_calls) == 2
    assert summary.duration_seconds >= 0


def test_format_json_rounding():
    s = ExecutionSummary(
        task="x",
        duration_seconds=1.23456789,
        cost_estimate=0.00123456789,
    )
    data = s.format_json()
    assert data["duration_seconds"] == 1.23
    assert data["cost_estimate"] == round(0.00123456789, 6)


def test_format_json_unique_tools_deduped():
    s = ExecutionSummary(
        task="x",
        tool_calls=["Bash", "Read", "Bash", "Bash", "Write"],
    )
    data = s.format_json()
    assert set(data["unique_tools"]) == {"Bash", "Read", "Write"}


def test_format_verbose_cost_omitted_when_zero():
    s = ExecutionSummary(task="x", provider="local", cost_estimate=0.0)
    out = s.format_verbose()
    assert "cost" not in out.lower()


def test_format_verbose_cost_shown_when_positive():
    s = ExecutionSummary(task="x", provider="kimi", cost_estimate=0.0042)
    out = s.format_verbose()
    assert "0.0042" in out


def test_format_json_tokens_structure():
    s = ExecutionSummary(
        task="x",
        tokens=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
    )
    data = s.format_json()
    assert data["tokens"]["input"] == 100
    assert data["tokens"]["output"] == 50
    assert data["tokens"]["total"] == 150


def test_execution_tracker_no_tool_calls():
    tracker = ExecutionTracker(task="simple")
    result = AgentResult(
        completed=False,
        final_response="",
        tool_calls_made=0,
        iterations_used=1,
        tokens_used=TokenUsage(),
        provider_used="local",
    )
    summary = tracker.build_summary(result)
    assert summary.tool_calls == []
    assert summary.completed is False
