"""Pure unit tests for agent errors and output modules.

Covers:
- mullet/core/agent/errors.py      -> format_error
- mullet/core/agent/output.py       -> ExecutionTracker, ExecutionSummary

All tests are isolated and have no external dependencies.
"""

from unittest.mock import patch

from mullet.core.agent.errors import format_error, ERROR_SUGGESTIONS
from mullet.core.agent.output import ExecutionTracker, ExecutionSummary
from mullet.core.agent.types import AgentResult, TokenUsage


# ============================================================================
# errors.py  —  format_error
# ============================================================================

class TestFormatErrorCategories:
    """Each of the 6 known error categories plus an unknown fallback."""

    def test_connection_refused(self):
        err = ConnectionRefusedError("Connection refused by target machine")
        out = format_error(err)
        assert "Cannot connect to LLM provider" in out
        assert "Connection refused" in out
        assert "Start LM Studio" in out

    def test_rate_limit(self):
        err = RuntimeError("rate limit exceeded, please retry")
        out = format_error(err)
        assert "Provider rate limit reached" in out
        assert "rate limit exceeded" in out
        assert "Switch to a different provider" in out

    def test_api_key(self):
        err = ValueError("Invalid API key provided")
        out = format_error(err)
        assert "API key issue" in out
        assert "Invalid API key" in out
        assert "ANTHROPIC_API_KEY" in out

    def test_timeout(self):
        err = TimeoutError("Connection timeout after 30s")
        out = format_error(err)
        assert "Request timed out" in out
        assert "Connection timeout after 30s" in out
        assert "--max-iterations" in out

    def test_context_length(self):
        err = RuntimeError("context length exceeded for model")
        out = format_error(err)
        assert "Input too long for model's context window" in out
        assert "context length exceeded" in out
        assert "Reduce the input size" in out

    def test_tool(self):
        err = RuntimeError("tool execution failed: bad args")
        out = format_error(err)
        assert "Tool execution failed" in out
        assert "tool execution failed" in out
        assert "Check the tool arguments" in out

    def test_unknown_error_fallback(self):
        err = RuntimeError("Something completely unexpected happened")
        out = format_error(err)
        assert "Something completely unexpected happened" in out
        assert "mullet doctor" in out
        # Should NOT contain any of the known suggestion sets
        assert "Suggestions:" not in out

    def test_error_message_truncated_at_200_chars(self):
        long_msg = "x" * 300
        err = RuntimeError(long_msg)
        out = format_error(err)
        # The raw error text appears only once in the generic fallback
        assert out.count("x" * 200) == 1
        assert "x" * 201 not in out

    def test_all_keys_present_in_error_suggestions(self):
        assert set(ERROR_SUGGESTIONS.keys()) == {
            "connection refused",
            "rate limit",
            "api key",
            "timeout",
            "context length",
            "tool",
        }


# ============================================================================
# output.py  —  ExecutionTracker
# ============================================================================

class TestExecutionTracker:
    def test_on_tool_call_accumulates_names(self):
        tracker = ExecutionTracker(task="demo")
        tracker.on_tool_call("read_file", "preview1")
        tracker.on_tool_call("edit_file", "preview2")
        tracker.on_tool_call("read_file", "preview3")
        assert tracker._tool_calls == ["read_file", "edit_file", "read_file"]

    @patch("mullet.core.agent.output.time.time")
    def test_build_summary_computes_duration_and_populates_fields(self, mock_time):
        mock_time.side_effect = [1000.0, 1005.5]
        tracker = ExecutionTracker(task="my task")
        tracker.on_tool_call("tool_a", "p1")
        tracker.on_tool_call("tool_b", "p2")

        result = AgentResult(
            completed=True,
            final_response="done",
            provider_used="openrouter",
            tier_used="fast",
            iterations_used=3,
            tokens_used=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        summary = tracker.build_summary(result)

        assert isinstance(summary, ExecutionSummary)
        assert summary.task == "my task"
        assert summary.provider == "openrouter"
        assert summary.tier == "fast"
        assert summary.tool_calls == ["tool_a", "tool_b"]
        assert summary.iterations == 3
        assert summary.duration_seconds == 5.5
        assert summary.completed is True
        assert summary.tokens.input_tokens == 10
        assert summary.tokens.output_tokens == 20
        assert summary.tokens.total_tokens == 30


# ============================================================================
# output.py  —  ExecutionSummary
# ============================================================================

class TestExecutionSummaryFormatVerbose:
    def test_without_tool_calls_and_without_cost(self):
        summary = ExecutionSummary(
            task="t",
            provider="p",
            model="m",
            tier="fast",
            iterations=2,
            tokens=TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10),
            duration_seconds=1.23,
            cost_estimate=0.0,
        )
        out = summary.format_verbose()
        assert "Provider: p/m" in out
        assert "Tier: fast" in out
        assert "Iterations: 2" in out
        assert "Tool calls: 0" in out
        assert "Tools used:" not in out
        assert "Tokens: 10" in out
        assert "Duration: 1.2s" in out
        assert "Est. cost:" not in out

    def test_with_tool_calls_and_with_cost(self):
        summary = ExecutionSummary(
            task="t",
            provider="p",
            model="m",
            tier="fast",
            iterations=2,
            tool_calls=["read", "edit", "read"],
            tokens=TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10),
            duration_seconds=1.23,
            cost_estimate=0.1234,
        )
        out = summary.format_verbose()
        assert "Tool calls: 3" in out
        assert "Tools used:" in out
        assert "read" in out
        assert "edit" in out
        assert "Est. cost: $0.1234" in out

    def test_default_values(self):
        summary = ExecutionSummary(task="t")
        out = summary.format_verbose()
        assert "Provider: /" in out
        assert "Tier: " in out
        assert "Iterations: 0" in out
        assert "Tool calls: 0" in out
        assert "Tokens: 0" in out
        assert "Duration: 0.0s" in out
        assert "Est. cost:" not in out


class TestExecutionSummaryFormatJson:
    def test_format_json_structure(self):
        summary = ExecutionSummary(
            task="my task",
            provider="openrouter",
            model="gpt-4",
            tier="fast",
            tool_calls=["read", "edit", "read"],
            duration_seconds=3.14159,
            tokens=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            iterations=4,
            completed=True,
            cost_estimate=0.0012345,
        )
        data = summary.format_json()
        assert data["task"] == "my task"
        assert data["completed"] is True
        assert data["provider"] == "openrouter"
        assert data["model"] == "gpt-4"
        assert data["tier"] == "fast"
        assert data["tool_calls"] == ["read", "edit", "read"]
        assert sorted(data["unique_tools"]) == ["edit", "read"]
        assert data["duration_seconds"] == 3.14
        assert data["tokens"] == {"input": 100, "output": 50, "total": 150}
        assert data["iterations"] == 4
        assert data["cost_estimate"] == 0.001234

    def test_format_json_defaults(self):
        summary = ExecutionSummary(task="t")
        data = summary.format_json()
        assert data["task"] == "t"
        assert data["completed"] is False
        assert data["provider"] == ""
        assert data["model"] == ""
        assert data["tier"] == ""
        assert data["tool_calls"] == []
        assert data["unique_tools"] == []
        assert data["duration_seconds"] == 0.0
        assert data["tokens"] == {"input": 0, "output": 0, "total": 0}
        assert data["iterations"] == 0
        assert data["cost_estimate"] == 0.0
