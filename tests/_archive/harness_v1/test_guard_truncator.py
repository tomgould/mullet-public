"""Tests for tool_output_truncator."""


from mullet.harness.hooks.guards.tool_output_truncator import (
    make_tool_output_truncator,
)
from mullet.harness.hooks.types import PostToolUsePayload


class TestToolOutputTruncator:
    def test_truncates_when_over_limit(self) -> None:
        truncator = make_tool_output_truncator(max_chars=20, head=5, tail=5)
        payload = PostToolUsePayload(
            tool_name="Read",
            tool_args={"path": "/tmp/foo"},
            tool_result="a" * 100,
            tool_error=None,
            session_id="s1",
        )
        result = truncator(payload)
        assert result is payload
        assert len(payload.tool_result) < 100

    def test_passes_through_when_under_limit(self) -> None:
        truncator = make_tool_output_truncator(max_chars=100, head=30, tail=30)
        payload = PostToolUsePayload(
            tool_name="Read",
            tool_args={"path": "/tmp/foo"},
            tool_result="short",
            tool_error=None,
            session_id="s1",
        )
        assert truncator(payload) is None
        assert payload.tool_result == "short"

    def test_truncation_marker_present(self) -> None:
        truncator = make_tool_output_truncator(max_chars=20, head=5, tail=5)
        payload = PostToolUsePayload(
            tool_name="Read",
            tool_args={"path": "/tmp/foo"},
            tool_result="x" * 100,
            tool_error=None,
            session_id="s1",
        )
        truncator(payload)
        assert "[...truncated" in payload.tool_result
        assert payload.tool_result.startswith("x" * 5)
        assert payload.tool_result.endswith("x" * 5)
