"""Tests for the stream accumulator module."""

import json
from mullet.core.agent.streaming import StreamAccumulator, ToolCallAccumulator


# ── StreamAccumulator ────────────────────────────────────────────────────────

def test_complete_json_in_one_chunk():
    acc = StreamAccumulator()
    results = acc.accumulate('{"key": "value"}')
    assert len(results) == 1
    assert results[0] == {"key": "value"}


def test_json_split_across_chunks():
    acc = StreamAccumulator()
    results1 = acc.accumulate('{"ke')
    assert results1 == []
    results2 = acc.accumulate('y": "val')
    assert results2 == []
    results3 = acc.accumulate('ue"}')
    assert len(results3) == 1
    assert results3[0] == {"key": "value"}


def test_nested_json():
    acc = StreamAccumulator()
    obj = {"outer": {"inner": "value"}}
    results = acc.accumulate(json.dumps(obj))
    assert len(results) == 1
    assert results[0] == obj


def test_multiple_objects():
    acc = StreamAccumulator()
    results = acc.accumulate('{"a": 1}{"b": 2}')
    assert len(results) == 2
    assert results[0] == {"a": 1}
    assert results[1] == {"b": 2}


def test_text_before_json():
    acc = StreamAccumulator()
    results = acc.accumulate('Some text before {"key": "value"}')
    assert len(results) == 1
    assert results[0] == {"key": "value"}


def test_flush_remaining():
    acc = StreamAccumulator()
    acc.accumulate("Just plain text")
    remaining = acc.flush()
    assert "plain text" in remaining


def test_flush_partial_json():
    acc = StreamAccumulator()
    acc.accumulate('{"incomplete": ')
    assert acc.has_pending_json
    remaining = acc.flush()
    assert "incomplete" in remaining


def test_empty_input():
    acc = StreamAccumulator()
    assert acc.accumulate("") == []
    assert acc.flush() == ""


def test_json_with_escaped_quotes():
    acc = StreamAccumulator()
    results = acc.accumulate('{"msg": "say \\"hello\\""}')
    assert len(results) == 1
    assert results[0]["msg"] == 'say "hello"'


def test_json_with_string_braces():
    acc = StreamAccumulator()
    results = acc.accumulate('{"code": "if (x) { y }"}')
    assert len(results) == 1
    assert "{" in results[0]["code"]


def test_pending_property():
    acc = StreamAccumulator()
    assert acc.pending == ""
    acc.accumulate('{"partial')
    assert "partial" in acc.pending


def test_has_pending_json():
    acc = StreamAccumulator()
    assert not acc.has_pending_json
    acc.accumulate('{"start')
    assert acc.has_pending_json


# ── ToolCallAccumulator ──────────────────────────────────────────────────────

def test_tool_call_single_delta():
    acc = ToolCallAccumulator()
    delta = {
        "tool_calls": [{
            "index": 0,
            "id": "call_1",
            "function": {"name": "read_file", "arguments": ""},
        }],
    }
    acc.accumulate_delta(delta)
    assert acc.pending_count == 1


def test_tool_call_accumulate_args():
    acc = ToolCallAccumulator()
    acc.accumulate_delta({
        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read", "arguments": '{"pa'}}],
    })
    acc.accumulate_delta({
        "tool_calls": [{"index": 0, "function": {"arguments": 'th": "/tmp"}'}}],
    })
    calls = acc.flush()
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["path"] == "/tmp"


def test_tool_call_multiple_parallel():
    acc = ToolCallAccumulator()
    acc.accumulate_delta({
        "tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "read", "arguments": '{"path": "/a"}'}},
            {"index": 1, "id": "call_2", "function": {"name": "write", "arguments": '{"path": "/b"}'}},
        ],
    })
    calls = acc.flush()
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "read"
    assert calls[1]["function"]["name"] == "write"


def test_tool_call_flush_resets():
    acc = ToolCallAccumulator()
    acc.accumulate_delta({
        "tool_calls": [{"index": 0, "id": "x", "function": {"name": "f", "arguments": "{}"}}],
    })
    acc.flush()
    assert acc.pending_count == 0


def test_tool_call_empty_delta():
    acc = ToolCallAccumulator()
    acc.accumulate_delta({})
    assert acc.pending_count == 0


def test_stream_accumulate_invalid_json_triggers_debug_log():
    """Balanced braces that aren't valid JSON hit the JSONDecodeError branch (lines 66-67)."""
    acc = StreamAccumulator()
    # A balanced-brace token that isn't valid JSON — will enter the except branch
    results = acc.accumulate("{not valid json}")
    # No result emitted on parse failure
    assert results == []


def test_stream_accumulate_unbalanced_closing_brace_resets():
    """An extra closing brace with no open context resets depth to 0 (lines 71-72)."""
    acc = StreamAccumulator()
    # A lone closing brace with no preceding open brace — brace_depth goes below 0
    results = acc.accumulate("}")
    assert results == []
    # Internal state should be reset (depth back to 0, buffer cleared)
    assert acc._brace_depth == 0
    assert acc._buffer == ""


def test_tool_call_flush_with_incomplete_invalid_json():
    """flush() with an incomplete call whose arguments are bad JSON hits lines 163-170."""
    acc = ToolCallAccumulator()
    # Start a tool call but never close the JSON
    acc.accumulate_delta({
        "tool_calls": [{"index": 0, "id": "call_x", "function": {"name": "f", "arguments": "{bad"}}],
    })
    # The call is still pending (not completed)
    assert acc.pending_count == 1
    calls = acc.flush()
    # It should still appear in the flushed output
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "f"
