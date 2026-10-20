"""Gap coverage tests for mullet/core/agent/parallel.py."""

import json
from unittest.mock import patch

from mullet.core.agent.parallel import should_parallelize, execute_parallel


def _tc(name, args=None, tool_id="call_1", arguments_raw=None):
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_raw if arguments_raw is not None else json.dumps(args or {}),
        },
    }


def test_should_parallelize_args_not_dict():
    """Line 94: parsed args is not a dict."""
    calls = [
        _tc("read_file", arguments_raw="[]"),
        _tc("read_file", {"path": "b.py"}, "c2"),
    ]
    assert should_parallelize(calls) is False


def test_should_parallelize_missing_path():
    """Line 99: _extract_scope_path returns None for missing path."""
    calls = [
        _tc("read_file", {"path": "/tmp/a.py"}, "c1"),
        _tc("write_file", {}, "c2"),  # missing path
    ]
    assert should_parallelize(calls) is False


def test_execute_parallel_on_tool_call_exception():
    """Lines 143-144: on_tool_call callback raises."""
    def dispatch(name, args):
        return "ok"

    def bad_on_tool_call(name, args):
        raise RuntimeError("boom")

    calls = [_tc("web_search", {"query": "a"}, "c1")]
    results = execute_parallel(calls, dispatch, on_tool_call=bad_on_tool_call)
    assert results[0]["success"] is True


def test_execute_parallel_on_tool_start_exception():
    """Lines 147-150: on_tool_start callback raises."""
    def dispatch(name, args):
        return "ok"

    def bad_on_tool_start(name, args):
        raise RuntimeError("boom")

    calls = [_tc("web_search", {"query": "a"}, "c1")]
    results = execute_parallel(calls, dispatch, on_tool_start=bad_on_tool_start)
    assert results[0]["success"] is True


def test_execute_parallel_on_tool_complete_exception():
    """Lines 164-168: on_tool_complete callback raises."""
    def dispatch(name, args):
        return "ok"

    def bad_on_tool_complete(name, preview):
        raise RuntimeError("boom")

    calls = [_tc("web_search", {"query": "a"}, "c1")]
    results = execute_parallel(calls, dispatch, on_tool_complete=bad_on_tool_complete)
    assert results[0]["success"] is True


def test_execute_parallel_on_tool_result_exception():
    """Lines 174-175: on_tool_result callback raises."""
    def dispatch(name, args):
        return "ok"

    def bad_on_tool_result(name, preview):
        raise RuntimeError("boom")

    calls = [_tc("web_search", {"query": "a"}, "c1")]
    results = execute_parallel(calls, dispatch, on_tool_result=bad_on_tool_result)
    assert results[0]["success"] is True


def test_execute_parallel_none_result_slot():
    """Line 201: results[i] is None when thread did not complete."""

    class FakeFuture:
        def result(self):
            return None

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, *args, **kwargs):
            # Don't run fn, so results[index] stays None
            return FakeFuture()

    def dispatch(name, args):
        return "ok"

    calls = [_tc("web_search", {"query": "a"}, "c1")]
    with patch("mullet.core.agent.parallel.ThreadPoolExecutor", FakeExecutor):
        results = execute_parallel(calls, dispatch)
    assert results[0]["success"] is False
    assert "Tool execution failed" in results[0]["result"]
