"""Tests for parallel tool execution."""

import json
import time
from pathlib import Path

from mullet.core.agent.parallel import (
    should_parallelize,
    execute_parallel,
    _paths_overlap,
    _extract_scope_path,
)


def _tc(name: str, args: dict | None = None, tool_id: str = "call_1") -> dict:
    """Helper to create a tool call dict."""
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args or {}),
        },
    }


# --- should_parallelize ---

def test_single_tool_not_parallel():
    assert should_parallelize([_tc("read_file", {"path": "a.py"})]) is False


def test_empty_not_parallel():
    assert should_parallelize([]) is False


def test_never_parallel_tools():
    calls = [_tc("web_search", {"query": "test"}), _tc("clarify", {"question": "what?"})]
    assert should_parallelize(calls) is False


def test_delegate_not_parallel():
    calls = [_tc("web_search", {"query": "a"}), _tc("delegate", {"goal": "test"})]
    assert should_parallelize(calls) is False


def test_parallel_safe_tools():
    calls = [
        _tc("web_search", {"query": "python"}, "c1"),
        _tc("git", {"action": "status"}, "c2"),
    ]
    assert should_parallelize(calls) is True


def test_parallel_safe_multiple():
    calls = [
        _tc("web_search", {"query": "a"}, "c1"),
        _tc("lsp", {"action": "diagnostics", "path": "x.py"}, "c2"),
        _tc("docker", {"action": "ps"}, "c3"),
    ]
    assert should_parallelize(calls) is True


def test_path_scoped_independent():
    calls = [
        _tc("read_file", {"path": "/tmp/a.py"}, "c1"),
        _tc("read_file", {"path": "/tmp/b.py"}, "c2"),
    ]
    assert should_parallelize(calls) is True


def test_path_scoped_overlapping():
    calls = [
        _tc("write_file", {"path": "/tmp/a.py"}, "c1"),
        _tc("patch", {"path": "/tmp/a.py"}, "c2"),
    ]
    assert should_parallelize(calls) is False


def test_path_scoped_parent_child_overlap():
    calls = [
        _tc("write_file", {"path": "/tmp/dir/a.py"}, "c1"),
        _tc("list_directory", {"path": "/tmp/dir"}, "c2"),
    ]
    assert should_parallelize(calls) is False


def test_unknown_tool_not_parallel():
    calls = [
        _tc("web_search", {"query": "a"}, "c1"),
        _tc("unknown_tool", {"x": 1}, "c2"),
    ]
    assert should_parallelize(calls) is False


def test_malformed_args_not_parallel():
    calls = [
        {"id": "c1", "function": {"name": "read_file", "arguments": "not json{{{"}},
        _tc("read_file", {"path": "b.py"}, "c2"),
    ]
    assert should_parallelize(calls) is False


def test_mixed_safe_and_path_scoped():
    calls = [
        _tc("web_search", {"query": "python"}, "c1"),
        _tc("read_file", {"path": "/tmp/a.py"}, "c2"),
    ]
    assert should_parallelize(calls) is True


# --- _paths_overlap ---

def test_paths_overlap_same():
    assert _paths_overlap(Path("/a/b.py"), Path("/a/b.py")) is True


def test_paths_overlap_parent():
    assert _paths_overlap(Path("/a"), Path("/a/b.py")) is True


def test_paths_overlap_child():
    assert _paths_overlap(Path("/a/b/c"), Path("/a/b")) is True


def test_paths_no_overlap():
    assert _paths_overlap(Path("/a/b"), Path("/c/d")) is False


def test_paths_empty():
    assert _paths_overlap(Path(""), Path("/a")) is False


# --- _extract_scope_path ---

def test_extract_path_absolute():
    p = _extract_scope_path({"path": "/tmp/a.py"})
    assert p is not None
    assert p == Path("/tmp/a.py")


def test_extract_path_relative():
    p = _extract_scope_path({"path": "src/main.py"})
    assert p is not None
    assert p.is_absolute()


def test_extract_path_missing():
    assert _extract_scope_path({}) is None
    assert _extract_scope_path({"path": ""}) is None
    assert _extract_scope_path({"path": 123}) is None


# --- execute_parallel ---

def test_execute_parallel_basic():
    def dispatch(name, args):
        return json.dumps({"tool": name, "ok": True})

    calls = [
        _tc("web_search", {"query": "a"}, "c1"),
        _tc("git", {"action": "status"}, "c2"),
    ]
    results = execute_parallel(calls, dispatch)
    assert len(results) == 2
    assert results[0]["name"] == "web_search"
    assert results[1]["name"] == "git"
    assert all(r["success"] for r in results)


def test_execute_parallel_preserves_order():
    def dispatch(name, args):
        parsed = json.loads(args) if isinstance(args, str) else args
        if parsed.get("query") == "slow":
            time.sleep(0.05)
        return f"result-{name}"

    calls = [
        _tc("web_search", {"query": "slow"}, "c1"),
        _tc("web_search", {"query": "fast"}, "c2"),
    ]
    results = execute_parallel(calls, dispatch)
    assert results[0]["tool_call_id"] == "c1"
    assert results[1]["tool_call_id"] == "c2"


def test_execute_parallel_error_in_one():
    def dispatch(name, args):
        if name == "bad_tool":
            raise RuntimeError("tool failed")
        return "ok"

    calls = [
        _tc("web_search", {"query": "a"}, "c1"),
        {"id": "c2", "function": {"name": "bad_tool", "arguments": "{}"}},
    ]
    results = execute_parallel(calls, dispatch)
    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert "error" in results[1]["result"]


def test_execute_parallel_callbacks():
    started = []
    completed = []

    def on_start(name, args):
        started.append(name)

    def on_result(name, preview):
        completed.append(name)

    def dispatch(name, args):
        return "ok"

    calls = [
        _tc("web_search", {"query": "a"}, "c1"),
        _tc("git", {"action": "status"}, "c2"),
    ]
    execute_parallel(calls, dispatch, on_tool_call=on_start, on_tool_result=on_result)
    assert len(started) == 2
    assert len(completed) == 2


def test_execute_parallel_records_duration():
    def dispatch(name, args):
        time.sleep(0.01)
        return "ok"

    results = execute_parallel([_tc("web_search", {"query": "a"})], dispatch)
    assert results[0]["duration_ms"] >= 5  # At least some measurable time
