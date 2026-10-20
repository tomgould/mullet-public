"""Tests for mullet.hud.transcript — fixture-driven, no real LLM calls."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


from mullet.hud.transcript import parse_transcript, TranscriptSnapshot, _write_cache


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _ts(dt_str: str = "2026-04-24T10:00:00+00:00") -> str:
    return dt_str


def _assistant_entry(
    tool_use_blocks: list | None = None,
    tool_result_blocks: list | None = None,
    timestamp: str = "2026-04-24T10:00:00+00:00",
    usage: dict | None = None,
) -> dict:
    content = []
    if tool_use_blocks:
        content.extend(tool_use_blocks)
    if tool_result_blocks:
        content.extend(tool_result_blocks)
    entry: dict = {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {"content": content},
    }
    if usage:
        entry["message"]["usage"] = usage
    return entry


def _user_entry(
    tool_result_blocks: list | None = None,
    timestamp: str = "2026-04-24T10:00:01+00:00",
) -> dict:
    content = tool_result_blocks or []
    return {
        "type": "human",
        "timestamp": timestamp,
        "message": {"content": content},
    }


def _tool_use_block(id: str, name: str, input: dict | None = None) -> dict:
    return {"type": "tool_use", "id": id, "name": name, "input": input or {}}


def _tool_result_block(tool_use_id: str, is_error: bool = False) -> dict:
    b: dict = {"type": "tool_result", "tool_use_id": tool_use_id}
    if is_error:
        b["is_error"] = True
    return b


def _write_jsonl(path: Path, lines: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_missing_file_returns_empty_snapshot():
    snap = parse_transcript("/nonexistent/path/to/transcript.jsonl")
    assert isinstance(snap, TranscriptSnapshot)
    assert snap.running_tools == []
    assert snap.completed_tools == []
    assert snap.agents == []
    assert snap.todos == []
    assert snap.session_started_at is None


def test_empty_file_returns_empty_snapshot(tmp_path):
    p = tmp_path / "transcript.jsonl"
    p.write_text("")
    snap = parse_transcript(str(p))
    assert isinstance(snap, TranscriptSnapshot)
    assert snap.running_tools == []
    assert snap.completed_tools == []


def test_malformed_json_line_skipped_others_parsed(tmp_path):
    p = tmp_path / "transcript.jsonl"
    good = _assistant_entry(
        tool_use_blocks=[_tool_use_block("id1", "Read", {"file_path": "/foo.py"})],
    )
    with open(p, "w") as f:
        f.write("NOT JSON AT ALL\n")
        f.write(json.dumps(good) + "\n")
        f.write("{broken\n")

    snap = parse_transcript(str(p))
    # The good line contributes one running tool
    assert len(snap.running_tools) == 1
    assert snap.running_tools[0].name == "Read"


def test_tool_use_without_result_is_running(tmp_path):
    p = tmp_path / "transcript.jsonl"
    entry = _assistant_entry(
        tool_use_blocks=[_tool_use_block("tid1", "Bash", {"command": "ls -la"})],
    )
    _write_jsonl(p, [entry])

    snap = parse_transcript(str(p))
    assert len(snap.running_tools) == 1
    assert snap.running_tools[0].id == "tid1"
    assert snap.running_tools[0].status == "running"


def test_tool_use_with_matching_result_is_completed(tmp_path):
    p = tmp_path / "transcript.jsonl"
    use_entry = _assistant_entry(
        tool_use_blocks=[_tool_use_block("tid2", "Read", {"file_path": "/bar.py"})],
        timestamp="2026-04-24T10:00:00+00:00",
    )
    result_entry = _user_entry(
        tool_result_blocks=[_tool_result_block("tid2")],
        timestamp="2026-04-24T10:00:02+00:00",
    )
    _write_jsonl(p, [use_entry, result_entry])

    snap = parse_transcript(str(p))
    assert len(snap.running_tools) == 0
    assert len(snap.completed_tools) == 1
    t = snap.completed_tools[0]
    assert t.id == "tid2"
    assert t.status == "completed"
    assert t.end_time is not None
    # Duration should be ~2 seconds
    duration = (t.end_time - t.start_time).total_seconds()
    assert 1.9 <= duration <= 2.1


def test_task_tool_use_goes_to_agents_not_tools(tmp_path):
    p = tmp_path / "transcript.jsonl"
    entry = _assistant_entry(
        tool_use_blocks=[
            _tool_use_block("agent1", "Task", {
                "subagent_type": "Explore",
                "model": "claude-opus-4-7",
                "description": "explore codebase",
            })
        ],
    )
    _write_jsonl(p, [entry])

    snap = parse_transcript(str(p))
    assert snap.running_tools == []
    assert len(snap.agents) == 1
    a = snap.agents[0]
    assert a.id == "agent1"
    assert a.agent_type == "Explore"
    assert a.model == "claude-opus-4-7"
    assert a.description == "explore codebase"
    assert a.status == "running"


def test_agent_tool_use_name_also_goes_to_agents(tmp_path):
    p = tmp_path / "transcript.jsonl"
    entry = _assistant_entry(
        tool_use_blocks=[
            _tool_use_block("agent2", "Agent", {
                "subagent_type": "general-purpose",
                "model": "claude-haiku-4-5",
                "description": "run tests",
            })
        ],
    )
    _write_jsonl(p, [entry])

    snap = parse_transcript(str(p))
    assert len(snap.agents) == 1
    a = snap.agents[0]
    assert a.agent_type == "general-purpose"
    assert a.model == "claude-haiku-4-5"
    assert a.description == "run tests"


def test_todo_write_latest_replaces_earlier(tmp_path):
    p = tmp_path / "transcript.jsonl"
    first_todo_entry = _assistant_entry(
        tool_use_blocks=[
            _tool_use_block("tw1", "TodoWrite", {
                "todos": [
                    {"content": "task 1", "status": "pending"},
                    {"content": "task 2", "status": "pending"},
                ]
            })
        ],
        timestamp="2026-04-24T10:00:00+00:00",
    )
    second_todo_entry = _assistant_entry(
        tool_use_blocks=[
            _tool_use_block("tw2", "TodoWrite", {
                "todos": [
                    {"content": "task A", "status": "in_progress"},
                    {"content": "task B", "status": "completed"},
                    {"content": "task C", "status": "pending"},
                ]
            })
        ],
        timestamp="2026-04-24T10:01:00+00:00",
    )
    _write_jsonl(p, [first_todo_entry, second_todo_entry])

    snap = parse_transcript(str(p))
    assert len(snap.todos) == 3
    assert snap.todos[0].content == "task A"
    assert snap.todos[1].status == "completed"
    assert snap.todos[2].content == "task C"


def test_session_tokens_summed_across_assistant_entries(tmp_path):
    p = tmp_path / "transcript.jsonl"
    e1 = _assistant_entry(
        timestamp="2026-04-24T10:00:00+00:00",
        usage={"input_tokens": 100, "output_tokens": 50,
               "cache_creation_input_tokens": 10, "cache_read_input_tokens": 20},
    )
    e2 = _assistant_entry(
        timestamp="2026-04-24T10:01:00+00:00",
        usage={"input_tokens": 200, "output_tokens": 80,
               "cache_creation_input_tokens": 5, "cache_read_input_tokens": 15},
    )
    _write_jsonl(p, [e1, e2])

    snap = parse_transcript(str(p))
    assert snap.session_tokens_in == 300
    assert snap.session_tokens_out == 130
    assert snap.session_cache_creation_tokens == 15
    assert snap.session_cache_read_tokens == 35


def test_first_entry_timestamp_sets_session_started_at(tmp_path):
    p = tmp_path / "transcript.jsonl"
    entry = {
        "type": "human",
        "timestamp": "2026-04-24T08:30:00+00:00",
        "message": {"content": []},
    }
    _write_jsonl(p, [entry])

    snap = parse_transcript(str(p))
    assert snap.session_started_at is not None
    assert snap.session_started_at == datetime.fromisoformat("2026-04-24T08:30:00+00:00")


def test_cache_hit_returns_without_reparse(tmp_path):
    """Second call with unchanged file returns cached result.

    We verify this by modifying the file AFTER the first parse but before
    the second, and checking the second call returns the stale (cached) data.
    Actually the cache is keyed on mtime_ns+size, so we need mtime unchanged.
    Instead: parse once, freeze mtime, corrupt the file, parse again — cache
    should return the correct first-parse result.
    """
    p = tmp_path / "transcript.jsonl"
    entry = _assistant_entry(
        tool_use_blocks=[_tool_use_block("c1", "Glob", {"pattern": "*.py"})],
    )
    _write_jsonl(p, [entry])

    cache_dir = str(tmp_path / "cache")

    snap1 = parse_transcript(str(p), cache_dir=cache_dir)
    assert len(snap1.running_tools) == 1

    # Corrupt the file contents but preserve mtime and size by overwriting
    # with same-length content and then restoring the original mtime.
    st = os.stat(str(p))
    original_mtime_ns = st.st_mtime_ns

    # Write garbage of the same size (forces re-parse if cache miss)
    garbage = "X" * st.st_size
    with open(p, "w") as f:
        f.write(garbage)

    # Restore original mtime to keep cache key valid
    os.utime(str(p), ns=(st.st_atime_ns, original_mtime_ns))

    snap2 = parse_transcript(str(p), cache_dir=cache_dir)
    # Should return cached result (1 running tool), not the corrupt parse (0 tools)
    assert len(snap2.running_tools) == 1


def test_cache_miss_on_size_change_reparses(tmp_path):
    p = tmp_path / "transcript.jsonl"
    entry = _assistant_entry(
        tool_use_blocks=[_tool_use_block("d1", "Write", {"file_path": "/out.txt"})],
    )
    _write_jsonl(p, [entry])

    cache_dir = str(tmp_path / "cache")

    snap1 = parse_transcript(str(p), cache_dir=cache_dir)
    assert len(snap1.running_tools) == 1

    # Append a new entry — changes the file size, triggering cache miss
    entry2 = _assistant_entry(
        tool_use_blocks=[_tool_use_block("d2", "Edit", {"file_path": "/out2.txt"})],
        timestamp="2026-04-24T10:02:00+00:00",
    )
    with open(p, "a") as f:
        f.write(json.dumps(entry2) + "\n")

    snap2 = parse_transcript(str(p), cache_dir=cache_dir)
    # Now 2 running tools
    assert len(snap2.running_tools) == 2


class TestTranscriptMissingLines:
    """Cover previously-missed lines in mullet/hud/transcript.py."""

    def test_empty_path_returns_empty(self):
        snap = parse_transcript("")
        assert isinstance(snap, TranscriptSnapshot)
        assert snap.running_tools == []

    def test_directory_path_returns_empty(self, tmp_path):
        snap = parse_transcript(str(tmp_path))
        assert isinstance(snap, TranscriptSnapshot)
        assert snap.running_tools == []

    def test_realpath_exception_returns_empty(self, monkeypatch, tmp_path):
        import mullet.hud.transcript as tmod

        monkeypatch.setattr(
            tmod.os.path, "realpath", lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        p = tmp_path / "t.jsonl"
        p.write_text("{}")
        snap = parse_transcript(str(p))
        assert snap.running_tools == []

    def test_empty_line_skipped(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        entry = _assistant_entry(
            tool_use_blocks=[_tool_use_block("e1", "Read", {"file_path": "/a.py"})],
        )
        with open(p, "w") as f:
            f.write("\n")
            f.write(json.dumps(entry) + "\n")
            f.write("   \n")
        snap = parse_transcript(str(p))
        assert len(snap.running_tools) == 1

    def test_non_dict_entry_skipped(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        entry = _assistant_entry(
            tool_use_blocks=[_tool_use_block("e2", "Read", {"file_path": "/b.py"})],
        )
        with open(p, "w") as f:
            f.write(json.dumps([1, 2, 3]) + "\n")
            f.write(json.dumps(entry) + "\n")
        snap = parse_transcript(str(p))
        assert len(snap.running_tools) == 1

    def test_invalid_timestamp_uses_now(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        entry = {
            "type": "assistant",
            "timestamp": "not-a-date",
            "message": {
                "content": [_tool_use_block("e3", "Read", {"file_path": "/c.py"})]
            },
        }
        _write_jsonl(p, [entry])
        snap = parse_transcript(str(p))
        assert len(snap.running_tools) == 1

    def test_content_not_list_skips(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        entry = {
            "type": "assistant",
            "timestamp": "2026-04-24T10:00:00+00:00",
            "message": {"content": "not a list"},
        }
        _write_jsonl(p, [entry])
        snap = parse_transcript(str(p))
        assert snap.running_tools == []

    def test_block_not_dict_skipped(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        entry = {
            "type": "assistant",
            "timestamp": "2026-04-24T10:00:00+00:00",
            "message": {
                "content": ["not a dict", _tool_use_block("e4", "Read", {"file_path": "/d.py"})]
            },
        }
        _write_jsonl(p, [entry])
        snap = parse_transcript(str(p))
        assert len(snap.running_tools) == 1

    def test_tool_result_without_tool_use_id_skipped(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        use = _assistant_entry(
            tool_use_blocks=[_tool_use_block("e5", "Read", {"file_path": "/e.py"})],
        )
        result = _user_entry(
            tool_result_blocks=[{"type": "tool_result"}],
        )
        _write_jsonl(p, [use, result])
        snap = parse_transcript(str(p))
        assert len(snap.running_tools) == 1

    def test_agent_completion_sets_status_and_end_time(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        use = _assistant_entry(
            tool_use_blocks=[
                _tool_use_block("agent1", "Task", {
                    "subagent_type": "Explore",
                    "model": "claude-opus-4-7",
                })
            ],
            timestamp="2026-04-24T10:00:00+00:00",
        )
        result = _user_entry(
            tool_result_blocks=[_tool_result_block("agent1")],
            timestamp="2026-04-24T10:00:05+00:00",
        )
        _write_jsonl(p, [use, result])
        snap = parse_transcript(str(p))
        assert len(snap.agents) == 1
        assert snap.agents[0].status == "completed"
        assert snap.agents[0].end_time is not None

    def test_compact_boundary_sets_last_compact(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        entry = {
            "type": "system",
            "subtype": "compact_boundary",
            "timestamp": "2026-04-24T10:00:00+00:00",
            "compactMetadata": {"postTokens": 1500},
        }
        _write_jsonl(p, [entry])
        snap = parse_transcript(str(p))
        assert snap.last_compact_boundary_at is not None
        assert snap.last_compact_post_tokens == 1500

    def test_compact_boundary_invalid_timestamp_ignored(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        entry = {
            "type": "system",
            "subtype": "compact_boundary",
            "timestamp": "bad-date",
            "compactMetadata": {"postTokens": 1500},
        }
        _write_jsonl(p, [entry])
        snap = parse_transcript(str(p))
        assert snap.last_compact_boundary_at is None

    def test_file_read_exception_returns_empty(self, tmp_path):
        from unittest.mock import patch

        p = tmp_path / "transcript.jsonl"
        p.write_text('{"type":"assistant","message":{"content":[]}}')
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == str(p):
                raise OSError("read fail")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", fake_open):
            snap = parse_transcript(str(p))
        assert snap.running_tools == []

    def test_cache_write_failure_not_fatal(self, tmp_path):
        snap = TranscriptSnapshot()
        # cache_dir is a file, so os.makedirs inside _write_cache will fail
        cache_file = tmp_path / "notadir"
        cache_file.write_text("x")
        # Should not raise
        _write_cache(str(tmp_path / "t.jsonl"), 0, 0, snap, str(cache_file))

