"""Tests for get_session_aggregates in mullet.core.agent.session."""

import json
import time
from pathlib import Path


from mullet.core.agent.session import SessionAggregates, get_session_aggregates


def _write_session(
    directory: Path,
    name: str,
    *,
    provider: str = "local",
    tool_calls: int = 0,
    tool_call_counts: dict | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    created_at: float | None = None,
    messages: list | None = None,
) -> Path:
    """Write a minimal session JSON file and return its path.

    The filename always starts with ``session_`` so it matches the glob
    ``session_*.json`` used by ``get_session_aggregates``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    # Ensure the filename matches the glob pattern expected by the helper
    fname = name if name.startswith("session_") else f"session_{name}"
    path = directory / f"{fname}.json"
    path.write_text(json.dumps({
        "id": fname,
        "metadata": {
            "provider": provider,
            "tool_calls": tool_calls,
            "tool_call_counts": tool_call_counts or {},
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "created_at": created_at if created_at is not None else time.time(),
        },
        "messages": messages or [{"role": "user", "content": "hi"}],
    }))
    return path


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_empty_dir_returns_zero_aggregates(tmp_path):
    result = get_session_aggregates(sessions_dir=tmp_path / "empty")
    assert isinstance(result, SessionAggregates)
    assert result.total_sessions == 0
    assert result.total_tool_calls == 0
    assert result.total_tokens_in == 0
    assert result.total_tokens_out == 0
    assert result.total_cost_usd == 0.0
    assert result.by_provider == {}
    assert result.by_tool == {}
    assert result.window_start is None
    assert result.window_end is None


def test_single_session_roundtrip(tmp_path):
    _write_session(
        tmp_path,
        "session_1",
        provider="kimi",
        tool_calls=5,
        tool_call_counts={"bash": 3, "read": 2},
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.05,
        messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
    )
    result = get_session_aggregates(sessions_dir=tmp_path)
    assert result.total_sessions == 1
    assert result.total_messages == 2
    assert result.total_tool_calls == 5
    assert result.total_tokens_in == 100
    assert result.total_tokens_out == 200
    assert abs(result.total_cost_usd - 0.05) < 1e-9
    assert result.by_tool == {"bash": 3, "read": 2}
    assert "kimi" in result.by_provider
    assert result.by_provider["kimi"]["calls"] == 1


def test_multiple_sessions_summed(tmp_path):
    _write_session(tmp_path, "session_a", provider="local", tool_calls=2, tokens_in=50, tokens_out=80)
    _write_session(tmp_path, "session_b", provider="kimi",  tool_calls=3, tokens_in=70, tokens_out=120)
    _write_session(tmp_path, "session_c", provider="local", tool_calls=1, tokens_in=30, tokens_out=40)

    result = get_session_aggregates(sessions_dir=tmp_path)
    assert result.total_sessions == 3
    assert result.total_tool_calls == 6
    assert result.total_tokens_in == 150
    assert result.total_tokens_out == 240
    assert result.by_provider["local"]["calls"] == 2
    assert result.by_provider["kimi"]["calls"] == 1


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def test_days_filter_excludes_old_sessions(tmp_path):
    old_ts = time.time() - 40 * 86400  # 40 days ago
    new_ts = time.time() - 1 * 86400   # 1 day ago

    old_path = _write_session(tmp_path, "session_old", created_at=old_ts)
    # Back-date mtime so the days filter sees it as old
    import os
    os.utime(old_path, (old_ts, old_ts))

    new_path = _write_session(tmp_path, "session_new", created_at=new_ts)
    os.utime(new_path, (new_ts, new_ts))

    result = get_session_aggregates(days=30, sessions_dir=tmp_path)
    assert result.total_sessions == 1


def test_limit_filter_takes_most_recent(tmp_path):
    base = time.time()
    for i in range(5):
        p = _write_session(tmp_path, f"session_{i:04d}", created_at=base + i)
        import os
        os.utime(p, (base + i, base + i))

    result = get_session_aggregates(limit=3, sessions_dir=tmp_path)
    assert result.total_sessions == 3


def test_limit_and_no_days_respects_most_recent(tmp_path):
    base = time.time()
    for i in range(10):
        p = _write_session(tmp_path, f"session_{i:04d}", tool_calls=i, created_at=base + i)
        import os
        os.utime(p, (base + i, base + i))

    result = get_session_aggregates(limit=5, sessions_dir=tmp_path)
    assert result.total_sessions == 5
    # The 5 most-recent have tool_calls 5,6,7,8,9 — sum = 35
    assert result.total_tool_calls == 5 + 6 + 7 + 8 + 9


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_malformed_json_skipped(tmp_path):
    (tmp_path / "session_bad.json").write_text("{not valid json")
    _write_session(tmp_path, "session_good", tool_calls=7)

    result = get_session_aggregates(sessions_dir=tmp_path)
    assert result.total_sessions == 1
    assert result.total_tool_calls == 7


def test_missing_metadata_fields_default_to_zero(tmp_path):
    (tmp_path / "session_minimal.json").write_text(json.dumps({
        "id": "session_minimal",
        "metadata": {},
        "messages": [],
    }))
    result = get_session_aggregates(sessions_dir=tmp_path)
    assert result.total_sessions == 1
    assert result.total_tool_calls == 0
    assert result.total_tokens_in == 0
    assert result.by_provider["unknown"]["calls"] == 1


# ---------------------------------------------------------------------------
# by_provider and by_tool
# ---------------------------------------------------------------------------

def test_by_provider_accumulates_tokens(tmp_path):
    _write_session(tmp_path, "s1", provider="gemini", tokens_in=10, tokens_out=20, cost_usd=0.01)
    _write_session(tmp_path, "s2", provider="gemini", tokens_in=15, tokens_out=25, cost_usd=0.02)

    result = get_session_aggregates(sessions_dir=tmp_path)
    g = result.by_provider["gemini"]
    assert g["calls"] == 2
    assert g["tokens_in"] == 25
    assert g["tokens_out"] == 45
    assert abs(g["cost"] - 0.03) < 1e-9


def test_by_tool_accumulated_across_sessions(tmp_path):
    _write_session(tmp_path, "s1", tool_call_counts={"bash": 2, "read": 1})
    _write_session(tmp_path, "s2", tool_call_counts={"bash": 3, "write": 4})

    result = get_session_aggregates(sessions_dir=tmp_path)
    assert result.by_tool["bash"] == 5
    assert result.by_tool["read"] == 1
    assert result.by_tool["write"] == 4


# ---------------------------------------------------------------------------
# Window timestamps
# ---------------------------------------------------------------------------

def test_window_start_end_from_created_at(tmp_path):
    t1 = 1_700_000_000.0
    t2 = 1_700_100_000.0
    _write_session(tmp_path, "s_old", created_at=t1)
    _write_session(tmp_path, "s_new", created_at=t2)

    result = get_session_aggregates(sessions_dir=tmp_path)
    assert result.window_start is not None
    assert result.window_end is not None
    # Oldest session → window_start; newest → window_end
    import datetime
    start_ts = datetime.datetime.fromisoformat(result.window_start).timestamp()
    end_ts = datetime.datetime.fromisoformat(result.window_end).timestamp()
    assert abs(start_ts - t1) < 1
    assert abs(end_ts - t2) < 1
