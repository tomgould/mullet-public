"""Tests for tool usage analytics."""

import json
import time
from pathlib import Path
import pytest
from mullet.core.agent.analytics import ToolAnalytics, ToolStats


@pytest.fixture(autouse=True)
def analytics_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("mullet.core.agent.analytics.ANALYTICS_DIR", tmp_path / "analytics")
    return tmp_path / "analytics"


def test_record_success():
    a = ToolAnalytics()
    a.record("terminal", success=True, duration_ms=50)
    assert len(a._records) == 1
    assert a._records[0].success is True


def test_record_failure():
    a = ToolAnalytics()
    a.record("read_file", success=False, error="File not found")
    assert a._records[0].success is False
    assert a._records[0].error == "File not found"


def test_get_stats():
    a = ToolAnalytics()
    a.record("terminal", success=True, duration_ms=50)
    a.record("terminal", success=True, duration_ms=100)
    a.record("terminal", success=False, error="timeout")
    a.record("read_file", success=True, duration_ms=10)

    stats = a.get_stats()
    assert "terminal" in stats
    assert stats["terminal"].total_calls == 3
    assert stats["terminal"].successes == 2
    assert stats["terminal"].failures == 1
    assert stats["terminal"].success_rate == pytest.approx(2/3, abs=0.01)
    assert stats["terminal"].avg_duration_ms == pytest.approx(75, abs=1)  # (50+100)/2 non-zero
    assert "read_file" in stats
    assert stats["read_file"].total_calls == 1


def test_get_summary():
    a = ToolAnalytics()
    a.record("terminal", success=True, duration_ms=50)
    a.record("git", success=True, duration_ms=30)

    summary = a.get_summary()
    assert summary["total_tool_calls"] == 2
    assert summary["unique_tools_used"] == 2
    assert "terminal" in summary["tools"]
    assert summary["tools"]["terminal"]["calls"] == 1


def test_save_and_load(analytics_dir):
    a = ToolAnalytics()
    a.record("terminal", success=True)
    a.record("read_file", success=False, error="not found")
    a.save()

    # Check file was created
    files = list(analytics_dir.glob("session_*.json"))
    assert len(files) == 1

    # Load and verify
    data = json.loads(files[0].read_text())
    assert data["total_tool_calls"] == 2


def test_aggregate_history(analytics_dir):
    # Create multiple session files
    for i in range(3):
        a = ToolAnalytics()
        a._session_start = 1000000 + i
        a.record("terminal", success=True)
        if i % 2 == 0:
            a.record("git", success=True)
        a.save()

    history = ToolAnalytics.aggregate_history(days=30)
    assert history["sessions"] == 3
    assert history["tools"]["terminal"]["total_calls"] == 3
    assert history["tools"]["git"]["total_calls"] == 2


def test_empty_analytics():
    a = ToolAnalytics()
    stats = a.get_stats()
    assert stats == {}

    summary = a.get_summary()
    assert summary["total_tool_calls"] == 0


def test_load_history_empty(analytics_dir):
    sessions = ToolAnalytics.load_history()
    assert sessions == []


def test_tool_stats_success_rate():
    s = ToolStats(name="test", total_calls=10, successes=8, failures=2)
    assert s.success_rate == 0.8


def test_tool_stats_zero_calls():
    s = ToolStats(name="test")
    assert s.success_rate == 0.0


def test_save_handles_write_error(analytics_dir, monkeypatch, caplog):
    a = ToolAnalytics()
    a.record("terminal", success=True)

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _raise)
    with caplog.at_level("DEBUG", logger="mullet.core.agent.analytics"):
        a.save()
    assert "Failed to save analytics" in caplog.text


def test_load_history_excludes_old_sessions(analytics_dir):
    analytics_dir.mkdir(parents=True, exist_ok=True)
    old_ts = int(time.time() - 100 * 86400)
    old_file = analytics_dir / f"session_{old_ts}.json"
    old_file.write_text('{"total_tool_calls": 1}')

    recent_ts = int(time.time())
    recent_file = analytics_dir / f"session_{recent_ts}.json"
    recent_file.write_text('{"total_tool_calls": 2}')

    sessions = ToolAnalytics.load_history(days=30)
    assert len(sessions) == 1
    assert sessions[0]["total_tool_calls"] == 2


def test_load_history_skips_corrupt_json(analytics_dir):
    analytics_dir.mkdir(parents=True, exist_ok=True)
    bad_file = analytics_dir / "session_9999999999.json"
    bad_file.write_text("not json")
    sessions = ToolAnalytics.load_history()
    assert sessions == []


def test_aggregate_history_empty(analytics_dir):
    history = ToolAnalytics.aggregate_history(days=30)
    assert history == {"sessions": 0, "tools": {}}


def test_load_history_skips_unparseable_timestamp(analytics_dir):
    """Lines 133-134: ValueError in float() is caught, file included anyway."""
    analytics_dir.mkdir(parents=True, exist_ok=True)
    bad_ts_file = analytics_dir / "session_notanumber.json"
    bad_ts_file.write_text('{"total_tool_calls": 5}')
    sessions = ToolAnalytics.load_history()
    assert any(s.get("total_tool_calls") == 5 for s in sessions)


def test_load_history_caps_at_100(analytics_dir):
    """Line 142: break when 100 sessions loaded."""
    analytics_dir.mkdir(parents=True, exist_ok=True)
    base_ts = 1_700_000_000  # a plausible Unix timestamp in the future (relative to cutoff)
    for i in range(105):
        (analytics_dir / f"session_{base_ts + i}.json").write_text(json.dumps({"i": i}))
    sessions = ToolAnalytics.load_history(days=365 * 10)  # wide window so none are filtered
    assert len(sessions) == 100
