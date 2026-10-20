"""Tests for the stats command."""

import json
from mullet.cli.stats import get_stats, format_stats


def test_no_sessions(tmp_path):
    stats = get_stats(sessions_dir=tmp_path / "empty")
    assert stats["sessions"] == 0


def test_with_sessions(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)

    for i in range(3):
        (session_dir / f"session_{i}.json").write_text(json.dumps({
            "id": f"session_{i}",
            "metadata": {"created_at": 1700000000 + i * 3600, "provider": "local"},
            "messages": [{"role": "user", "content": f"msg {i}"}],
        }))

    stats = get_stats(sessions_dir=session_dir)
    assert stats["sessions"] == 3
    assert stats["messages"] == 3
    assert stats["providers"]["local"] == 3


def test_format_empty(tmp_path, monkeypatch):
    # Redirect all data sources to empty dirs so the "no data" branch is hit
    monkeypatch.setattr("mullet.core.agent.session.SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr("mullet.core.agent.analytics.ANALYTICS_DIR", tmp_path / "no_analytics")
    monkeypatch.setattr("mullet.core.router.learning.LEARNING_DIR", tmp_path / "no_routing")
    output = format_stats()
    assert "No usage data" in output


def test_format_with_data(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    (session_dir / "session_1.json").write_text(json.dumps({
        "id": "session_1",
        "metadata": {"created_at": 1700000000, "provider": "local"},
        "messages": [{"role": "user", "content": "hello"}],
    }))

    monkeypatch.setattr("mullet.core.agent.session.SESSION_DIR", session_dir)
    output = format_stats()
    assert "Sessions:" in output
    assert "local" in output


# ---------------------------------------------------------------------------
# Parity tests: get_stats output matches pre-refactor shape
# ---------------------------------------------------------------------------

def test_get_stats_returns_expected_keys(tmp_path):
    """All keys that callers depend on must still be present."""
    stats = get_stats(sessions_dir=tmp_path / "empty")
    expected_keys = {"sessions", "messages", "tool_calls", "avg_messages_per_session", "providers", "earliest", "latest"}
    assert expected_keys.issubset(stats.keys())


def test_get_stats_tool_calls_aggregated(tmp_path):
    session_dir = tmp_path
    for i, tc in enumerate([2, 5, 3]):
        (session_dir / f"session_{i}.json").write_text(json.dumps({
            "id": f"session_{i}",
            "metadata": {"created_at": 1700000000 + i, "provider": "kimi", "tool_calls": tc},
            "messages": [],
        }))
    stats = get_stats(sessions_dir=session_dir)
    assert stats["tool_calls"] == 10
    assert stats["sessions"] == 3


def test_get_stats_avg_messages_per_session(tmp_path):
    session_dir = tmp_path
    for i, n_msgs in enumerate([2, 4]):
        (session_dir / f"session_{i}.json").write_text(json.dumps({
            "id": f"session_{i}",
            "metadata": {"created_at": 1700000000 + i, "provider": "local"},
            "messages": [{"role": "user", "content": "x"}] * n_msgs,
        }))
    stats = get_stats(sessions_dir=session_dir)
    assert stats["avg_messages_per_session"] == 3.0


def test_format_routing_history(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    (session_dir / "session_1.json").write_text(json.dumps({
        "id": "session_1",
        "metadata": {"created_at": 1700000000, "provider": "local"},
        "messages": [{"role": "user", "content": "hello"}],
    }))

    monkeypatch.setattr("mullet.core.agent.session.SESSION_DIR", session_dir)

    def _fake_summary():
        return {
            "openrouter": {
                "calls": 10,
                "successes": 9,
                "success_rate": 0.9,
                "total_cost": 1.2345,
                "avg_cost": 0.12345,
                "avg_wall_ms": 150.0,
                "p50_cost": 0.1,
                "p95_cost": 0.2,
                "p50_wall_ms": 120.0,
                "p95_wall_ms": 250.0,
            }
        }

    monkeypatch.setattr("mullet.core.routing_history.summary_by_provider", _fake_summary)
    output = format_stats()
    assert "Routing History" in output
    assert "openrouter" in output
    assert "$1.2345" in output
    assert "10" in output
    assert "90%" in output
    assert "250.0ms" in output


# ---------------------------------------------------------------------------
# Exception / branch coverage tests
# ---------------------------------------------------------------------------

def test_get_stats_invalid_iso_timestamp(monkeypatch):
    """_iso_to_epoch returns None on invalid ISO string."""
    from types import SimpleNamespace
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=1,
        total_tool_calls=0,
        by_provider={},
        window_start="not-an-iso-string",
        window_end=None,
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)
    stats = get_stats()
    assert stats["earliest"] is None
    assert stats["latest"] is None


def test_format_has_analytics_exception(monkeypatch, capsys):
    from unittest.mock import MagicMock
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: _empty_agg())
    monkeypatch.setattr("mullet.core.agent.analytics.ToolAnalytics.load_history", MagicMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr("mullet.core.router.learning._load_history", MagicMock(return_value=[]))
    output = format_stats()
    assert "No usage data" in output


def test_format_has_routing_exception(monkeypatch, capsys):
    from unittest.mock import MagicMock
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: _empty_agg())
    monkeypatch.setattr("mullet.core.agent.analytics.ToolAnalytics.load_history", MagicMock(return_value=[]))
    monkeypatch.setattr("mullet.core.router.learning._load_history", MagicMock(side_effect=RuntimeError("boom")))
    output = format_stats()
    assert "No usage data" in output


def test_format_sessions_zero_with_analytics(monkeypatch, capsys):
    """When sessions==0 but analytics exist, show the zero-sessions line."""
    from unittest.mock import MagicMock
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: _empty_agg())
    monkeypatch.setattr("mullet.core.agent.analytics.ToolAnalytics.load_history", MagicMock(return_value=[{"x": 1}]))
    monkeypatch.setattr("mullet.core.router.learning._load_history", MagicMock(return_value=[]))
    output = format_stats()
    assert "Sessions:    0" in output


def test_format_tool_calls_shown(monkeypatch, capsys):
    from types import SimpleNamespace
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=2,
        total_tool_calls=5,
        by_provider={"local": {"calls": 1}},
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-02T00:00:00",
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)
    output = format_stats()
    assert "Tool Calls:  5" in output


def test_format_database_with_data(monkeypatch, tmp_path):
    from types import SimpleNamespace
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=1,
        total_tool_calls=0,
        by_provider={},
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-01T00:00:00",
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)

    # Create a real sqlite db with usage_log data
    import sqlite3
    db_file = tmp_path / "usage.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE usage_log (input_tokens INT, output_tokens INT, estimated_cost REAL)")
    conn.execute("INSERT INTO usage_log VALUES (100, 200, 0.5)")
    conn.commit()
    conn.close()

    from unittest.mock import MagicMock
    monkeypatch.setattr("mullet.server.db.get_db_path", MagicMock(return_value=str(db_file)))
    output = format_stats()
    assert "API Calls:" in output
    assert "Tokens In:" in output
    assert "Tokens Out:" in output
    assert "Total Cost:" in output


def test_format_database_exception(monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=1,
        total_tool_calls=0,
        by_provider={},
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-01T00:00:00",
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)
    monkeypatch.setattr("mullet.server.db.get_db_path", MagicMock(side_effect=RuntimeError("db boom")))
    output = format_stats()
    assert "Sessions:" in output


def test_format_tool_analytics_exception(monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=1,
        total_tool_calls=0,
        by_provider={},
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-01T00:00:00",
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)
    monkeypatch.setattr("mullet.core.agent.analytics.ToolAnalytics.aggregate_history", MagicMock(side_effect=RuntimeError("ta boom")))
    output = format_stats()
    assert "Sessions:" in output


def test_format_routing_learning_exception(monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=1,
        total_tool_calls=0,
        by_provider={},
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-01T00:00:00",
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)
    monkeypatch.setattr("mullet.core.router.learning._load_history", MagicMock(side_effect=RuntimeError("rl boom")))
    output = format_stats()
    assert "Sessions:" in output


def test_format_routing_history_calls_zero_skipped(monkeypatch, capsys):
    from types import SimpleNamespace
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=1,
        total_tool_calls=0,
        by_provider={},
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-01T00:00:00",
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)

    def _fake_summary():
        return {
            "openrouter": {
                "calls": 0,
                "successes": 0,
                "success_rate": 0.0,
                "total_cost": 0.0,
                "avg_cost": 0.0,
                "avg_wall_ms": 0.0,
                "p50_cost": 0.0,
                "p95_cost": 0.0,
                "p50_wall_ms": 0.0,
                "p95_wall_ms": 0.0,
            }
        }

    monkeypatch.setattr("mullet.core.routing_history.summary_by_provider", _fake_summary)
    output = format_stats()
    # The provider with calls==0 should be skipped, but header still printed
    assert "Routing History" in output
    assert "openrouter" not in output


def test_format_routing_history_no_data(monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=1,
        total_tool_calls=0,
        by_provider={},
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-01T00:00:00",
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)
    monkeypatch.setattr("mullet.core.routing_history.summary_by_provider", MagicMock(return_value={}))
    output = format_stats()
    assert "No data on disk" in output


def test_format_routing_history_exception(monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=1,
        total_tool_calls=0,
        by_provider={},
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-01T00:00:00",
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)
    monkeypatch.setattr("mullet.core.routing_history.summary_by_provider", MagicMock(side_effect=RuntimeError("rh boom")))
    output = format_stats()
    assert "Sessions:" in output


def test_format_daily_cost_exception(monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    fake_agg = SimpleNamespace(
        total_sessions=1,
        total_messages=1,
        total_tool_calls=0,
        by_provider={},
        window_start="2024-01-01T00:00:00",
        window_end="2024-01-01T00:00:00",
    )
    monkeypatch.setattr("mullet.cli.stats.get_session_aggregates", lambda **kw: fake_agg)
    monkeypatch.setattr("mullet.core.routing_history.daily_cost_summary", MagicMock(side_effect=RuntimeError("dc boom")))
    output = format_stats()
    assert "Sessions:" in output


# helper
from types import SimpleNamespace

def _empty_agg():
    return SimpleNamespace(
        total_sessions=0,
        total_messages=0,
        total_tool_calls=0,
        by_provider={},
        window_start=None,
        window_end=None,
    )
