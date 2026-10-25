"""Tests for mullet.cli.logs_cmd."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.cli import logs_cmd
from mullet.cli.logs_cmd import (
    _format_latency,
    _format_time,
    run_logs_stats,
    run_logs_suggest,
    run_logs_tail,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(path: Path) -> Path:
    """Create a minimal interactions.db with a few rows."""
    db = path / "interactions.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE interactions (
            id INTEGER PRIMARY KEY,
            ts TEXT,
            provider TEXT,
            model TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency_ms INTEGER,
            success INTEGER,
            prompt_preview TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (1, "2026-04-26T10:00:00Z", "kimi", "kimi-k2.6", 500, 200, 1200, 1, "refactor auth module"),
            (2, "2026-04-26T10:01:00Z", "kimi", "kimi-k2.6", 300, 100, 950, 1, "add docstrings"),
            (3, "2026-04-26T10:02:00Z", "gemini", "gemini-2.5-flash", 1000, 400, 35000, 0, "large analysis task"),
        ],
    )
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# _format_time
# ---------------------------------------------------------------------------

class TestFormatTime:
    def test_iso_utc(self) -> None:
        assert _format_time("2026-04-26T10:30:45Z") == "10:30:45"

    def test_iso_offset(self) -> None:
        assert _format_time("2026-04-26T10:30:45+00:00") == "10:30:45"

    def test_fallback_on_bad_input(self) -> None:
        result = _format_time("not-a-timestamp")
        assert isinstance(result, str)
        assert len(result) <= 8


# ---------------------------------------------------------------------------
# _format_latency
# ---------------------------------------------------------------------------

class TestFormatLatency:
    def test_under_1s_shows_ms(self) -> None:
        assert _format_latency(500) == "500ms"

    def test_over_1s_shows_seconds(self) -> None:
        assert _format_latency(1500) == "1.5s"

    def test_exactly_999_shows_ms(self) -> None:
        assert _format_latency(999) == "999ms"

    def test_exactly_1000_shows_s(self) -> None:
        assert _format_latency(1000) == "1.0s"


# ---------------------------------------------------------------------------
# run_logs_tail
# ---------------------------------------------------------------------------

class TestRunLogsTail:
    def test_no_db_prints_message(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        with patch.object(logs_cmd, "_get_db_path", return_value=tmp_path / "missing.db"):
            run_logs_tail()
        out = capsys.readouterr().out
        assert "No interactions logged yet" in out

    def test_empty_table_prints_message(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        db = tmp_path / "interactions.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE interactions "
            "(id INTEGER PRIMARY KEY, ts TEXT, provider TEXT, model TEXT, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, "
            "latency_ms INTEGER, success INTEGER, prompt_preview TEXT)"
        )
        conn.commit()
        conn.close()
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_tail()
        out = capsys.readouterr().out
        assert "No interactions found" in out

    def test_shows_rows(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_tail(n=10)
        out = capsys.readouterr().out
        assert "kimi" in out
        assert "gemini" in out

    def test_limit_n_respected(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_tail(n=1)
        out = capsys.readouterr().out
        # With LIMIT 1 only the newest row (id=3, gemini) should appear
        lines = [l for l in out.splitlines() if l.strip() and not l.startswith("-") and "Provider" not in l]
        assert len(lines) == 1

    def test_long_preview_truncated(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        db = tmp_path / "interactions.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE interactions "
            "(id INTEGER PRIMARY KEY, ts TEXT, provider TEXT, model TEXT, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, "
            "latency_ms INTEGER, success INTEGER, prompt_preview TEXT)"
        )
        long_preview = "x" * 100
        conn.execute(
            "INSERT INTO interactions VALUES (1,'2026-04-26T10:00:00Z','k','m',1,1,100,1,?)",
            (long_preview,),
        )
        conn.commit()
        conn.close()
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_tail()
        out = capsys.readouterr().out
        assert "..." in out


# ---------------------------------------------------------------------------
# run_logs_stats
# ---------------------------------------------------------------------------

class TestRunLogsStats:
    def test_no_db_prints_message(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        with patch.object(logs_cmd, "_get_db_path", return_value=tmp_path / "missing.db"):
            run_logs_stats()
        out = capsys.readouterr().out
        assert "No interactions logged yet" in out

    def test_shows_providers(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_stats()
        out = capsys.readouterr().out
        assert "kimi" in out
        assert "gemini" in out

    def test_shows_totals_row(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_stats()
        out = capsys.readouterr().out
        assert "TOTAL" in out

    def test_empty_table_prints_message(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        db = tmp_path / "interactions.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE interactions "
            "(id INTEGER PRIMARY KEY, ts TEXT, provider TEXT, model TEXT, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, "
            "latency_ms INTEGER, success INTEGER, prompt_preview TEXT)"
        )
        conn.commit()
        conn.close()
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_stats()
        out = capsys.readouterr().out
        assert "No interactions found" in out


# ---------------------------------------------------------------------------
# run_logs_suggest
# ---------------------------------------------------------------------------

class TestRunLogsSuggest:
    def test_no_db_prints_message(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        with patch.object(logs_cmd, "_get_db_path", return_value=tmp_path / "missing.db"):
            run_logs_suggest()
        out = capsys.readouterr().out
        assert "No interactions logged yet" in out

    def test_not_enough_data_message(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Fewer than 5 calls per provider → not enough data."""
        db = _make_db(tmp_path)
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_suggest()
        out = capsys.readouterr().out
        # Only 2 kimi rows and 1 gemini row — below HAVING calls >= 5
        assert "Not enough data" in out

    def test_high_failure_rate_flagged(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Provider with >5% failure rate should produce a suggestion."""
        db = tmp_path / "interactions.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE interactions "
            "(id INTEGER PRIMARY KEY, ts TEXT, provider TEXT, model TEXT, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, "
            "latency_ms INTEGER, success INTEGER, prompt_preview TEXT)"
        )
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        # Insert 10 rows for "badprovider" with 5 failures (50% failure rate)
        for i in range(10):
            success = 1 if i < 5 else 0
            conn.execute(
                "INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?)",
                (i + 1, now, "badprovider", "bad-model", 100, 50, 1000, success, "task"),
            )
        conn.commit()
        conn.close()
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_suggest()
        out = capsys.readouterr().out
        assert "failure rate" in out

    def test_high_latency_flagged(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Provider with avg latency >30s should produce a suggestion."""
        db = tmp_path / "interactions.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE interactions "
            "(id INTEGER PRIMARY KEY, ts TEXT, provider TEXT, model TEXT, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, "
            "latency_ms INTEGER, success INTEGER, prompt_preview TEXT)"
        )
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for i in range(10):
            conn.execute(
                "INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?)",
                (i + 1, now, "slowprovider", "slow-model", 100, 50, 40000, 1, "task"),
            )
        conn.commit()
        conn.close()
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_suggest()
        out = capsys.readouterr().out
        assert "latency" in out

    def test_all_well_message(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """When no issues found, print 'performing well'."""
        db = tmp_path / "interactions.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE interactions "
            "(id INTEGER PRIMARY KEY, ts TEXT, provider TEXT, model TEXT, "
            "prompt_tokens INTEGER, completion_tokens INTEGER, "
            "latency_ms INTEGER, success INTEGER, prompt_preview TEXT)"
        )
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for i in range(10):
            conn.execute(
                "INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?)",
                (i + 1, now, "kimi", "kimi-k2.6", 100, 50, 1000, 1, "task"),
            )
        conn.commit()
        conn.close()
        with patch.object(logs_cmd, "_get_db_path", return_value=db):
            run_logs_suggest()
        out = capsys.readouterr().out
        assert "performing well" in out


def test_get_db_path_returns_path():
    """Line 9: _get_db_path returns interactions.db in the state directory."""
    from mullet.cli.logs_cmd import _get_db_path
    from mullet.core.paths import state_dir
    p = _get_db_path()
    assert p == state_dir() / "interactions.db"


def test_run_logs_tail_sqlite_error_printed(tmp_path, capsys):
    """Lines 71-72: sqlite3.Error in run_logs_tail → error message printed."""
    db = tmp_path / "bad.db"
    db.write_bytes(b"not a sqlite db")  # file exists but db_path.exists() returns True
    with patch.object(logs_cmd, "_get_db_path", return_value=db), \
         patch("mullet.cli.logs_cmd.sqlite3.connect", side_effect=sqlite3.Error("db corrupt")):
        run_logs_tail(n=10)
    out = capsys.readouterr().out
    assert "Error reading database" in out


def test_run_logs_stats_sqlite_error_printed(tmp_path, capsys):
    """Lines 135-136: sqlite3.Error in run_logs_stats → error message printed."""
    db = tmp_path / "bad.db"
    db.write_bytes(b"not a sqlite db")
    with patch.object(logs_cmd, "_get_db_path", return_value=db), \
         patch("mullet.cli.logs_cmd.sqlite3.connect", side_effect=sqlite3.Error("db corrupt")):
        run_logs_stats()
    out = capsys.readouterr().out
    assert "Error reading database" in out


def test_run_logs_suggest_sqlite_error_printed(tmp_path, capsys):
    """Lines 190-191: sqlite3.Error in run_logs_suggest → error message printed."""
    db = tmp_path / "bad.db"
    db.write_bytes(b"not a sqlite db")
    with patch.object(logs_cmd, "_get_db_path", return_value=db), \
         patch("mullet.cli.logs_cmd.sqlite3.connect", side_effect=sqlite3.Error("db corrupt")):
        run_logs_suggest()
    out = capsys.readouterr().out
    assert "Error reading database" in out
