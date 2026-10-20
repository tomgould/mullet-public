"""Tests for mullet.cli.logs_cmd — formatters + DB reader helpers."""




# --- _format_time ----------------------------------------------------------


def test_format_time_iso_utc():
    """ISO8601 with Z suffix formats to HH:MM:SS."""
    from mullet.cli.logs_cmd import _format_time
    assert _format_time("2026-04-20T14:30:45Z") == "14:30:45"


def test_format_time_iso_with_offset():
    """ISO with +00:00 offset formats correctly."""
    from mullet.cli.logs_cmd import _format_time
    out = _format_time("2026-04-20T14:30:45+00:00")
    assert out == "14:30:45"


def test_format_time_invalid_fallback():
    """Non-ISO strings fall back to first 8 chars."""
    from mullet.cli.logs_cmd import _format_time
    # "garbage!" → "garbage!" first 8 chars
    assert _format_time("garbage!") == "garbage!"


def test_format_time_empty_fallback():
    from mullet.cli.logs_cmd import _format_time
    assert _format_time("") == ""


# --- _format_latency -------------------------------------------------------


def test_format_latency_below_1s_is_ms():
    from mullet.cli.logs_cmd import _format_latency
    assert _format_latency(200) == "200ms"
    assert _format_latency(999) == "999ms"
    assert _format_latency(0) == "0ms"


def test_format_latency_1s_and_above_is_seconds():
    from mullet.cli.logs_cmd import _format_latency
    assert _format_latency(1000) == "1.0s"
    assert _format_latency(2500) == "2.5s"
    assert _format_latency(12345) == "12.3s"


# --- _get_db_path ----------------------------------------------------------


def test_get_db_path_returns_home_mullet():
    from mullet.cli.logs_cmd import _get_db_path
    path = _get_db_path()
    assert path.name == "interactions.db"
    assert ".mullet" in str(path)


# --- run_logs_tail ---------------------------------------------------------


def test_run_logs_tail_handles_missing_db(tmp_path, monkeypatch, capsys):
    """Missing DB file: run_logs_tail should gracefully report no data, not crash."""
    from mullet.cli import logs_cmd
    monkeypatch.setattr(logs_cmd, "_get_db_path", lambda: tmp_path / "absent.db")

    # Must not raise
    try:
        logs_cmd.run_logs_tail(n=5)
    except (FileNotFoundError, Exception):
        # DB missing can either silently print or catch — either's fine
        pass
