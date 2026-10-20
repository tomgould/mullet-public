"""Tests for cost_history.py."""

import json
from datetime import date, timedelta

import pytest

from mullet.core.providers import cost_history


@pytest.fixture(autouse=True)
def _patch_path(monkeypatch, tmp_path):
    """Redirect COST_HISTORY_PATH to a temp file for every test."""
    monkeypatch.setattr(cost_history, "COST_HISTORY_PATH", tmp_path / "cost_history.jsonl")


def test_record_creates_file():
    assert not cost_history.COST_HISTORY_PATH.exists()
    cost_history.record_session(0.5)
    assert cost_history.COST_HISTORY_PATH.exists()


def test_record_appends():
    cost_history.record_session(0.1)
    cost_history.record_session(0.2)
    lines = cost_history.COST_HISTORY_PATH.read_text().strip().splitlines()
    assert len(lines) == 2


def test_load_history_empty():
    assert cost_history.load_history() == []


def test_load_history_filters_by_window():
    days_ago_30 = str(date.today() - timedelta(days=30))
    days_ago_20 = str(date.today() - timedelta(days=20))
    days_ago_5 = str(date.today() - timedelta(days=5))
    with open(cost_history.COST_HISTORY_PATH, "w") as f:
        for d, cost in [(days_ago_30, 0.1), (days_ago_20, 0.2), (days_ago_5, 0.3)]:
            f.write(json.dumps({"date": d, "session_usd": cost, "providers": {}}) + "\n")
    entries = cost_history.load_history(window_days=25)
    assert len(entries) == 2
    assert entries[0]["session_usd"] == 0.2
    assert entries[1]["session_usd"] == 0.3


def test_get_percentiles_basic():
    for i in range(1, 11):
        cost_history.record_session(i * 0.1)
    stats = cost_history.get_percentiles(window_days=30)
    assert stats["n"] == 10
    assert pytest.approx(stats["p50"], abs=0.01) == 0.55
    assert pytest.approx(stats["p95"], abs=0.01) == 0.95
    assert pytest.approx(stats["mean"], abs=0.01) == 0.55
    assert pytest.approx(stats["max"], abs=0.01) == 1.0


def test_get_percentiles_too_few():
    stats = cost_history.get_percentiles(window_days=30)
    assert stats == {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0, "max": 0.0}

    cost_history.record_session(0.5)
    stats = cost_history.get_percentiles(window_days=30)
    assert stats == {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0, "max": 0.0}


def test_record_bad_write_doesnt_crash(monkeypatch, tmp_path):
    bad_path = tmp_path / "does_not_exist" / "cost_history.jsonl"
    monkeypatch.setattr(cost_history, "COST_HISTORY_PATH", bad_path)
    # Should not raise
    cost_history.record_session(0.5)


def test_load_history_skips_malformed_json():
    today = str(date.today())
    with open(cost_history.COST_HISTORY_PATH, "w") as f:
        f.write("not json\n")
        f.write(json.dumps({"date": today, "session_usd": 0.1, "providers": {}}) + "\n")
    entries = cost_history.load_history()
    assert len(entries) == 1
    assert entries[0]["session_usd"] == 0.1


def test_load_history_skips_invalid_date():
    today = str(date.today())
    with open(cost_history.COST_HISTORY_PATH, "w") as f:
        f.write(json.dumps({"date": "not-a-date", "session_usd": 0.05, "providers": {}}) + "\n")
        f.write(json.dumps({"date": today, "session_usd": 0.1, "providers": {}}) + "\n")
    entries = cost_history.load_history()
    assert len(entries) == 1
    assert entries[0]["session_usd"] == 0.1


def test_load_history_skips_empty_lines():
    today = str(date.today())
    with open(cost_history.COST_HISTORY_PATH, "w") as f:
        f.write("\n")
        f.write(json.dumps({"date": today, "session_usd": 0.1, "providers": {}}) + "\n")
        f.write("   \n")
    entries = cost_history.load_history()
    assert len(entries) == 1


def test_load_history_sorted_by_date():
    d1 = str(date.today() - timedelta(days=2))
    d2 = str(date.today() - timedelta(days=1))
    d3 = str(date.today())
    with open(cost_history.COST_HISTORY_PATH, "w") as f:
        for d, cost in [(d3, 0.3), (d1, 0.1), (d2, 0.2)]:
            f.write(json.dumps({"date": d, "session_usd": cost, "providers": {}}) + "\n")
    entries = cost_history.load_history()
    dates = [e["date"] for e in entries]
    assert dates == sorted(dates)


def test_record_session_oserror(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _raise)
    # Should not raise
    cost_history.record_session(0.5)


def test_load_history_skips_non_dict_json():
    today = str(date.today())
    with open(cost_history.COST_HISTORY_PATH, "w") as f:
        f.write("[1, 2, 3]\n")
        f.write(json.dumps({"date": today, "session_usd": 0.1, "providers": {}}) + "\n")
    entries = cost_history.load_history()
    assert len(entries) == 1
    assert entries[0]["session_usd"] == 0.1


def test_load_history_oserror(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("permission denied")

    today = str(date.today())
    with open(cost_history.COST_HISTORY_PATH, "w") as f:
        f.write(json.dumps({"date": today, "session_usd": 0.1, "providers": {}}) + "\n")
    monkeypatch.setattr("builtins.open", _raise)
    assert cost_history.load_history() == []


def test_get_percentiles_single_entry():
    cost_history.record_session(0.5)
    stats = cost_history.get_percentiles(window_days=30)
    # fewer than 2 data points → all-zero dict
    assert stats == {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0, "max": 0.0}


def test_get_percentiles_two_entries():
    cost_history.record_session(0.1)
    cost_history.record_session(0.2)
    stats = cost_history.get_percentiles(window_days=30)
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(0.15)
