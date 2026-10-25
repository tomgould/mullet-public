"""Coverage-gap tests for mullet/core/providers/cost_history.py.

Targets lines not hit by tests/test_cost_history.py.
"""

import json
from datetime import date
from unittest.mock import patch

import pytest

from mullet.core.providers import cost_history


@pytest.fixture(autouse=True)
def _patch_path(monkeypatch, tmp_path):
    """Redirect COST_HISTORY_PATH to a temp file for every test."""
    monkeypatch.setattr(cost_history, "COST_HISTORY_PATH", tmp_path / "cost_history.jsonl")


# ---------------------------------------------------------------------------
# record_session OSError — lines 41-42
# ---------------------------------------------------------------------------

def test_record_session_oserror_on_fsync_is_logged(caplog):
    """Line 41-42: OSError during write/fsync is caught and logged."""
    with patch("mullet.core.providers.cost_history.os.fsync", side_effect=OSError("disk full")):
        cost_history.record_session(0.5)
    assert any("failed to write entry" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# load_history non-dict object — line 69
# ---------------------------------------------------------------------------

def test_load_history_skips_non_dict_object():
    today = str(date.today())
    with open(cost_history.COST_HISTORY_PATH, "w") as f:
        f.write(json.dumps({"date": today, "session_usd": 0.1, "providers": {}}) + "\n")
        f.write("[1, 2, 3]\n")
        f.write(json.dumps({"date": today, "session_usd": 0.2, "providers": {}}) + "\n")
    entries = cost_history.load_history()
    assert len(entries) == 2
    assert entries[0]["session_usd"] == 0.1
    assert entries[1]["session_usd"] == 0.2


# ---------------------------------------------------------------------------
# load_history OSError — lines 79-81
# ---------------------------------------------------------------------------

def test_load_history_oserror_on_read_returns_empty():
    """Line 79-81: OSError when opening/reading file returns [] after logging."""
    # Create the file first so exists() passes, then patch open to fail
    cost_history.COST_HISTORY_PATH.touch()
    with patch("builtins.open", side_effect=OSError("permission denied")):
        entries = cost_history.load_history()
    assert entries == []


# ---------------------------------------------------------------------------
# get_percentiles edge cases
# ---------------------------------------------------------------------------

def test_get_percentiles_single_entry():
    """Only one valid entry → returns all zeros because n < 2."""
    cost_history.record_session(0.5)
    stats = cost_history.get_percentiles(window_days=30)
    assert stats == {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0, "max": 0.0}


def test_get_percentiles_exactly_two_entries():
    """Two entries → p95 uses interpolation."""
    cost_history.record_session(0.1)
    cost_history.record_session(0.2)
    stats = cost_history.get_percentiles(window_days=30)
    assert stats["n"] == 2
    assert stats["p50"] == pytest.approx(0.15, abs=0.001)
    assert stats["p95"] == pytest.approx(0.195, abs=0.001)


def test_get_percentiles_out_of_range_date():
    """Entry with impossible date string is skipped via ValueError."""
    today = str(date.today())
    with open(cost_history.COST_HISTORY_PATH, "w") as f:
        f.write(json.dumps({"date": "9999-99-99", "session_usd": 0.5, "providers": {}}) + "\n")
        f.write(json.dumps({"date": today, "session_usd": 0.2, "providers": {}}) + "\n")
    entries = cost_history.load_history()
    assert len(entries) == 1
    assert entries[0]["session_usd"] == 0.2
