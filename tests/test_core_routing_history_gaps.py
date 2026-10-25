"""Gap tests for mullet.core.routing_history — targets uncovered lines."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.core.routing_history import (
    RoutingOutcome,
    append_decision_log,
    record,
    iter_recent,
    success_rate,
    _percentile,
    summary_by_provider,
    provider_spend_5h,
    daily_cost_summary,
    history_path,
)


# ---------------------------------------------------------------------------
# append_decision_log  (lines 123-124)
# ---------------------------------------------------------------------------

def test_append_decision_log_oserror_logged(tmp_path, caplog):
    """Lines 123-124: OSError during append is logged as debug."""
    with caplog.at_level("DEBUG", logger="mullet.core.routing_history"):
        with patch("builtins.open", side_effect=OSError("disk full")):
            append_decision_log(
                provider="kimi",
                model="k2",
                tier="STANDARD",
                reasons=["r1"],
                task_preview="task",
                base=tmp_path,
            )
    assert "append_decision_log failed" in caplog.text


# ---------------------------------------------------------------------------
# record  (lines 201-202)
# ---------------------------------------------------------------------------

def test_record_oserror_logged(tmp_path, caplog):
    """Lines 201-202: OSError during record is logged as warning."""
    outcome = _make_routing_outcome()
    with caplog.at_level("WARNING", logger="mullet.core.routing_history"):
        with patch("builtins.open", side_effect=OSError("disk full")):
            record(outcome, base=tmp_path)
    assert "failed to record outcome" in caplog.text


# ---------------------------------------------------------------------------
# iter_recent  (lines 220-222, 228)
# ---------------------------------------------------------------------------

def test_iter_recent_oserror_on_read(tmp_path, caplog):
    """Lines 220-222: OSError reading history yields nothing."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text("{}")
    with caplog.at_level("WARNING", logger="mullet.core.routing_history"):
        with patch.object(Path, "read_text", side_effect=OSError("denied")):
            results = list(iter_recent(base=tmp_path))
    assert results == []
    assert "failed to read history" in caplog.text


def test_iter_recent_skips_blank_lines(tmp_path):
    """Line 228: blank lines are skipped."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    entry = _outcome_dict()
    hp.write_text(json.dumps(entry) + "\n\n   \n" + json.dumps(entry) + "\n")
    results = list(iter_recent(base=tmp_path))
    assert len(results) == 2


# ---------------------------------------------------------------------------
# success_rate  (lines 255, 264-266)
# ---------------------------------------------------------------------------

def test_success_rate_skips_blank_lines(tmp_path):
    """Line 255: blank lines in history are skipped."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    entry = _outcome_dict()
    hp.write_text("\n\n" + json.dumps(entry) + "\n")
    rate = success_rate(entry["task_signature"], entry["provider"], base=tmp_path, min_samples=1)
    assert rate == 1.0


def test_success_rate_oserror_on_read(tmp_path, caplog):
    """Lines 264-266: OSError reading for success_rate returns None."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text("{}")
    with caplog.at_level("WARNING", logger="mullet.core.routing_history"):
        with patch.object(Path, "read_text", side_effect=OSError("denied")):
            rate = success_rate("sig", "kimi", base=tmp_path)
    assert rate is None
    assert "failed to read for success_rate" in caplog.text


# ---------------------------------------------------------------------------
# _percentile  (lines 277, 284)
# ---------------------------------------------------------------------------

def test_percentile_empty_list():
    """Line 277: empty values → 0.0."""
    assert _percentile([], 50) == 0.0


def test_percentile_hi_equals_n():
    """Line 284: hi >= n → last element."""
    assert _percentile([1.0, 2.0, 3.0], 100) == 3.0


# ---------------------------------------------------------------------------
# summary_by_provider  (lines 303-305, 314-315)
# ---------------------------------------------------------------------------

def test_summary_by_provider_oserror_on_read(tmp_path, caplog):
    """Lines 303-305: OSError reading for summary returns {}."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text("{}")
    with caplog.at_level("WARNING", logger="mullet.core.routing_history"):
        with patch.object(Path, "read_text", side_effect=OSError("denied")):
            result = summary_by_provider(base=tmp_path)
    assert result == {}
    assert "failed to read for summary" in caplog.text


def test_summary_by_provider_skips_malformed_json(tmp_path, caplog):
    """Lines 314-315: JSONDecodeError lines are skipped."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    good = _outcome_dict(provider="kimi")
    hp.write_text("not json\n" + json.dumps(good) + "\n")
    with caplog.at_level("WARNING", logger="mullet.core.routing_history"):
        result = summary_by_provider(base=tmp_path)
    assert result["kimi"]["calls"] == 1


# ---------------------------------------------------------------------------
# provider_spend_5h  (lines 381-382, 388, 391-392, 395, 399-400)
# ---------------------------------------------------------------------------

def test_provider_spend_5h_oserror_on_read(tmp_path):
    """Lines 381-382: OSError reading history returns {}."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text("{}")
    with patch.object(Path, "read_text", side_effect=OSError("denied")):
        result = provider_spend_5h(base=tmp_path)
    assert result == {}


def test_provider_spend_5h_skips_blank_lines(tmp_path):
    """Line 388: blank lines are skipped."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    entry = _outcome_dict(cost_usd=1.23)
    hp.write_text("\n\n" + json.dumps(entry) + "\n")
    result = provider_spend_5h(base=tmp_path)
    assert result.get("kimi", 0.0) == pytest.approx(1.23)


def test_provider_spend_5h_skips_malformed_json(tmp_path):
    """Lines 391-392: JSONDecodeError lines are skipped."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    good = _outcome_dict(cost_usd=2.0)
    hp.write_text("bad json\n" + json.dumps(good) + "\n")
    result = provider_spend_5h(base=tmp_path)
    assert result.get("kimi", 0.0) == pytest.approx(2.0)


def test_provider_spend_5h_skips_empty_timestamp(tmp_path):
    """Line 395: missing timestamp → skip."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    entry = _outcome_dict(cost_usd=3.0)
    entry["timestamp"] = ""
    hp.write_text(json.dumps(entry) + "\n")
    result = provider_spend_5h(base=tmp_path)
    assert result == {}


def test_provider_spend_5h_skips_bad_timestamp(tmp_path):
    """Lines 399-400: unparseable timestamp → skip."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    entry = _outcome_dict(cost_usd=4.0)
    entry["timestamp"] = "not-a-date"
    hp.write_text(json.dumps(entry) + "\n")
    result = provider_spend_5h(base=tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# daily_cost_summary  (lines 442-443, 451, 454-455, 458, 462-463, 469)
# ---------------------------------------------------------------------------

def test_daily_cost_summary_oserror_on_read(tmp_path):
    """Lines 442-443: OSError reading history returns {}."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text("{}")
    with patch.object(Path, "read_text", side_effect=OSError("denied")):
        result = daily_cost_summary(base=tmp_path)
    assert result == {}


def test_daily_cost_summary_skips_blank_lines(tmp_path):
    """Line 451: blank lines are skipped."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    entry = _outcome_dict(cost_usd=1.0)
    hp.write_text("\n\n" + json.dumps(entry) + "\n")
    result = daily_cost_summary(base=tmp_path, limit_days=365)
    assert result["count"] == 1


def test_daily_cost_summary_skips_malformed_json(tmp_path):
    """Lines 454-455: JSONDecodeError lines are skipped."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    good = _outcome_dict(cost_usd=2.0)
    hp.write_text("bad json\n" + json.dumps(good) + "\n")
    result = daily_cost_summary(base=tmp_path, limit_days=365)
    assert result["count"] == 1


def test_daily_cost_summary_skips_empty_timestamp(tmp_path):
    """Line 458: missing timestamp → skip."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    entry = _outcome_dict(cost_usd=3.0)
    entry["timestamp"] = ""
    hp.write_text(json.dumps(entry) + "\n")
    result = daily_cost_summary(base=tmp_path, limit_days=365)
    assert result == {}


def test_daily_cost_summary_skips_bad_date(tmp_path):
    """Lines 462-463: unparseable date → skip."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    entry = _outcome_dict(cost_usd=4.0)
    entry["timestamp"] = "2026-13-45T99:99:99+00:00"
    hp.write_text(json.dumps(entry) + "\n")
    result = daily_cost_summary(base=tmp_path, limit_days=365)
    assert result == {}


def test_daily_cost_summary_empty_after_filtering(tmp_path):
    """Line 469: all entries filtered out → {}."""
    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    entry = _outcome_dict(cost_usd=5.0)
    entry["timestamp"] = "2020-01-01T00:00:00+00:00"
    hp.write_text(json.dumps(entry) + "\n")
    result = daily_cost_summary(base=tmp_path, limit_days=1)
    assert result == {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_routing_outcome(
    provider: str = "kimi",
    model: str = "kimi-k2",
    success: bool = True,
    cost_usd: float = 0.0,
) -> RoutingOutcome:
    from datetime import datetime, timezone
    return RoutingOutcome(
        timestamp=datetime.now(timezone.utc).isoformat(),
        task_signature="sig",
        task_preview="test",
        provider=provider,
        model=model,
        tier="STANDARD",
        success=success,
        error_class=None,
        tokens_in=100,
        tokens_out=50,
        wall_time_ms=100,
        cost_usd=cost_usd,
        tags=[],
    )


def _outcome_dict(
    provider: str = "kimi",
    model: str = "kimi-k2",
    cost_usd: float = 0.0,
) -> dict:
    from datetime import datetime, timezone
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_signature": "sig",
        "task_preview": "test",
        "provider": provider,
        "model": model,
        "tier": "STANDARD",
        "success": True,
        "error_class": None,
        "tokens_in": 100,
        "tokens_out": 50,
        "wall_time_ms": 100,
        "cost_usd": cost_usd,
        "tags": [],
    }
