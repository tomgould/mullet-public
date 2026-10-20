"""Tests for mullet.hud.state — read_state_file and boot-time staleness detection."""
import json
import time
from unittest.mock import patch


from mullet.hud.state import read_state_file, _boot_time


# --- _boot_time ---

def test_boot_time_returns_float():
    bt = _boot_time()
    assert isinstance(bt, float)
    assert bt > 0


def test_boot_time_is_before_now():
    assert _boot_time() < time.time()


def test_boot_time_fallback_on_missing_proc(tmp_path):
    with patch("mullet.hud.state.Path") as mock_path_cls:
        mock_path_cls.return_value.read_text.side_effect = OSError("no /proc")
        result = _boot_time()
    # Fallback returns 0.0 so pre-boot check never rejects on non-Linux
    assert result == 0.0


# --- read_state_file ---

def test_returns_empty_when_missing(tmp_path):
    assert read_state_file(tmp_path / "nonexistent.json") == {}


def test_returns_data_for_fresh_file(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"status": "go"}))
    result = read_state_file(p)
    assert result == {"status": "go"}


def test_returns_empty_for_stale_file(tmp_path):
    p = tmp_path / "stale.json"
    p.write_text(json.dumps({"status": "throttle"}))
    # Back-date mtime beyond max_age
    old = time.time() - 9000
    import os
    os.utime(p, (old, old))
    assert read_state_file(p, max_age=1800) == {}


def test_returns_empty_for_pre_boot_file(tmp_path):
    p = tmp_path / "pre_boot.json"
    p.write_text(json.dumps({"status": "throttle"}))
    # Simulate boot time 60s ago; backdate file to 120s ago
    now = time.time()
    fake_boot = now - 60
    old_mtime = now - 120
    import os
    os.utime(p, (old_mtime, old_mtime))
    with patch("mullet.hud.state._boot_time", return_value=fake_boot):
        result = read_state_file(p)
    assert result == {}


def test_accepts_post_boot_file(tmp_path):
    p = tmp_path / "fresh.json"
    p.write_text(json.dumps({"status": "go"}))
    # Boot was 120s ago; file written 60s ago — post-boot, should be accepted
    now = time.time()
    fake_boot = now - 120
    recent_mtime = now - 60
    import os
    os.utime(p, (recent_mtime, recent_mtime))
    with patch("mullet.hud.state._boot_time", return_value=fake_boot):
        result = read_state_file(p)
    assert result == {"status": "go"}


def test_returns_empty_for_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert read_state_file(p) == {}


def test_custom_max_age(tmp_path):
    p = tmp_path / "short.json"
    p.write_text(json.dumps({"x": 1}))
    import os
    # 10s old, max_age=5 → stale
    os.utime(p, (time.time() - 10, time.time() - 10))
    assert read_state_file(p, max_age=5) == {}
    # Same file, max_age=30 → fresh
    assert read_state_file(p, max_age=30) == {"x": 1}


# --- build_context budget_status ---

from unittest.mock import MagicMock


def test_build_context_populates_budget_status(tmp_path):
    from mullet.hud.state import build_context

    mock_budget = MagicMock()
    mock_budget.status.value = "throttle"
    mock_budget.five_hour_pct = 65.0
    mock_budget.weekly_pct = 30.0
    mock_budget.session_pressure = 0.45
    mock_budget.weekly_pressure = 0.2
    mock_budget.reset_time = None

    with (
        patch("mullet.hud.state._budget_cache", None),
        patch("mullet.hud.state._spend_cache", None),
        patch("mullet.hud.guard.check_budget", return_value=mock_budget),
        patch("mullet.hud.state.get_git_info", return_value=(None, False)),
    ):
        ctx = build_context(str(tmp_path))

    assert ctx.budget_status == "throttle"
    assert ctx.five_hour_pct == 65.0
    assert ctx.pressure_5h == 0.45


def test_build_context_budget_status_defaults_to_empty_on_failure(tmp_path):
    from mullet.hud.state import build_context

    with (
        patch("mullet.hud.state._budget_cache", None),
        patch("mullet.hud.state._spend_cache", None),
        patch("mullet.hud.guard.check_budget", side_effect=Exception("API down")),
        patch("mullet.hud.state.get_git_info", return_value=(None, False)),
    ):
        ctx = build_context(str(tmp_path))

    assert ctx.budget_status == ""
    assert ctx.five_hour_pct is None


class TestHudStateMissingLines:
    """Cover lines 145-146 in state.py — provider spend fetch failure."""

    def test_build_context_provider_spend_failure_logged(self, tmp_path):
        """When provider_spend_5h raises, build_context logs warning and continues."""
        from mullet.hud.state import build_context
        from unittest.mock import MagicMock
        from datetime import datetime, timezone, timedelta

        mock_budget = MagicMock()
        mock_budget.status.value = "go"
        mock_budget.five_hour_pct = 10.0
        mock_budget.weekly_pct = 20.0
        mock_budget.session_pressure = 0.1
        mock_budget.weekly_pressure = 0.1
        mock_budget.reset_time = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

        with (
            patch("mullet.hud.state._budget_cache", None),
            patch("mullet.hud.state._spend_cache", None),
            patch("mullet.hud.guard.check_budget", return_value=mock_budget),
            patch("mullet.hud.state.get_git_info", return_value=(None, False)),
            patch("mullet.core.routing_history.provider_spend_5h", side_effect=Exception("disk full")),
        ):
            ctx = build_context(str(tmp_path))

        assert ctx.budget_status == "go"
        assert ctx.provider_costs_5h == {}
        assert ctx.rate_limit_reset_s > 0
