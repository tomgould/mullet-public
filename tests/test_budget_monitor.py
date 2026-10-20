"""Tests for budget monitor."""
from unittest.mock import patch
from mullet.hud.budget_monitor import check_headroom, format_advice
from mullet.hud.usage import RateLimits


@patch("mullet.hud.budget_monitor.get_resource_guard")
@patch("mullet.hud.budget_monitor.fetch_usage")
def test_ok_headroom(mock_fetch, mock_guard):
    mock_fetch.return_value = RateLimits(five_hour_pct=30.0, weekly_pct=10.0)
    mock_guard.return_value.get_system_snapshot.return_value = {"load_avg": 1.0}
    result = check_headroom()
    assert result["can_use_anthropic"] is True
    assert "OK" in result["advice"]


@patch("mullet.hud.budget_monitor.get_resource_guard")
@patch("mullet.hud.budget_monitor.fetch_usage")
def test_caution_headroom(mock_fetch, mock_guard):
    mock_fetch.return_value = RateLimits(five_hour_pct=65.0, weekly_pct=20.0)
    mock_guard.return_value.get_system_snapshot.return_value = {}
    result = check_headroom()
    assert result["can_use_anthropic"] is True
    assert "CAUTION" in result["advice"]


@patch("mullet.hud.budget_monitor.get_resource_guard")
@patch("mullet.hud.budget_monitor.fetch_usage")
def test_throttle_headroom(mock_fetch, mock_guard):
    mock_fetch.return_value = RateLimits(five_hour_pct=85.0, weekly_pct=30.0)
    mock_guard.return_value.get_system_snapshot.return_value = {}
    result = check_headroom()
    assert result["can_use_anthropic"] is False
    assert "THROTTLE" in result["advice"]


@patch("mullet.hud.budget_monitor.get_resource_guard")
@patch("mullet.hud.budget_monitor.fetch_usage")
def test_unknown_headroom(mock_fetch, mock_guard):
    mock_fetch.return_value = None
    mock_guard.return_value.get_system_snapshot.return_value = {}
    result = check_headroom()
    assert "UNKNOWN" in result["advice"]


def test_format_advice():
    result = {"advice": "OK — available", "usage_5h_pct": 30.0}
    s = format_advice(result)
    assert "OK" in s
    assert "30%" in s
