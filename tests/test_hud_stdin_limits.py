from datetime import datetime, timezone


from mullet.hud.stdin_limits import _clamp_pct, _unix_to_iso, parse_stdin_limits
from mullet.hud.usage import RateLimits


class TestClampPct:
    def test_valid_float(self):
        assert _clamp_pct(50.0) == 50.0

    def test_valid_int(self):
        assert _clamp_pct(75) == 75.0

    def test_clamps_above_100(self):
        assert _clamp_pct(150.0) == 100.0

    def test_clamps_below_0(self):
        assert _clamp_pct(-10.0) == 0.0

    def test_nan_returns_none(self):
        assert _clamp_pct(float("nan")) is None

    def test_non_numeric_returns_none(self):
        assert _clamp_pct("foo") is None
        assert _clamp_pct(None) is None
        assert _clamp_pct([]) is None

    def test_boundary_100(self):
        assert _clamp_pct(100) == 100.0

    def test_boundary_0(self):
        assert _clamp_pct(0) == 0.0


class TestUnixToIso:
    def test_valid_timestamp(self):
        ts = 1700000000.0
        result = _unix_to_iso(ts)
        expected = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        assert result == expected

    def test_zero_returns_empty(self):
        assert _unix_to_iso(0) == ""

    def test_negative_returns_empty(self):
        assert _unix_to_iso(-1) == ""

    def test_non_numeric_returns_empty(self):
        assert _unix_to_iso("foo") == ""
        assert _unix_to_iso(None) == ""

    def test_overflow_returns_empty(self):
        assert _unix_to_iso(1e30) == ""


class TestParseStdinLimits:
    def test_valid_data(self):
        data = {
            "rate_limits": {
                "five_hour": {"used_percentage": 50.0, "resets_at": 1700000000},
                "seven_day": {"used_percentage": 30.0, "resets_at": 1700100000},
            }
        }
        result = parse_stdin_limits(data)
        assert isinstance(result, RateLimits)
        assert result.five_hour_pct == 50.0
        assert result.weekly_pct == 30.0
        assert result.five_hour_resets != ""
        assert result.weekly_resets != ""

    def test_missing_rate_limits(self):
        assert parse_stdin_limits({}) is None

    def test_non_dict_input(self):
        assert parse_stdin_limits(None) is None
        assert parse_stdin_limits([]) is None
        assert parse_stdin_limits("foo") is None

    def test_no_percentages_returns_none(self):
        data = {
            "rate_limits": {
                "five_hour": {"resets_at": 1700000000},
                "seven_day": {"resets_at": 1700100000},
            }
        }
        assert parse_stdin_limits(data) is None

    def test_only_five_hour(self):
        data = {
            "rate_limits": {
                "five_hour": {"used_percentage": 60.0, "resets_at": 1700000000},
            }
        }
        result = parse_stdin_limits(data)
        assert result.five_hour_pct == 60.0
        assert result.weekly_pct == 0.0

    def test_invalid_percentage_skipped(self):
        data = {
            "rate_limits": {
                "five_hour": {"used_percentage": "bad", "resets_at": 1700000000},
                "seven_day": {"used_percentage": 25.0},
            }
        }
        result = parse_stdin_limits(data)
        assert result.weekly_pct == 25.0
        assert result.five_hour_pct == 0.0

    def test_clamps_percentage(self):
        data = {
            "rate_limits": {
                "five_hour": {"used_percentage": 200.0},
            }
        }
        result = parse_stdin_limits(data)
        assert result.five_hour_pct == 100.0


class TestClampPctMissingLines:
    """Cover lines 19-20 in stdin_limits.py — float() conversion failure."""

    def test_float_conversion_raises_returns_none(self):
        """Values where float() raises should return None."""
        # Most non-numeric types are caught by isinstance, but some objects
        # pass isinstance(int, float) check yet fail float(). We mock to force it.
        from unittest.mock import patch
        with patch("builtins.float", side_effect=TypeError("no float")):
            assert _clamp_pct(42) is None

    def test_valueerror_on_float_returns_none(self):
        """Values where float() raises ValueError should return None."""
        from unittest.mock import patch
        with patch("builtins.float", side_effect=ValueError("bad float")):
            assert _clamp_pct(42) is None
