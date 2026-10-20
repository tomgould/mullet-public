import json
from unittest.mock import MagicMock, patch

import pytest

from mullet.hud.usage import RateLimits, fetch_usage, parse_resets_at


class TestRateLimits:
    def test_defaults(self):
        r = RateLimits()
        assert r.five_hour_pct == 0.0
        assert r.five_hour_resets == ""
        assert r.weekly_pct == 0.0
        assert r.weekly_resets == ""
        assert r.sonnet_weekly_pct == 0.0
        assert r.sonnet_weekly_resets == ""
        assert r.has_sonnet_limit is False

    def test_custom_values(self):
        r = RateLimits(five_hour_pct=50.0, weekly_pct=30.0, sonnet_weekly_pct=10.0)
        assert r.five_hour_pct == 50.0
        assert r.weekly_pct == 30.0
        assert r.sonnet_weekly_pct == 10.0

    def test_accepts_has_sonnet_limit_true(self):
        r = RateLimits(has_sonnet_limit=True)
        assert r.has_sonnet_limit is True


class TestFetchUsage:
    @pytest.fixture(autouse=True)
    def _patch_cache_and_reset_state(self, monkeypatch, tmp_path):
        """Patch cache file to a temp path and reset module-level state."""
        import mullet.hud.usage as usage_mod

        monkeypatch.setattr(usage_mod, "_CACHE_FILE", tmp_path / "test-usage-cache.json")
        monkeypatch.setattr(usage_mod, "_last_fetch_attempt", 0.0)

    def _make_api_response(self, seven_day_sonnet=None):
        """Build a JSON API response blob."""
        data = {
            "five_hour": {"utilization": 30.0, "resets_at": "2026-01-15T12:00:00Z"},
            "seven_day": {"utilization": 25.0, "resets_at": "2026-01-16T00:00:00Z"},
        }
        if seven_day_sonnet is not None:
            data["seven_day_sonnet"] = seven_day_sonnet
        return json.dumps(data).encode()

    def test_sets_has_sonnet_limit_true_when_present(self, monkeypatch):
        import mullet.hud.usage as usage_mod

        monkeypatch.setattr(usage_mod, "_get_access_token", lambda: "fake-token")

        mock_resp = MagicMock()
        mock_resp.read.return_value = self._make_api_response(
            seven_day_sonnet={"utilization": 15.0, "resets_at": "2026-01-16T06:00:00Z"}
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_usage()

        assert result is not None
        assert result.has_sonnet_limit is True
        assert result.sonnet_weekly_pct == 15.0

    def test_sets_has_sonnet_limit_false_when_null(self, monkeypatch):
        import mullet.hud.usage as usage_mod

        monkeypatch.setattr(usage_mod, "_get_access_token", lambda: "fake-token")

        mock_resp = MagicMock()
        mock_resp.read.return_value = self._make_api_response(seven_day_sonnet=None)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_usage()

        assert result is not None
        assert result.has_sonnet_limit is False
        assert result.sonnet_weekly_pct == 0.0

    def test_parses_sonnet_weekly_pct_from_seven_day_sonnet(self, monkeypatch):
        import mullet.hud.usage as usage_mod

        monkeypatch.setattr(usage_mod, "_get_access_token", lambda: "fake-token")

        mock_resp = MagicMock()
        mock_resp.read.return_value = self._make_api_response(
            seven_day_sonnet={"utilization": 42.0, "resets_at": "2026-01-17T00:00:00Z"}
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_usage()

        assert result is not None
        assert result.sonnet_weekly_pct == 42.0

    def test_cache_roundtrip_preserves_has_sonnet_limit(self, monkeypatch):
        import mullet.hud.usage as usage_mod

        monkeypatch.setattr(usage_mod, "_get_access_token", lambda: "fake-token")

        mock_resp = MagicMock()
        mock_resp.read.return_value = self._make_api_response(
            seven_day_sonnet={"utilization": 20.0, "resets_at": "2026-01-16T00:00:00Z"}
        )
        # First call — hits API and writes cache
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result1 = fetch_usage()

        assert result1 is not None
        assert result1.has_sonnet_limit is True

        # Second call — should read from cache; patch urlopen to explode so we
        # know the cache path is taken.
        with patch("urllib.request.urlopen", side_effect=Exception("should not be called")):
            result2 = fetch_usage()

        assert result2 is not None
        assert result2.has_sonnet_limit is True
        assert result2.sonnet_weekly_pct == 20.0


class TestParseResetsAt:
    def test_empty_string(self):
        assert parse_resets_at("") is None

    def test_none_input(self):
        assert parse_resets_at(None) is None

    def test_z_suffix(self):
        dt = parse_resets_at("2026-01-15T12:00:00Z")
        assert dt is not None
        assert dt.year == 2026
        assert dt.hour == 12

    def test_offset(self):
        dt = parse_resets_at("2026-01-15T12:00:00+05:00")
        assert dt is not None
        assert dt.utcoffset().total_seconds() == 18000

    def test_invalid_string(self):
        assert parse_resets_at("not-a-date") is None

    def test_malformed_iso(self):
        assert parse_resets_at("2026-13-45T99:00:00Z") is None


class TestUsageMissingLines:
    """Cover previously-missed lines in mullet/hud/usage.py."""

    def test_get_access_token_no_creds_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        import mullet.hud.usage as usage_mod

        token = usage_mod._get_access_token()
        assert token is None

    def test_get_access_token_valid_creds(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        import mullet.hud.usage as usage_mod

        creds = tmp_path / ".claude" / ".credentials.json"
        creds.parent.mkdir(parents=True)
        creds.write_text('{"claudeAiOauth": {"accessToken": "my-secret-token"}}')
        token = usage_mod._get_access_token()
        assert token == "my-secret-token"

    def test_get_access_token_bad_json(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        import mullet.hud.usage as usage_mod

        creds = tmp_path / ".claude" / ".credentials.json"
        creds.parent.mkdir(parents=True)
        creds.write_text("not json")
        token = usage_mod._get_access_token()
        assert token is None

    def test_get_access_token_oserror(self, monkeypatch, tmp_path):
        import os

        monkeypatch.setenv("HOME", str(tmp_path))
        import mullet.hud.usage as usage_mod

        creds = tmp_path / ".claude" / ".credentials.json"
        creds.parent.mkdir(parents=True)
        creds.write_text('{"claudeAiOauth": {"accessToken": "tok"}}')
        os.chmod(str(creds), 0o000)
        try:
            token = usage_mod._get_access_token()
            assert token is None
        finally:
            os.chmod(str(creds), 0o644)

    def test_read_cache_exception_returns_none(self, monkeypatch, tmp_path):
        import mullet.hud.usage as usage_mod

        monkeypatch.setattr(usage_mod, "_CACHE_FILE", tmp_path / "cache.json")
        (tmp_path / "cache.json").write_text("bad json")
        result = usage_mod._read_cache()
        assert result is None

    def test_write_cache_oserror(self, monkeypatch, tmp_path):
        import mullet.hud.usage as usage_mod

        cache = tmp_path / "cache.json"
        cache.mkdir()
        monkeypatch.setattr(usage_mod, "_CACHE_FILE", cache)
        limits = usage_mod.RateLimits()
        # Should not raise
        usage_mod._write_cache(limits)

    def test_fetch_usage_retry_cooldown_returns_cached(self, monkeypatch, tmp_path):
        import time
        import mullet.hud.usage as usage_mod

        monkeypatch.setattr(usage_mod, "_CACHE_FILE", tmp_path / "cache.json")
        # Stale cache
        (tmp_path / "cache.json").write_text(
            '{"ts": ' + str(time.time() - 400) + ', "limits": {}}'
        )
        monkeypatch.setattr(usage_mod, "_last_fetch_attempt", time.time())
        monkeypatch.setattr(usage_mod, "_get_access_token", lambda: "token")
        result = usage_mod.fetch_usage()
        assert result is not None

    def test_fetch_usage_no_token_returns_cached(self, monkeypatch, tmp_path):
        import time
        import mullet.hud.usage as usage_mod

        monkeypatch.setattr(usage_mod, "_CACHE_FILE", tmp_path / "cache.json")
        # Stale cache
        (tmp_path / "cache.json").write_text(
            '{"ts": ' + str(time.time() - 400) + ', "limits": {}}'
        )
        monkeypatch.setattr(usage_mod, "_last_fetch_attempt", 0.0)
        monkeypatch.setattr(usage_mod, "_get_access_token", lambda: None)
        result = usage_mod.fetch_usage()
        assert result is not None

    def test_fetch_usage_api_error_returns_cached(self, monkeypatch, tmp_path):
        import time
        from unittest.mock import patch
        import mullet.hud.usage as usage_mod

        monkeypatch.setattr(usage_mod, "_CACHE_FILE", tmp_path / "cache.json")
        # Stale cache
        (tmp_path / "cache.json").write_text(
            '{"ts": ' + str(time.time() - 400) + ', "limits": {}}'
        )
        monkeypatch.setattr(usage_mod, "_last_fetch_attempt", 0.0)
        monkeypatch.setattr(usage_mod, "_get_access_token", lambda: "token")
        with patch("urllib.request.urlopen", side_effect=OSError("api fail")):
            result = usage_mod.fetch_usage()
        assert result is not None
