import inspect
import json
import os
import stat
import time
from unittest.mock import patch

import pytest

import mullet.hud.usage as usage_mod
from mullet.hud.cli import HudDeps, main
from mullet.hud.stdin_limits import fill_missing_windows, parse_stdin_limits
from mullet.hud.types import HudRenderContext
from mullet.hud.usage import RateLimits, fetch_usage, parse_resets_at, record_usage


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


class TestRecordAndFetch:
    @pytest.fixture(autouse=True)
    def _cache(self, monkeypatch, tmp_path):
        self.cache = tmp_path / "usage-cache.json"
        monkeypatch.setattr(usage_mod, "_CACHE_FILE", self.cache)

    def _write(self, age: float, limits: dict) -> None:
        self.cache.write_text(json.dumps({"ts": time.time() - age, "limits": limits}))

    def test_roundtrip(self):
        limits = RateLimits(
            five_hour_pct=42.0, five_hour_resets="2099-01-15T12:00:00+00:00",
            weekly_pct=17.5, weekly_resets="2099-01-20T00:00:00+00:00",
            sonnet_weekly_pct=3.0, has_sonnet_limit=True,
        )
        record_usage(limits)
        assert fetch_usage() == limits

    def test_cache_is_private_to_the_user(self):
        record_usage(RateLimits(five_hour_pct=1.0))
        assert stat.S_IMODE(os.stat(self.cache).st_mode) == 0o600

    def test_no_cache_means_unknown(self):
        assert fetch_usage() is None

    def test_recent_figures_are_returned(self):
        self._write(usage_mod._MAX_AGE - 60, {"five_hour_pct": 80.0})
        assert fetch_usage().five_hour_pct == 80.0

    def test_stale_figures_are_ignored(self):
        self._write(usage_mod._MAX_AGE + 60, {"five_hour_pct": 80.0})
        assert fetch_usage() is None

    def test_corrupt_cache_means_unknown(self):
        self.cache.write_text("bad json")
        assert fetch_usage() is None

    def test_write_failure_is_logged_not_raised(self, monkeypatch, tmp_path):
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        monkeypatch.setattr(usage_mod, "_CACHE_FILE", blocked)
        record_usage(RateLimits())
        assert os.listdir(tmp_path) == ["blocked"]

    def test_record_replaces_the_previous_figures(self):
        record_usage(RateLimits(five_hour_pct=10.0))
        record_usage(RateLimits(five_hour_pct=20.0))
        assert fetch_usage().five_hour_pct == 20.0
        assert [p.name for p in self.cache.parent.iterdir()] == [self.cache.name]


class TestNoClaudeCredentials:
    """Mullet must never read Claude's login or call the usage API itself."""

    def test_fetch_makes_no_network_call(self, monkeypatch, tmp_path):
        monkeypatch.setattr(usage_mod, "_CACHE_FILE", tmp_path / "usage-cache.json")
        with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            assert fetch_usage() is None

    def test_credentials_file_is_never_opened(self, monkeypatch, tmp_path):
        creds = tmp_path / ".claude" / ".credentials.json"
        creds.parent.mkdir()
        creds.write_text('{"claudeAiOauth": {"accessToken": "x"}}')
        creds.chmod(0)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(usage_mod, "_CACHE_FILE", tmp_path / "usage-cache.json")
        opened = []
        real_open = open

        def spy(path, *args, **kwargs):
            opened.append(str(path))
            return real_open(path, *args, **kwargs)

        # Path.read_text opens through io.open, so watch both names.
        with patch("builtins.open", spy), patch("io.open", spy):
            fetch_usage()
            record_usage(RateLimits())
            fetch_usage()
        assert not any(".claude" in p for p in opened)

    def test_module_has_no_credential_code(self):
        source = inspect.getsource(usage_mod)
        for needle in (".credentials", "accessToken", "oauth", "urlopen"):
            assert needle not in source


class TestResetWindows:
    @pytest.fixture(autouse=True)
    def _cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(usage_mod, "_CACHE_FILE", tmp_path / "usage-cache.json")

    def test_a_window_past_its_reset_reads_as_zero(self):
        record_usage(RateLimits(
            five_hour_pct=96.0, five_hour_resets="2020-01-01T00:00:00+00:00",
            weekly_pct=40.0, weekly_resets="2099-01-01T00:00:00+00:00",
        ))
        limits = fetch_usage()
        assert limits.five_hour_pct == 0.0
        assert limits.weekly_pct == 40.0

    def test_a_window_without_a_reset_time_is_kept(self):
        record_usage(RateLimits(five_hour_pct=50.0))
        assert fetch_usage().five_hour_pct == 50.0


class TestFillMissingWindows:
    previous = RateLimits(
        five_hour_pct=30.0, five_hour_resets="2099-01-01T05:00:00+00:00",
        weekly_pct=92.0, weekly_resets="2099-01-07T00:00:00+00:00",
    )

    def test_a_partial_tick_keeps_the_last_weekly_figure(self):
        stdin = {"rate_limits": {"five_hour": {"used_percentage": 40, "resets_at": 4070912400}}}
        limits = fill_missing_windows(parse_stdin_limits(stdin), stdin, self.previous)
        assert limits.five_hour_pct == 40.0
        assert limits.weekly_pct == 92.0
        assert limits.weekly_resets == self.previous.weekly_resets

    def test_a_partial_tick_keeps_the_last_five_hour_figure(self):
        stdin = {"rate_limits": {"seven_day": {"used_percentage": 93}}}
        limits = fill_missing_windows(parse_stdin_limits(stdin), stdin, self.previous)
        assert limits.five_hour_pct == 30.0
        assert limits.weekly_pct == 93.0

    def test_a_full_tick_replaces_both(self):
        stdin = {"rate_limits": {"five_hour": {"used_percentage": 1}, "seven_day": {"used_percentage": 2}}}
        limits = fill_missing_windows(parse_stdin_limits(stdin), stdin, self.previous)
        assert (limits.five_hour_pct, limits.weekly_pct) == (1.0, 2.0)

    def test_nothing_to_keep_without_previous_figures(self):
        stdin = {"rate_limits": {"five_hour": {"used_percentage": 40}}}
        limits = fill_missing_windows(parse_stdin_limits(stdin), stdin, None)
        assert limits.weekly_pct == 0.0


class TestHudRecordsStdinLimits:
    def _deps(self, stdin_limits, recorded):
        return HudDeps(
            read_stdin=lambda: {"cwd": "/tmp"},
            parse_stdin_limits=lambda data: stdin_limits,
            fetch_usage_fallback=lambda: None,
            record_usage=recorded.append,
            scan_processes=lambda: [],
            build_context=lambda cwd: HudRenderContext(cwd=cwd),
            render=lambda ctx, cfg: "",
            get_terminal_cols=lambda: 120,
            emit=lambda line: None,
        )

    def test_limits_from_claude_code_are_recorded(self):
        recorded = []
        limits = RateLimits(five_hour_pct=12.0)
        main(self._deps(limits, recorded))
        assert recorded == [limits]

    def test_partial_tick_records_the_merged_copy_but_shows_its_own(self):
        recorded, shown = [], []
        stdin = {"cwd": "/tmp", "rate_limits": {"five_hour": {"used_percentage": 12}}}
        deps = self._deps(None, recorded)
        deps.read_stdin = lambda: stdin
        deps.parse_stdin_limits = parse_stdin_limits
        deps.fetch_usage_fallback = lambda: RateLimits(weekly_pct=92.0)
        deps.render = lambda ctx, cfg: shown.append(ctx.weekly_pct) or ""
        main(deps)
        assert recorded[0].five_hour_pct == 12.0
        assert recorded[0].weekly_pct == 92.0
        assert shown == [0.0]

    def test_nothing_recorded_without_limits(self):
        recorded = []
        main(self._deps(None, recorded))
        assert recorded == []

    def test_record_failure_does_not_break_the_hud(self):
        def boom(limits):
            raise OSError("disk full")

        deps = self._deps(RateLimits(five_hour_pct=12.0), [])
        deps.record_usage = boom
        main(deps)


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
