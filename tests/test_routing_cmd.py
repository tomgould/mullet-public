"""Tests for mullet.cli.routing_cmd — dispatch and formatting helpers."""
import json
from unittest.mock import patch, MagicMock

import pytest

from mullet.cli.routing_cmd import run, _rate_str
from mullet.core.routing_history import RoutingOutcome
from mullet.core.router.decision import RoutingDecision
from mullet.core.router.complexity import Tier


def _outcome(provider, model, tier="STANDARD", preview="do a task"):
    return RoutingOutcome(
        timestamp="2026-06-21T00:00:00+00:00",
        task_signature="abc1234567890def",
        task_preview=preview,
        provider=provider,
        model=model,
        tier=tier,
        success=True,
        error_class=None,
        tokens_in=0,
        tokens_out=0,
        wall_time_ms=10,
        cost_usd=0.0,
        tags=[],
    )


def _decision(provider, model, tier=Tier.STANDARD, reasons=None):
    return RoutingDecision(
        provider=provider,
        model=model,
        tier=tier,
        reasons=reasons or ["test"],
    )


class TestRateStr:
    def test_high_rate_no_special_prefix(self):
        result = _rate_str(0.95)
        assert "95.0%" in result

    def test_medium_rate(self):
        result = _rate_str(0.75)
        assert "75.0%" in result

    def test_low_rate(self):
        result = _rate_str(0.50)
        assert "50.0%" in result


class TestRunDispatch:
    def test_no_args_shows_usage(self, capsys):
        rc = run([])
        out = capsys.readouterr().out
        assert "Usage" in out
        assert rc == 0

    def test_unknown_action_returns_1(self, capsys):
        rc = run(["doesnotexist"])
        assert rc == 1

    def test_show_with_empty_history(self, capsys):
        with patch("mullet.cli.routing_cmd.summary_by_provider", return_value={}):
            rc = run(["show"])
        out = capsys.readouterr().out
        assert "No routing history" in out
        assert rc == 0

    def test_recent_with_empty_history(self, capsys):
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=iter([])):
            rc = run(["recent"])
        out = capsys.readouterr().out
        assert "No routing history" in out
        assert rc == 0

    def test_signature_requires_task(self, capsys):
        rc = run(["signature"])
        assert rc == 1

    def test_signature_prints_16char_hash(self, capsys):
        with patch("mullet.cli.routing_cmd.make_signature", return_value="abcd1234efgh5678"):
            rc = run(["signature", "refactor the auth module"])
        out = capsys.readouterr().out
        assert "abcd1234efgh5678" in out
        assert rc == 0


class TestLogCommand:
    def test_log_no_file_shows_message(self, tmp_path, monkeypatch, capsys):
        """mullet routing log prints a friendly message when routing.log doesn't exist."""
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = run(["log"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No routing.log" in out or "routing decisions" in out.lower()

    def test_log_tails_existing_file(self, tmp_path, monkeypatch, capsys):
        """mullet routing log shows entries from ~/.mullet/routing.log."""
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        log_lines = [
            "2026-04-28 00:01:00  kimi/kimi-k2.6  STANDARD  forced provider: kimi  |  write unit tests",
            "2026-04-28 00:02:00  gemini/flash  LOW  complexity: low  |  summarise docs",
            "2026-04-28 00:03:00  local/llama  LOW  local-only mode  |  grep for symbol",
        ]
        (mullet_dir / "routing.log").write_text("\n".join(log_lines) + "\n")
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = run(["log"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "kimi" in out
        assert "gemini" in out
        assert "local" in out

    def test_log_tail_limits_output(self, tmp_path, monkeypatch, capsys):
        """--tail N limits displayed lines."""
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        lines = [f"2026-04-28 00:0{i}:00  kimi/k2  STD  r  |  task {i}" for i in range(5)]
        (mullet_dir / "routing.log").write_text("\n".join(lines) + "\n")
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = run(["log", "--tail", "2"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "task 4" in out   # last line
        assert "task 3" in out   # second to last
        assert "task 0" not in out  # excluded by tail

    def test_log_action_in_usage(self, capsys):
        """mullet routing (no args) mentions log in usage."""
        run([])
        out = capsys.readouterr().out
        assert "log" in out


class TestCmdInsights:
    def test_insights_empty_history(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: [])
        from mullet.cli.routing_cmd import run
        rc = run(["insights"])
        out = capsys.readouterr().out
        assert "no data" in out.lower()

    def test_insights_shows_sections_with_data(self, tmp_path, monkeypatch, capsys):
        from mullet.core.routing_history import make_outcome
        outcomes = [
            make_outcome("summarise this doc", "kimi", "kimi-k2", "STANDARD", True, 1000, 200, 800),
            make_outcome("summarise this doc", "kimi", "kimi-k2", "STANDARD", True, 1100, 210, 850),
            make_outcome("analyse code", "local", "llama3", "LOW", True, 500, 100, 300),
        ]
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: outcomes)
        from mullet.cli.routing_cmd import run
        rc = run(["insights"])
        out = capsys.readouterr().out
        assert "kimi" in out.lower()
        assert rc == 0

    def test_insights_section5_shows_cost_tail(self, tmp_path, monkeypatch, capsys):
        """Section 5 renders p50/p95 columns and legend from iter_recent outcomes."""
        from mullet.core.routing_history import RoutingOutcome
        # Craft outcomes directly to control cost_usd
        base_fields = dict(
            timestamp="2026-01-01T00:00:00+00:00",
            task_signature="aaaa",
            task_preview="task",
            provider="kimi",
            model="kimi-k2",
            tier="STANDARD",
            success=True,
            error_class=None,
            tokens_in=100,
            tokens_out=50,
            wall_time_ms=1000,
            tags=[],
        )
        outcomes = [RoutingOutcome(**{**base_fields, "cost_usd": 0.001}) for _ in range(3)]
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: outcomes)
        from mullet.cli.routing_cmd import run
        rc = run(["insights"])
        out = capsys.readouterr().out
        assert "Cost" in out and "Tail" in out
        assert "p50" in out.lower()
        assert "spike" in out.lower()  # legend line always present when data exists
        assert rc == 0

    def test_insights_section5_spike_flag_on_fat_tail(self, tmp_path, monkeypatch, capsys):
        """Section 5 shows ⚠ spike when a provider's p95 cost is >= 5x p50."""
        from mullet.core.routing_history import RoutingOutcome
        base = dict(
            timestamp="2026-01-01T00:00:00+00:00",
            task_signature="bbbb",
            task_preview="task",
            provider="openrouter",
            model="deepseek",
            tier="STANDARD",
            success=True,
            error_class=None,
            tokens_in=100,
            tokens_out=50,
            wall_time_ms=100,
            tags=[],
        )
        # 9 cheap outcomes + 1 expensive outlier → p95 >> p50
        costs = [0.001] * 9 + [0.100]  # p95 ≈ 0.055 (>> 5x p50 ≈ 0.001)
        outcomes = [RoutingOutcome(**{**base, "cost_usd": c}) for c in costs]
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: outcomes)
        from mullet.cli.routing_cmd import run
        rc = run(["insights"])
        out = capsys.readouterr().out
        # The ⚠ spike marker (not just the legend) should appear for this provider
        # Detect it by checking that ⚠ spike appears before the legend line
        lines = out.splitlines()
        spike_on_data_line = any("⚠ spike" in line and "openrouter" in line for line in lines)
        assert spike_on_data_line, f"Expected ⚠ spike on openrouter data line, got:\n{out}"
        assert rc == 0


class TestCmdShowWithData:
    def test_show_prints_table_row(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "mullet.cli.routing_cmd.summary_by_provider",
            lambda limit: {
                "kimi": {
                    "calls": 5, "successes": 4, "success_rate": 0.8,
                    "avg_cost": 0.00012, "total_cost": 0.00060, "avg_wall_ms": 600,
                },
            },
        )
        from mullet.cli.routing_cmd import run
        rc = run(["show"])
        out = capsys.readouterr().out
        assert "kimi" in out
        assert "Total $" in out
        assert "0.0006" in out  # total_cost rendered
        assert rc == 0


class TestCmdRecentWithData:
    def test_recent_shows_outcomes(self, tmp_path, monkeypatch, capsys):
        from mullet.core.routing_history import make_outcome
        monkeypatch.setattr(
            "mullet.cli.routing_cmd.iter_recent",
            lambda limit: [make_outcome("test task", "kimi", "kimi-k2", "STANDARD", True, 100, 50, 200)],
        )
        from mullet.cli.routing_cmd import run
        rc = run(["recent"])
        out = capsys.readouterr().out
        assert "kimi" in out.lower()
        assert rc == 0

    def test_recent_provider_filter(self, tmp_path, monkeypatch, capsys):
        """--provider filters to only that provider's outcomes."""
        from mullet.core.routing_history import make_outcome
        outcomes = [
            make_outcome("task1", "kimi", "kimi-k2", "STANDARD", True, 100, 50, 200),
            make_outcome("task2", "openrouter", "deepseek", "STANDARD", True, 200, 100, 300),
            make_outcome("task3", "kimi", "kimi-k2", "STANDARD", False, 100, 50, 150),
        ]
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: outcomes)
        from mullet.cli.routing_cmd import run
        rc = run(["recent", "--provider", "kimi"])
        out = capsys.readouterr().out
        assert "kimi" in out.lower()
        assert "openrouter" not in out.lower()
        assert rc == 0

    def test_recent_failed_filter(self, tmp_path, monkeypatch, capsys):
        """--failed shows only failed outcomes."""
        from mullet.core.routing_history import make_outcome
        outcomes = [
            make_outcome("task1", "kimi", "kimi-k2", "STANDARD", True, 100, 50, 200),
            make_outcome("task2", "openrouter", "deepseek", "STANDARD", False, 200, 100, 300),
        ]
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: outcomes)
        from mullet.cli.routing_cmd import run
        rc = run(["recent", "--failed"])
        out = capsys.readouterr().out
        assert "openrouter" in out.lower()
        assert "FAIL" in out
        assert rc == 0


class TestResetCommand:
    def test_reset_no_history(self, tmp_path, monkeypatch, capsys):
        """reset prints friendly message when history doesn't exist."""
        absent = tmp_path / "routing_history.jsonl"
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: absent)
        rc = run(["reset", "--yes"])
        out = capsys.readouterr().out
        assert "No routing history" in out
        assert rc == 0

    def test_reset_clears_file(self, tmp_path, monkeypatch, capsys):
        """reset with --yes removes all records."""
        import json as _json
        hist = tmp_path / "routing_history.jsonl"
        hist.write_text(
            _json.dumps({"provider": "kimi", "success": True}) + "\n"
            + _json.dumps({"provider": "openrouter", "success": True}) + "\n"
        )
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: hist)
        rc = run(["reset", "--yes"])
        out = capsys.readouterr().out
        assert "Cleared 2" in out
        assert hist.read_text() == ""
        assert rc == 0

    def test_reset_aborts_without_yes(self, tmp_path, monkeypatch, capsys):
        """reset without --yes aborts on 'n' answer."""
        import json as _json
        hist = tmp_path / "routing_history.jsonl"
        hist.write_text(_json.dumps({"provider": "kimi", "success": True}) + "\n")
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: hist)
        monkeypatch.setattr("builtins.input", lambda _: "n")
        rc = run(["reset"])
        out = capsys.readouterr().out
        assert "Aborted" in out
        assert rc == 1
        # File should be untouched
        assert "kimi" in hist.read_text()


class TestPurgeProviderCommand:
    def test_purge_no_match(self, tmp_path, monkeypatch, capsys):
        """purge-provider with no matching entries reports none found."""
        import json as _json
        hist = tmp_path / "routing_history.jsonl"
        hist.write_text(_json.dumps({"provider": "kimi", "success": True}) + "\n")
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: hist)
        rc = run(["purge-provider", "openrouter", "--yes"])
        out = capsys.readouterr().out
        assert "No entries found" in out
        assert rc == 0

    def test_purge_removes_matching(self, tmp_path, monkeypatch, capsys):
        """purge-provider removes only the named provider's entries."""
        import json as _json
        hist = tmp_path / "routing_history.jsonl"
        lines = [
            _json.dumps({"provider": "kimi", "success": True}),
            _json.dumps({"provider": "openrouter", "success": True}),
            _json.dumps({"provider": "kimi", "success": False}),
        ]
        hist.write_text("\n".join(lines) + "\n")
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: hist)
        rc = run(["purge-provider", "kimi", "--yes"])
        out = capsys.readouterr().out
        assert "Removed 2" in out
        remaining = hist.read_text()
        assert "kimi" not in remaining
        assert "openrouter" in remaining
        assert rc == 0


class TestCompareCommand:
    def test_compare_no_data(self, capsys):
        """compare with no history for either provider prints an error."""
        with patch("mullet.cli.routing_cmd.summary_by_provider", return_value={}):
            rc = run(["compare", "kimi", "openrouter"])
        out = capsys.readouterr().out
        assert "No data" in out
        assert rc == 1

    def test_compare_shows_both_providers(self, capsys):
        """compare shows a side-by-side table for two providers with data."""
        mock_summary = {
            "kimi": {
                "calls": 10, "successes": 9, "success_rate": 0.9,
                "avg_cost": 0.001, "total_cost": 0.01, "avg_wall_ms": 1200,
                "p50_cost": 0.0009, "p95_cost": 0.003,
                "p50_wall_ms": 1100.0, "p95_wall_ms": 2500.0,
            },
            "openrouter": {
                "calls": 5, "successes": 5, "success_rate": 1.0,
                "avg_cost": 0.0002, "total_cost": 0.001, "avg_wall_ms": 400,
                "p50_cost": 0.0002, "p95_cost": 0.0005,
                "p50_wall_ms": 390.0, "p95_wall_ms": 800.0,
            },
        }
        with patch("mullet.cli.routing_cmd.summary_by_provider", return_value=mock_summary):
            rc = run(["compare", "kimi", "openrouter"])
        out = capsys.readouterr().out
        assert "kimi" in out
        assert "openrouter" in out
        assert "Success rate" in out
        assert "Avg cost" in out
        assert rc == 0


class TestExportCommand:
    """Tests for ``mullet routing export``."""

    def _make_outcomes(self, n: int = 2):
        from mullet.core.routing_history import RoutingOutcome
        return [
            RoutingOutcome(
                timestamp=f"2026-06-16T00:0{i}:00+00:00",
                task_signature="abc123",
                task_preview="test task",
                provider="openrouter",
                model="deepseek/deepseek-v4-flash",
                tier="STANDARD",
                success=True,
                error_class=None,
                tokens_in=100,
                tokens_out=50,
                wall_time_ms=300,
                cost_usd=0.0001,
                tags=[],
            )
            for i in range(n)
        ]

    def test_export_no_history(self, capsys):
        """export with no history prints a message and returns 0."""
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=iter([])):
            rc = run(["export"])
        err = capsys.readouterr().err
        assert "No routing history" in err
        assert rc == 0

    def test_export_json_stdout(self, capsys):
        """export --format json prints valid JSON list to stdout."""
        outcomes = self._make_outcomes(2)
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=iter(outcomes)):
            rc = run(["export", "--format", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["provider"] == "openrouter"
        assert rc == 0

    def test_export_csv_stdout(self, capsys):
        """export --format csv prints a CSV with header to stdout."""
        outcomes = self._make_outcomes(1)
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=iter(outcomes)):
            rc = run(["export", "--format", "csv"])
        out = capsys.readouterr().out
        lines = out.strip().splitlines()
        assert lines[0].startswith("timestamp,")
        assert "provider" in lines[0]
        assert "openrouter" in lines[1]
        assert rc == 0

    def test_export_json_to_file(self, tmp_path, capsys):
        """export -o FILE writes JSON to disk and prints summary to stdout."""
        out_file = tmp_path / "history.json"
        outcomes = self._make_outcomes(3)
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=iter(outcomes)):
            rc = run(["export", "--format", "json", "-o", str(out_file)])
        assert rc == 0
        data = json.loads(out_file.read_text())
        assert len(data) == 3
        captured = capsys.readouterr().out
        assert "Exported 3 records" in captured
        assert str(out_file) in captured


class TestCmdShowFlagged:
    def test_show_flagged_quality_and_rl(self, capsys):
        """_cmd_show prints flags when provider is below quality threshold or has high RL rate."""
        mock_summary = {
            "badprov": {
                "calls": 5,
                "successes": 2,
                "success_rate": 0.40,
                "avg_cost": 0.001,
                "total_cost": 0.005,
                "avg_wall_ms": 500,
                "rate_limit_rate": 0.50,
            },
        }
        with patch("mullet.cli.routing_cmd.summary_by_provider", return_value=mock_summary):
            rc = run(["show"])
        out = capsys.readouterr().out
        assert "⚠" in out
        assert rc == 0


class TestCmdRecentFilter:
    def test_recent_empty_provider_filter(self, monkeypatch, capsys):
        """Empty outcomes with --provider prints filter description."""
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: [])
        rc = run(["recent", "--provider", "kimi"])
        out = capsys.readouterr().out
        assert "for provider='kimi'" in out
        assert rc == 0

    def test_recent_empty_failed_filter(self, monkeypatch, capsys):
        """Empty outcomes with --failed prints filter description."""
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: [])
        rc = run(["recent", "--failed"])
        out = capsys.readouterr().out
        assert "(failed only)" in out
        assert rc == 0


class TestCmdSignatureEmptyTask:
    def test_signature_no_task_text(self, capsys):
        """signature without a task argument prints error."""
        rc = run(["signature", "--tier", "HIGH"])
        err = capsys.readouterr().err
        assert "task string is required" in err
        assert rc == 1


class TestCmdInsightsAntiRec:
    def test_insights_anti_recommendations(self, monkeypatch, capsys):
        """Anti-recommendations section lists providers with high failure rates."""
        from mullet.core.routing_history import RoutingOutcome
        outcomes = [
            RoutingOutcome(
                timestamp="2026-01-01T00:00:00+00:00",
                task_signature="sig1",
                task_preview="task1",
                provider="badprovider",
                model="m1",
                tier="STANDARD",
                success=False,
                error_class="rate_limit",
                tokens_in=100,
                tokens_out=50,
                wall_time_ms=200,
                cost_usd=0.001,
                tags=[],
            ),
            RoutingOutcome(
                timestamp="2026-01-01T00:01:00+00:00",
                task_signature="sig1",
                task_preview="task1",
                provider="badprovider",
                model="m1",
                tier="STANDARD",
                success=False,
                error_class="rate_limit",
                tokens_in=100,
                tokens_out=50,
                wall_time_ms=200,
                cost_usd=0.001,
                tags=[],
            ),
            RoutingOutcome(
                timestamp="2026-01-01T00:02:00+00:00",
                task_signature="sig1",
                task_preview="task1",
                provider="badprovider",
                model="m1",
                tier="STANDARD",
                success=False,
                error_class="timeout",
                tokens_in=100,
                tokens_out=50,
                wall_time_ms=200,
                cost_usd=0.001,
                tags=[],
            ),
            RoutingOutcome(
                timestamp="2026-01-01T00:03:00+00:00",
                task_signature="sig1",
                task_preview="task1",
                provider="badprovider",
                model="m1",
                tier="STANDARD",
                success=False,
                error_class="timeout",
                tokens_in=100,
                tokens_out=50,
                wall_time_ms=200,
                cost_usd=0.001,
                tags=[],
            ),
        ]
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: outcomes)
        rc = run(["insights"])
        out = capsys.readouterr().out
        assert "badprovider" in out
        assert rc == 0

    def test_insights_rl_watchlist(self, monkeypatch, capsys):
        """Rate-limit watchlist shows providers with high RL rate despite good success."""
        from mullet.core.routing_history import RoutingOutcome
        base = dict(
            timestamp="2026-01-01T00:00:00+00:00",
            task_signature="sig2",
            task_preview="task2",
            provider="rl_prov",
            model="m2",
            tier="STANDARD",
            tokens_in=100,
            tokens_out=50,
            wall_time_ms=200,
            cost_usd=0.001,
            tags=[],
        )
        outcomes = [
            RoutingOutcome(**{**base, "success": True, "error_class": None}),
            RoutingOutcome(**{**base, "success": True, "error_class": None}),
            RoutingOutcome(**{**base, "success": True, "error_class": None}),
            RoutingOutcome(**{**base, "success": True, "error_class": None}),
            RoutingOutcome(**{**base, "success": True, "error_class": "rate_limit"}),
        ]
        monkeypatch.setattr("mullet.cli.routing_cmd.iter_recent", lambda limit: outcomes)
        rc = run(["insights"])
        out = capsys.readouterr().out
        assert "watchlist" in out.lower()
        assert rc == 0


class TestCmdLogShortLine:
    def test_log_short_line_displayed_verbatim(self, tmp_path, monkeypatch, capsys):
        """Log lines with fewer than 4 double-space-separated parts are printed verbatim."""
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        (mullet_dir / "routing.log").write_text("2026-01-01  brief note\n")
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = run(["log"])
        out = capsys.readouterr().out
        assert "2026-01-01  brief note" in out
        assert rc == 0


class TestCmdCompareOneMissing:
    def test_compare_one_provider_missing(self, capsys):
        """compare handles missing data for one provider with 'no data'."""
        mock_summary = {
            "kimi": {
                "calls": 10,
                "successes": 9,
                "success_rate": 0.9,
                "avg_cost": 0.001,
                "total_cost": 0.01,
                "avg_wall_ms": 1200,
                "p50_cost": 0.0009,
                "p95_cost": 0.003,
                "p50_wall_ms": 1100.0,
                "p95_wall_ms": 2500.0,
            },
        }
        with patch("mullet.cli.routing_cmd.summary_by_provider", return_value=mock_summary):
            rc = run(["compare", "kimi", "openrouter"])
        out = capsys.readouterr().out
        assert "kimi" in out
        assert rc == 0


class TestResetEOF:
    def test_reset_eof_aborts(self, tmp_path, monkeypatch, capsys):
        """reset aborts when EOFError is raised during confirmation."""
        import json as _json
        hist = tmp_path / "routing_history.jsonl"
        hist.write_text(_json.dumps({"provider": "kimi", "success": True}) + "\n")
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: hist)

        def raise_eof(_):
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)
        rc = run(["reset"])
        out = capsys.readouterr().out
        assert "Aborted" in out
        assert rc == 1
        assert "kimi" in hist.read_text()


class TestPurgeProviderExtra:
    def test_purge_no_history(self, tmp_path, monkeypatch, capsys):
        """purge-provider when history file doesn't exist prints friendly message."""
        absent = tmp_path / "routing_history.jsonl"
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: absent)
        rc = run(["purge-provider", "kimi"])
        out = capsys.readouterr().out
        assert "No routing history found" in out
        assert rc == 0

    def test_purge_blank_lines_skipped(self, tmp_path, monkeypatch, capsys):
        """Blank lines in history file are silently skipped during purge."""
        hist = tmp_path / "routing_history.jsonl"
        hist.write_text("\n\nsome blank\n")
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: hist)
        rc = run(["purge-provider", "other", "--yes"])
        out = capsys.readouterr().out
        assert "No entries found" in out
        assert rc == 0

    def test_purge_invalid_json_kept(self, tmp_path, monkeypatch, capsys):
        """Invalid JSON lines are preserved in the history file during purge."""
        hist = tmp_path / "routing_history.jsonl"
        hist.write_text("not valid json\n")
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: hist)
        rc = run(["purge-provider", "kimi", "--yes"])
        out = capsys.readouterr().out
        assert "No entries found" in out
        assert rc == 0
        assert "not valid json" in hist.read_text()

    def test_purge_confirm_aborted(self, tmp_path, monkeypatch, capsys):
        """purge-provider aborts when user answers 'n' at confirmation prompt."""
        import json as _json
        hist = tmp_path / "routing_history.jsonl"
        hist.write_text(_json.dumps({"provider": "kimi", "success": True}) + "\n")
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: hist)
        monkeypatch.setattr("builtins.input", lambda _: "n")
        rc = run(["purge-provider", "kimi"])
        out = capsys.readouterr().out
        assert "Aborted" in out
        assert rc == 1
        assert "kimi" in hist.read_text()

    def test_purge_confirm_eof(self, tmp_path, monkeypatch, capsys):
        """purge-provider aborts when EOFError is raised during confirmation."""
        import json as _json
        hist = tmp_path / "routing_history.jsonl"
        hist.write_text(_json.dumps({"provider": "kimi", "success": True}) + "\n")
        monkeypatch.setattr("mullet.core.routing_history.history_path", lambda: hist)

        def raise_eof(_):
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)
        rc = run(["purge-provider", "kimi"])
        out = capsys.readouterr().out
        assert "Aborted" in out
        assert rc == 1
        assert "kimi" in hist.read_text()


class TestCmdSimulate:
    """Tests for ``mullet routing simulate`` (TDD red phase)."""

    def test_simulate_empty_history(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("sonnet", "claude-sonnet-4-6", Tier.ADVANCED)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[]), \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate"])
        out = capsys.readouterr().out
        assert "No routing history" in out
        assert rc == 0

    def test_simulate_limit_zero(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("sonnet", "claude-sonnet-4-6", Tier.ADVANCED)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[]) as mock_iter, \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate", "--limit", "0"])
        out = capsys.readouterr().out
        assert "No routing history" in out
        assert rc == 0
        mock_iter.assert_called_once_with(limit=0)

    def test_simulate_provider_no_match(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("sonnet", "claude-sonnet-4-6", Tier.ADVANCED)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[]) as mock_iter, \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate", "--provider", "zzz", "--limit", "2"])
        out = capsys.readouterr().out
        assert "No routing history" in out.lower() or "no matching" in out.lower()
        assert rc == 0
        mock_iter.assert_called_once_with(limit=10)

    def test_simulate_single_changed(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("sonnet", "claude-sonnet-4-6", Tier.ADVANCED)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[_outcome("gemini", "gemini-2.5-flash", "STANDARD")]), \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate"])
        out = capsys.readouterr().out
        assert "gemini" in out
        assert "sonnet" in out
        assert "1 changed" in out.lower()
        assert rc == 0

    def test_simulate_single_unchanged(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("kimi", "kimi-k2.7-code", Tier.STANDARD)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[_outcome("kimi", "kimi-k2.7-code", "STANDARD")]), \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate"])
        out = capsys.readouterr().out
        assert "0 changed" in out.lower() or "1 unchanged" in out.lower()
        assert rc == 0

    def test_simulate_changed_only(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.side_effect = [
            _decision("sonnet", "claude-sonnet-4-6", Tier.ADVANCED),
            _decision("kimi", "kimi-k2.7-code", Tier.STANDARD),
        ]
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[
            _outcome("gemini", "gemini-2.5-flash", "STANDARD"),
            _outcome("kimi", "kimi-k2.7-code", "STANDARD"),
        ]), patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate", "--changed-only"])
        out = capsys.readouterr().out
        assert "gemini-2.5-flash" in out
        assert "claude-sonnet-4-6" in out
        assert "kimi-k2.7-code" not in out
        assert rc == 0

    def test_simulate_changed_only_all_unchanged(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("kimi", "kimi-k2.7-code", Tier.STANDARD)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[
            _outcome("kimi", "kimi-k2.7-code", "STANDARD"),
        ]), patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate", "--changed-only"])
        out = capsys.readouterr().out
        assert "kimi-k2.7-code" not in out
        assert "unchanged" in out.lower() or "summary" in out.lower()
        assert rc == 0

    def test_simulate_preset_allowlist(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        allowlist = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("sonnet", "claude-sonnet-4-6", Tier.ADVANCED)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = allowlist
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[_outcome("gemini", "gemini-2.5-flash", "STANDARD")]), \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate"])
        out = capsys.readouterr().out
        assert fake_router.route.call_args.kwargs["allowed_providers"] == allowlist
        assert rc == 0

    def test_simulate_no_preset(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("sonnet", "claude-sonnet-4-6", Tier.ADVANCED)
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[_outcome("gemini", "gemini-2.5-flash", "STANDARD")]), \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=MagicMock()):
            rc = run(["simulate", "--no-preset"])
        out = capsys.readouterr().out
        assert fake_router.route.call_args.kwargs["allowed_providers"] is None
        assert rc == 0

    def test_simulate_format_json(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("sonnet", "claude-sonnet-4-6", Tier.ADVANCED)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[_outcome("gemini", "gemini-2.5-flash", "STANDARD")]), \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate", "--format", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data
        assert "rows" in data
        assert isinstance(data["rows"], list)
        assert rc == 0

    def test_simulate_limit(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("sonnet", "claude-sonnet-4-6", Tier.ADVANCED)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[]) as mock_iter, \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate", "--limit", "10"])
        out = capsys.readouterr().out
        mock_iter.assert_called_once_with(limit=10)
        assert rc == 0

    def test_simulate_tier_enum_normalization(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = _decision("gemini", "gemini-2.5-flash", Tier.ADVANCED)
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[_outcome("gemini", "gemini-2.5-flash", "ADVANCED")]), \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate"])
        out = capsys.readouterr().out
        assert "0 changed" in out.lower() or "1 unchanged" in out.lower()
        assert rc == 0

    def test_simulate_none_provider(self, capsys):
        """fails because _cmd_simulate is not implemented yet"""
        fake_router = MagicMock()
        fake_router.route.return_value = RoutingDecision(
            provider=None, model="", tier=Tier.STANDARD, reasons=["allowlist cannot serve"]
        )
        fake_preset = MagicMock()
        fake_preset.router_allowlist.return_value = frozenset({"kimi", "gemini", "cheap_cloud"})
        fake_mgr = MagicMock()
        fake_mgr.active.return_value = fake_preset
        with patch("mullet.cli.routing_cmd.iter_recent", return_value=[_outcome("gemini", "gemini-2.5-flash", "STANDARD")]), \
             patch("mullet.cli.routing_cmd.build_default_router", return_value=(fake_router, None)), \
             patch("mullet.cli.routing_cmd.PresetManager", return_value=fake_mgr):
            rc = run(["simulate"])
        out = capsys.readouterr().out
        assert "none" in out.lower()
        assert rc == 0

    def test_simulate_invalid_format(self):
        """fails because _cmd_simulate is not implemented yet"""
        with pytest.raises(SystemExit):
            run(["simulate", "--format", "yaml"])

    def test_simulate_dispatch(self):
        """fails because _cmd_simulate is not implemented yet"""
        with patch("mullet.cli.routing_cmd._cmd_simulate", return_value=0) as mock_cmd:
            rc = run(["simulate", "--foo"])
        mock_cmd.assert_called_once_with(["--foo"])
        assert rc == 0
