"""Tests for mullet.cli.memory_sync_cmd and mullet.core.tools.website_policy helpers."""

import json

from mullet.cli.memory_sync_cmd import _dry_run_observations, _dry_run_decisions
from mullet.core.tools.website_policy import RobotsTxtChecker, _MAX_CACHE_SIZE


# ---------------------------------------------------------------------------
# _dry_run_observations
# ---------------------------------------------------------------------------

class TestDryRunObservations:
    def test_no_summaries_dir_returns_zero(self, tmp_path):
        assert _dry_run_observations(tmp_path) == 0

    def test_empty_summaries_dir_returns_zero(self, tmp_path):
        summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
        summaries_dir.mkdir(parents=True)
        assert _dry_run_observations(tmp_path) == 0

    def test_counts_json_files(self, tmp_path):
        summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
        summaries_dir.mkdir(parents=True)
        for i in range(3):
            (summaries_dir / f"summary_{i}.json").write_text("{}")
        result = _dry_run_observations(tmp_path)
        assert result == 3

    def test_ignores_non_json_files(self, tmp_path):
        summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
        summaries_dir.mkdir(parents=True)
        (summaries_dir / "summary.json").write_text("{}")
        (summaries_dir / "readme.txt").write_text("not json")
        result = _dry_run_observations(tmp_path)
        assert result == 1

    def test_returns_correct_count_for_many_files(self, tmp_path):
        summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
        summaries_dir.mkdir(parents=True)
        for i in range(7):
            (summaries_dir / f"s{i}.json").write_text("{}")
        result = _dry_run_observations(tmp_path)
        assert result == 7


# ---------------------------------------------------------------------------
# _dry_run_decisions
# ---------------------------------------------------------------------------

class TestDryRunDecisions:
    def test_no_decisions_file_returns_zero(self, tmp_path):
        assert _dry_run_decisions(tmp_path) == 0

    def test_empty_file_returns_zero(self, tmp_path):
        d = tmp_path / ".mullet"
        d.mkdir()
        (d / "decisions.jsonl").write_text("")
        assert _dry_run_decisions(tmp_path) == 0

    def test_counts_lines(self, tmp_path):
        d = tmp_path / ".mullet"
        d.mkdir()
        lines = [
            json.dumps({"kind": "decision", "excerpt": "Use Kimi"}),
            json.dumps({"kind": "rejection", "excerpt": "Rejected Django"}),
        ]
        (d / "decisions.jsonl").write_text("\n".join(lines))
        result = _dry_run_decisions(tmp_path)
        assert result == 2

    def test_ignores_blank_lines(self, tmp_path):
        d = tmp_path / ".mullet"
        d.mkdir()
        content = json.dumps({"kind": "decision", "excerpt": "x"}) + "\n\n\n"
        (d / "decisions.jsonl").write_text(content)
        result = _dry_run_decisions(tmp_path)
        assert result == 1

    def test_single_line_returns_one(self, tmp_path):
        d = tmp_path / ".mullet"
        d.mkdir()
        (d / "decisions.jsonl").write_text(json.dumps({"kind": "decision", "excerpt": "Use pytest"}))
        result = _dry_run_decisions(tmp_path)
        assert result == 1


# ---------------------------------------------------------------------------
# RobotsTxtChecker constants
# ---------------------------------------------------------------------------

class TestWebsitePolicyConstants:
    def test_max_cache_size_positive(self):
        assert _MAX_CACHE_SIZE > 0

    def test_max_cache_size_is_int(self):
        assert isinstance(_MAX_CACHE_SIZE, int)


# ---------------------------------------------------------------------------
# RobotsTxtChecker instantiation
# ---------------------------------------------------------------------------

class TestRobotsTxtCheckerInit:
    def test_can_be_instantiated(self):
        checker = RobotsTxtChecker()
        assert checker is not None

    def test_cache_starts_empty(self):
        checker = RobotsTxtChecker()
        assert len(checker._cache) == 0

    def test_lock_is_present(self):
        checker = RobotsTxtChecker()
        assert hasattr(checker, "_lock")

    def test_multiple_instances_independent(self):
        c1 = RobotsTxtChecker()
        c2 = RobotsTxtChecker()
        c1._cache["test"] = "value"
        assert "test" not in c2._cache

    def test_check_returns_dict(self):
        checker = RobotsTxtChecker()
        # Use a URL that will fail to fetch (localhost non-existent)
        result = checker.check_robots_txt("http://127.0.0.1:19999/some/path")
        assert isinstance(result, dict)

    def test_failed_fetch_returns_error_key(self):
        checker = RobotsTxtChecker()
        result = checker.check_robots_txt("http://127.0.0.1:19999/page")
        # Either succeeds or returns error key — never crashes
        assert "allowed" in result or "error" in result
