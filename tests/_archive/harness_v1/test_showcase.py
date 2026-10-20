"""Tests for `mullet showcase` demo sequence."""

from __future__ import annotations



from mullet.cli.showcase_cli import cmd_showcase


class TestShowcaseCLI:
    """Test the `mullet showcase` visual demo sequence."""

    def test_showcase_runs_with_skip_real_llm(self, capsys):
        """With --skip-real-llm the showcase should complete and return 0."""
        rc = cmd_showcase(["--skip-real-llm", "--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "Mullet" in out

    def test_showcase_outputs_banner(self, capsys):
        """The banner should appear in output."""
        rc = cmd_showcase(["--skip-real-llm", "--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "Cloud in the front" in out or "Mullet" in out

    def test_showcase_outputs_skills_and_hooks(self, capsys):
        """Skills count and hooks count should be printed."""
        rc = cmd_showcase(["--skip-real-llm", "--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "skills" in out.lower()
        assert "hooks" in out.lower()

    def test_showcase_shows_permission_block(self, capsys):
        """The permission block demo should show a Block result."""
        rc = cmd_showcase(["--skip-real-llm", "--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "BLOCKED" in out or "blocked" in out.lower()

    def test_showcase_shows_metrics_snapshot(self, capsys):
        """The metrics snapshot section should appear."""
        rc = cmd_showcase(["--skip-real-llm", "--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "metrics" in out.lower()

    def test_showcase_shootout_mocked_when_skipped(self, capsys):
        """When --skip-real-llm is passed, shootout results should be mocked."""
        rc = cmd_showcase(["--skip-real-llm", "--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "kimi" in out.lower()
        assert "gemini" in out.lower()
        assert "mocked" in out.lower() or "shootout" in out.lower()

    def test_showcase_slow_flag_does_not_crash(self, capsys):
        """--slow should not cause an error (we won't actually pause in tests)."""
        rc = cmd_showcase(["--skip-real-llm", "--no-color", "--slow"])
        assert rc == 0

    def test_showcase_closing_banner(self, capsys):
        """The closing GitHub banner should appear."""
        rc = cmd_showcase(["--skip-real-llm", "--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "github.com/tomgould/mullet-public" in out
