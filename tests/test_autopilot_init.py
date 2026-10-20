"""Tests for mullet.cli.autopilot_init."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


from mullet.cli.autopilot_init import (
    STATE_FILES,
    TEMPLATES,
    _load_template,
    _target_dir,
    require_goal,
    run_autopilot_init,
)


class TestTargetDir:
    def test_returns_correct_path(self, tmp_path: Path) -> None:
        result = _target_dir(tmp_path)
        assert result == tmp_path / ".mullet" / "autopilot"

    def test_does_not_create_dir(self, tmp_path: Path) -> None:
        _target_dir(tmp_path)
        assert not (tmp_path / ".mullet").exists()


class TestLoadTemplate:
    def test_missing_template_returns_skeleton(self) -> None:
        content = _load_template("greenfield-api", "nonexistent_file.md")
        assert "nonexistent_file.md" in content
        assert "fill in manually" in content

    def test_returns_string(self) -> None:
        content = _load_template("greenfield-api", "goal.md")
        assert isinstance(content, str)
        assert len(content) > 0


class TestRunAutopilotInitHelp:
    def test_help_flag_returns_zero(self, tmp_path: Path, capsys) -> None:
        rc = run_autopilot_init(["--help"], cwd=tmp_path)
        assert rc == 0

    def test_help_short_flag(self, tmp_path: Path, capsys) -> None:
        rc = run_autopilot_init(["-h"], cwd=tmp_path)
        assert rc == 0

    def test_list_flag_returns_zero(self, tmp_path: Path, capsys) -> None:
        rc = run_autopilot_init(["--list"], cwd=tmp_path)
        assert rc == 0
        out = capsys.readouterr().out
        for t in TEMPLATES:
            assert t in out

    def test_list_short_flag(self, tmp_path: Path, capsys) -> None:
        rc = run_autopilot_init(["-l"], cwd=tmp_path)
        assert rc == 0

    def test_unknown_arg_returns_two(self, tmp_path: Path) -> None:
        rc = run_autopilot_init(["--unknown-bogus"], cwd=tmp_path)
        assert rc == 2

    def test_bad_template_returns_two(self, tmp_path: Path) -> None:
        rc = run_autopilot_init(["--template", "no-such-template"], cwd=tmp_path)
        assert rc == 2


class TestRunAutopilotInitScaffold:
    def test_default_creates_all_state_files(self, tmp_path: Path) -> None:
        rc = run_autopilot_init([], cwd=tmp_path)
        assert rc == 0
        target = _target_dir(tmp_path)
        for name in STATE_FILES:
            assert (target / name).exists(), f"Missing {name}"

    def test_explicit_template_flag(self, tmp_path: Path) -> None:
        rc = run_autopilot_init(["--template", "refactor"], cwd=tmp_path)
        assert rc == 0
        assert (_target_dir(tmp_path) / "goal.md").exists()

    def test_short_template_flag(self, tmp_path: Path) -> None:
        rc = run_autopilot_init(["-t", "feature-sprint"], cwd=tmp_path)
        assert rc == 0
        assert (_target_dir(tmp_path) / "goal.md").exists()

    def test_existing_files_are_skipped_without_force(self, tmp_path: Path) -> None:
        target = _target_dir(tmp_path)
        target.mkdir(parents=True)
        goal = target / "goal.md"
        goal.write_text("original content")

        run_autopilot_init([], cwd=tmp_path)
        assert goal.read_text() == "original content"

    def test_force_overwrites_existing_files(self, tmp_path: Path) -> None:
        target = _target_dir(tmp_path)
        target.mkdir(parents=True)
        goal = target / "goal.md"
        goal.write_text("original content")

        run_autopilot_init(["--force"], cwd=tmp_path)
        assert goal.read_text() != "original content"

    def test_returns_zero_on_success(self, tmp_path: Path) -> None:
        rc = run_autopilot_init([], cwd=tmp_path)
        assert rc == 0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        # target dir doesn't exist yet
        target = _target_dir(tmp_path)
        assert not target.exists()
        run_autopilot_init([], cwd=tmp_path)
        assert target.is_dir()


class TestRequireGoal:
    def test_missing_file_returns_false(self, tmp_path: Path) -> None:
        ok, msg = require_goal(tmp_path)
        assert ok is False
        assert "goal.md" in msg

    def test_filled_goal_returns_true(self, tmp_path: Path) -> None:
        target = _target_dir(tmp_path)
        target.mkdir(parents=True)
        goal = target / "goal.md"
        goal.write_text(
            "# Goal\n\nWe need to build X.\nProblem: current system is slow.\n"
            "Outcome: 3x faster API.\nAcceptance: p95 < 200ms under load.\n"
        )
        ok, msg = require_goal(tmp_path)
        assert ok is True
        assert "content lines" in msg

    def test_empty_goal_returns_false(self, tmp_path: Path) -> None:
        target = _target_dir(tmp_path)
        target.mkdir(parents=True)
        goal = target / "goal.md"
        goal.write_text("# Goal\n\n")
        ok, msg = require_goal(tmp_path)
        assert ok is False

    def test_header_only_goal_returns_false(self, tmp_path: Path) -> None:
        target = _target_dir(tmp_path)
        target.mkdir(parents=True)
        goal = target / "goal.md"
        goal.write_text("# Goal\n## Section\n### Sub\n")
        ok, msg = require_goal(tmp_path)
        assert ok is False

    def test_returns_tuple(self, tmp_path: Path) -> None:
        result = require_goal(tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestTemplateSignatures:
    def test_continue_when_template_goal_missing(self):
        """Line 174: continue when a template's goal.md doesn't exist."""
        from mullet.cli.autopilot_init import _template_signatures, _TEMPLATES_ROOT
        # Mock path.exists() to return False for one template, True for none
        with patch("mullet.cli.autopilot_init.Path.exists", return_value=False):
            sigs = _template_signatures()
        assert sigs == []
