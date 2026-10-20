"""`mullet setup-claude --dry-run` previews like `--check` instead of installing."""

from unittest.mock import patch

import pytest

from mullet.cli import main as cli_main


@pytest.mark.parametrize("flag", ["--check", "--dry-run"])
def test_setup_claude_preview_flags(flag, monkeypatch):
    monkeypatch.setattr("sys.argv", ["mullet", "setup-claude", flag])
    with patch("mullet.cli.setup_claude.run_setup_claude") as run:
        cli_main._main_inner()
    run.assert_called_once_with(dry_run=True, yes=False, force=False)


def test_setup_claude_without_flag_installs(monkeypatch):
    monkeypatch.setattr("sys.argv", ["mullet", "setup-claude"])
    with patch("mullet.cli.setup_claude.run_setup_claude") as run:
        cli_main._main_inner()
    run.assert_called_once_with(dry_run=False, yes=False, force=False)
