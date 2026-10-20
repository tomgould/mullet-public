"""Tests for the install helper and install script."""

import stat
from pathlib import Path

from mullet.harness.install.script import print_install_instructions


def test_print_install_instructions_outputs_curl_line(capsys):
    print_install_instructions()
    captured = capsys.readouterr()
    assert "curl -fsSL" in captured.out
    assert "raw.githubusercontent.com" in captured.out


def test_install_sh_exists_and_executable():
    repo_root = Path(__file__).resolve().parents[2]
    install_sh = repo_root / "scripts" / "install.sh"
    assert install_sh.exists(), f"{install_sh} does not exist"
    mode = install_sh.stat().st_mode
    assert mode & stat.S_IXUSR, f"{install_sh} is not executable by owner"


def test_install_sh_starts_with_shebang():
    repo_root = Path(__file__).resolve().parents[2]
    install_sh = repo_root / "scripts" / "install.sh"
    first_line = install_sh.read_text().splitlines()[0]
    assert first_line.startswith("#!/usr/bin/env bash")
