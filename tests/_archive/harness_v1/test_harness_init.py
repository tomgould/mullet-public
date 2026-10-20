"""Tests for `mullet harness-init` scaffolding command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mullet.cli.harness_init import cmd_harness_init


@pytest.mark.harness
def test_detects_python_via_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'foo'\n")
    assert cmd_harness_init(["--no-pool"]) == 0
    perms = yaml.safe_load((tmp_path / ".mullet" / "permissions.yaml").read_text())
    assert "rm -rf /" in perms["denied_commands"]


@pytest.mark.harness
def test_detects_js_via_package_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "package.json").write_text('{"name": "foo"}')
    assert cmd_harness_init(["--no-pool"]) == 0
    assert (tmp_path / ".mullet" / "MEMORY.md").exists()


@pytest.mark.harness
def test_detects_rust_via_cargo_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n")
    assert cmd_harness_init(["--no-pool"]) == 0
    assert (tmp_path / ".mullet").exists()


@pytest.mark.harness
def test_detects_go_via_go_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "go.mod").write_text("module foo\n")
    assert cmd_harness_init(["--no-pool"]) == 0
    assert (tmp_path / ".mullet").exists()


@pytest.mark.harness
def test_detects_other_when_no_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cmd_harness_init(["--no-pool"]) == 0
    assert (tmp_path / ".mullet").exists()


@pytest.mark.harness
def test_creates_dot_mullet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cmd_harness_init(["--no-pool"]) == 0
    dot_mullet = tmp_path / ".mullet"
    assert dot_mullet.is_dir()
    assert (dot_mullet / "skills").is_dir()
    assert (dot_mullet / "plugins").is_dir()
    assert (dot_mullet / "memory").is_dir()


@pytest.mark.harness
def test_writes_permissions_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cmd_harness_init(["--no-pool"]) == 0
    perms_path = tmp_path / ".mullet" / "permissions.yaml"
    assert perms_path.exists()
    perms = yaml.safe_load(perms_path.read_text())
    assert perms["denied_commands"] == ["rm -rf /", "DROP TABLE", "format c:"]
    assert "path_rules" not in perms


@pytest.mark.harness
def test_writes_memory_md_with_readme_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    readme = "This is a great project.\n" * 100
    (tmp_path / "README.md").write_text(readme)
    # Set up a minimal git repo so git log works
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=False)
    subprocess.run(["git", "config", "user.email", "a@b.com"], cwd=str(tmp_path), capture_output=True, check=False)
    subprocess.run(["git", "config", "user.name", "A"], cwd=str(tmp_path), capture_output=True, check=False)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=False)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(tmp_path), capture_output=True, check=False)

    assert cmd_harness_init(["--no-pool"]) == 0
    memory = (tmp_path / ".mullet" / "MEMORY.md").read_text()
    assert "This is a great project." in memory
    assert "## Recent commits" in memory
    assert "initial" in memory


@pytest.mark.harness
def test_force_overwrites_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    dot_mullet = tmp_path / ".mullet"
    dot_mullet.mkdir()
    (dot_mullet / "permissions.yaml").write_text("old: value\n")
    (dot_mullet / "MEMORY.md").write_text("old memory\n")
    (dot_mullet / ".gitignore").write_text("old\n")
    (tmp_path / ".mcp.json").write_text("old\n")

    assert cmd_harness_init(["--force", "--no-pool"]) == 0
    perms = yaml.safe_load((dot_mullet / "permissions.yaml").read_text())
    assert "rm -rf /" in perms["denied_commands"]
    assert "old memory" not in (dot_mullet / "MEMORY.md").read_text()
    assert "sessions/" in (dot_mullet / ".gitignore").read_text()
    assert "// .mcp.json" in (tmp_path / ".mcp.json").read_text()


@pytest.mark.harness
def test_yes_flag_skips_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Pretend we are on a TTY so it would normally prompt
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    called: list[str] = []

    def fake_input(_: str) -> str:
        pytest.fail("Should not prompt when --yes is passed")

    monkeypatch.setattr("builtins.input", fake_input)

    def fake_run_pool_create(name: str) -> None:
        called.append(name)

    monkeypatch.setattr("mullet.cli.harness_init._run_pool_create", fake_run_pool_create)

    assert cmd_harness_init(["--yes"]) == 0
    assert called == [tmp_path.name]


@pytest.mark.harness
def test_no_pool_flag_skips_pool_suggestion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Pretend we are on a TTY
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def fake_input(_: str) -> str:
        pytest.fail("Should not prompt when --no-pool is passed")

    monkeypatch.setattr("builtins.input", fake_input)

    called: list[str] = []

    def fake_run_pool_create(name: str) -> None:
        called.append(name)

    monkeypatch.setattr("mullet.cli.harness_init._run_pool_create", fake_run_pool_create)

    assert cmd_harness_init(["--no-pool"]) == 0
    assert called == []
