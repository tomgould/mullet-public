"""Tests for plugin manifest schema and loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mullet.harness.plugins.manifest import (
    ManifestError,
    PluginManifest,
    parse_manifest,
)

pytestmark = pytest.mark.harness


def _write_manifest(plugin_dir: Path, data: dict, *, root: bool = False) -> Path:
    """Write a manifest under either .claude-plugin/ or plugin root."""
    if root:
        plugin_dir.mkdir(parents=True, exist_ok=True)
        target = plugin_dir / "plugin.json"
    else:
        target_dir = plugin_dir / ".claude-plugin"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "plugin.json"
    target.write_text(json.dumps(data))
    return target


def test_parse_valid_minimal_manifest(tmp_path: Path) -> None:
    plugin = tmp_path / "demo"
    _write_manifest(plugin, {"name": "demo", "version": "1.0.0"})

    m = parse_manifest(plugin)
    assert isinstance(m, PluginManifest)
    assert m.name == "demo"
    assert m.version == "1.0.0"
    assert m.description is None
    assert m.commands == []
    assert m.agents == []
    assert m.skills == []
    assert m.hooks is None
    assert m.mcpServers is None
    assert m.lspServers is None


def test_parse_full_manifest(tmp_path: Path) -> None:
    plugin = tmp_path / "demo"
    data = {
        "name": "demo",
        "version": "2.3.4-beta.1",
        "description": "A full-featured plugin",
        "commands": ["commands"],
        "agents": ["agents/a", "agents/b"],
        "skills": ["skills"],
        "hooks": "hooks.json",
        "mcpServers": ".mcp.json",
        "lspServers": "lsp",
    }
    _write_manifest(plugin, data)

    # Pre-create some referenced paths so resolve() returns canonicalised values.
    (plugin / "commands").mkdir()
    (plugin / "agents" / "a").mkdir(parents=True)
    (plugin / "agents" / "b").mkdir(parents=True)
    (plugin / "skills").mkdir()
    (plugin / "hooks.json").write_text("{}")
    (plugin / ".mcp.json").write_text("{}")
    (plugin / "lsp").mkdir()

    m = parse_manifest(plugin)
    assert m.description == "A full-featured plugin"
    assert m.version == "2.3.4-beta.1"
    assert all(Path(p).is_absolute() for p in m.commands)
    assert all(Path(p).is_absolute() for p in m.agents)
    assert all(Path(p).is_absolute() for p in m.skills)
    assert m.hooks is not None and Path(m.hooks).is_absolute()
    assert m.mcpServers is not None and Path(m.mcpServers).is_absolute()
    assert m.lspServers is not None and Path(m.lspServers).is_absolute()
    # And the resolved paths point at the right files
    assert Path(m.hooks).name == "hooks.json"
    assert Path(m.mcpServers).name == ".mcp.json"


@pytest.mark.parametrize("bad_name", ["Bad Name", "BAD_NAME", "", "-leading", "has_underscore"])
def test_rejects_bad_name_format(tmp_path: Path, bad_name: str) -> None:
    plugin = tmp_path / "demo"
    _write_manifest(plugin, {"name": bad_name, "version": "1.0.0"})

    with pytest.raises(ManifestError):
        parse_manifest(plugin)


@pytest.mark.parametrize("bad_version", ["1", "1.0", "v1.0.0", "1.0.0.0", "1.x.0", ""])
def test_rejects_bad_semver(tmp_path: Path, bad_version: str) -> None:
    plugin = tmp_path / "demo"
    _write_manifest(plugin, {"name": "demo", "version": bad_version})

    with pytest.raises(ManifestError):
        parse_manifest(plugin)


def test_resolves_relative_paths(tmp_path: Path) -> None:
    plugin = tmp_path / "demo"
    _write_manifest(
        plugin,
        {
            "name": "demo",
            "version": "1.0.0",
            "commands": ["cmd1", "cmd2"],
            "hooks": "hooks.json",
        },
    )
    (plugin / "cmd1").mkdir()
    (plugin / "cmd2").mkdir()
    (plugin / "hooks.json").write_text("{}")

    m = parse_manifest(plugin)
    expected_plugin = plugin.resolve()
    for resolved, original in zip(m.commands, ["cmd1", "cmd2"], strict=True):
        assert Path(resolved) == (expected_plugin / original).resolve()
    assert m.hooks is not None
    assert Path(m.hooks) == (expected_plugin / "hooks.json").resolve()


def test_missing_file_raises_manifest_error(tmp_path: Path) -> None:
    plugin = tmp_path / "missing"
    plugin.mkdir()

    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(plugin)
    assert excinfo.value.path == plugin


def test_invalid_json_raises_manifest_error(tmp_path: Path) -> None:
    plugin = tmp_path / "demo"
    target_dir = plugin / ".claude-plugin"
    target_dir.mkdir(parents=True)
    (target_dir / "plugin.json").write_text("{not valid json")

    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(plugin)
    assert "invalid JSON" in str(excinfo.value)


def test_supports_dot_claude_plugin_subdir(tmp_path: Path) -> None:
    plugin = tmp_path / "demo"
    _write_manifest(plugin, {"name": "demo", "version": "1.0.0"})

    assert (plugin / ".claude-plugin" / "plugin.json").is_file()
    m = parse_manifest(plugin)
    assert m.name == "demo"


def test_supports_root_plugin_json(tmp_path: Path) -> None:
    plugin = tmp_path / "demo"
    _write_manifest(plugin, {"name": "demo", "version": "1.0.0"}, root=True)

    assert (plugin / "plugin.json").is_file()
    assert not (plugin / ".claude-plugin").exists()
    m = parse_manifest(plugin)
    assert m.name == "demo"


def test_dot_claude_plugin_takes_precedence(tmp_path: Path) -> None:
    plugin = tmp_path / "demo"
    _write_manifest(plugin, {"name": "demo", "version": "1.0.0"}, root=True)
    _write_manifest(plugin, {"name": "demo-claude", "version": "2.0.0"}, root=False)

    m = parse_manifest(plugin)
    assert m.name == "demo-claude"
    assert m.version == "2.0.0"
