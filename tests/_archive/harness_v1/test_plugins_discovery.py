"""Tests for mullet.harness.plugins.discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mullet.harness.plugins.discovery import (
    default_plugin_scope_dirs,
    discover_plugins,
)

pytestmark = pytest.mark.harness


def _write_dot_claude_plugin(plugin_dir: Path, name: str = "demo", version: str = "1.0.0") -> None:
    (plugin_dir / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (plugin_dir / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": name, "version": version})
    )


def _write_root_plugin_json(plugin_dir: Path, name: str = "demo", version: str = "1.0.0") -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.json").write_text(
        json.dumps({"name": name, "version": version})
    )


def test_discover_finds_dot_claude_plugin_subdir(tmp_path: Path) -> None:
    project = tmp_path / "project_scope"
    project.mkdir()
    plugin = project / "alpha"
    _write_dot_claude_plugin(plugin)

    result = discover_plugins({"project": [project]})
    assert "project" in result
    assert plugin in result["project"]


def test_discover_finds_root_plugin_json(tmp_path: Path) -> None:
    user = tmp_path / "user_scope"
    user.mkdir()
    plugin = user / "beta"
    _write_root_plugin_json(plugin)

    result = discover_plugins({"user": [user]})
    assert plugin in result["user"]


def test_discover_returns_empty_for_nonexistent_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    result = discover_plugins({"project": [missing]})
    assert result == {"project": []}


def test_discover_3_scopes_returns_3_keys(tmp_path: Path) -> None:
    project = tmp_path / "p"
    user = tmp_path / "u"
    glob = tmp_path / "g"
    project.mkdir()
    user.mkdir()
    glob.mkdir()

    result = discover_plugins({"project": [project], "user": [user], "global": [glob]})
    assert set(result.keys()) == {"project", "user", "global"}


def test_default_plugin_scope_dirs_resolves_paths() -> None:
    defaults = default_plugin_scope_dirs()
    assert set(defaults.keys()) == {"project", "user", "global"}
    for scope, dirs in defaults.items():
        assert isinstance(dirs, list) and dirs
        for d in dirs:
            assert isinstance(d, Path)
            assert d.is_absolute()


def test_discover_skips_non_plugin_subdirs(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    scope.mkdir()
    (scope / "not_a_plugin").mkdir()
    plugin = scope / "real_plugin"
    _write_dot_claude_plugin(plugin)

    result = discover_plugins({"project": [scope]})
    assert result["project"] == [plugin]


def test_discover_skips_files_at_scope_root(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    scope.mkdir()
    (scope / "loose.txt").write_text("ignore me")
    plugin = scope / "alpha"
    _write_dot_claude_plugin(plugin)

    result = discover_plugins({"project": [scope]})
    assert result["project"] == [plugin]


def test_discover_returns_multiple_plugins_in_one_scope(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    scope.mkdir()
    a = scope / "alpha"
    b = scope / "bravo"
    _write_dot_claude_plugin(a, name="alpha")
    _write_root_plugin_json(b, name="bravo")

    result = discover_plugins({"project": [scope]})
    assert sorted(result["project"]) == sorted([a, b])


def test_discover_handles_missing_global_builtin_dir() -> None:
    # default_plugin_scope_dirs points global at mullet/harness/plugins/builtin
    # which may not exist yet — must not raise, just yield empty list.
    defaults = default_plugin_scope_dirs()
    result = discover_plugins(defaults)
    # Whatever the project/user dirs hold, the call must succeed and
    # the returned mapping must include the global key.
    assert "global" in result
    assert isinstance(result["global"], list)
