"""Tests for mullet.harness.plugins.loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mullet.harness.plugins.loader import (
    LoadedPlugin,
    PluginLoadResult,
    load_all_plugins,
)

pytestmark = pytest.mark.harness


def _write_manifest(plugin_dir: Path, data: dict) -> None:
    target = plugin_dir / ".claude-plugin"
    target.mkdir(parents=True, exist_ok=True)
    (target / "plugin.json").write_text(json.dumps(data))


def _write_bad_manifest(plugin_dir: Path) -> None:
    target = plugin_dir / ".claude-plugin"
    target.mkdir(parents=True, exist_ok=True)
    (target / "plugin.json").write_text("{not valid json")


def test_load_single_valid_plugin(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    plugin = project / "alpha"
    _write_manifest(plugin, {"name": "alpha", "version": "1.0.0"})

    result = load_all_plugins({"project": [project], "user": [], "global": []})
    assert isinstance(result, PluginLoadResult)
    assert len(result.loaded) == 1
    assert isinstance(result.loaded[0], LoadedPlugin)
    assert result.loaded[0].manifest.name == "alpha"
    assert result.loaded[0].scope == "project"
    assert result.errors == []


def test_load_skips_invalid_manifest_logs_error(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    scope.mkdir()

    good = scope / "good"
    bad = scope / "bad"
    _write_manifest(good, {"name": "good", "version": "1.0.0"})
    _write_bad_manifest(bad)

    result = load_all_plugins({"project": [scope], "user": [], "global": []})
    loaded_names = sorted(p.manifest.name for p in result.loaded)
    assert loaded_names == ["good"]
    assert len(result.errors) == 1
    err_path, err_msg = result.errors[0]
    assert err_path == bad
    assert "invalid JSON" in err_msg or "JSON" in err_msg


def test_load_logs_warning_on_missing_referenced_dir(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    scope.mkdir()
    plugin = scope / "alpha"
    _write_manifest(
        plugin,
        {
            "name": "alpha",
            "version": "1.0.0",
            "skills": ["skills"],
        },
    )
    # NOTE: no skills/ dir created — should produce a warning, not error.

    result = load_all_plugins({"project": [scope], "user": [], "global": []})
    assert len(result.loaded) == 1
    plugin_record = result.loaded[0]
    assert plugin_record.manifest.name == "alpha"
    assert any("skills" in w for w in plugin_record.errors), plugin_record.errors


def test_load_priority_project_over_user(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user = tmp_path / "user"
    project.mkdir()
    user.mkdir()

    project_plugin = project / "shared"
    user_plugin = user / "shared"
    _write_manifest(project_plugin, {"name": "shared", "version": "9.9.9"})
    _write_manifest(user_plugin, {"name": "shared", "version": "1.0.0"})

    result = load_all_plugins({"project": [project], "user": [user], "global": []})
    assert len(result.loaded) == 1
    chosen = result.loaded[0]
    assert chosen.scope == "project"
    assert chosen.manifest.version == "9.9.9"


def test_load_priority_user_over_global(tmp_path: Path) -> None:
    user = tmp_path / "user"
    glob = tmp_path / "global"
    user.mkdir()
    glob.mkdir()

    user_plugin = user / "shared"
    glob_plugin = glob / "shared"
    _write_manifest(user_plugin, {"name": "shared", "version": "2.0.0"})
    _write_manifest(glob_plugin, {"name": "shared", "version": "0.1.0"})

    result = load_all_plugins({"project": [], "user": [user], "global": [glob]})
    assert len(result.loaded) == 1
    assert result.loaded[0].scope == "user"
    assert result.loaded[0].manifest.version == "2.0.0"


def test_load_returns_empty_when_no_plugins(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    result = load_all_plugins({"project": [project], "user": [], "global": []})
    assert result.loaded == []
    assert result.errors == []


def test_one_bad_plugin_does_not_break_others(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    scope.mkdir()

    a = scope / "alpha"
    b = scope / "broken"
    c = scope / "charlie"
    _write_manifest(a, {"name": "alpha", "version": "1.0.0"})
    _write_bad_manifest(b)
    _write_manifest(c, {"name": "charlie", "version": "1.0.0"})

    result = load_all_plugins({"project": [scope], "user": [], "global": []})
    loaded_names = sorted(p.manifest.name for p in result.loaded)
    assert loaded_names == ["alpha", "charlie"]
    assert len(result.errors) == 1
    assert result.errors[0][0] == b


def test_load_result_errors_list_populated_correctly(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    scope.mkdir()
    bad = scope / "bad"
    _write_bad_manifest(bad)

    result = load_all_plugins({"project": [scope], "user": [], "global": []})
    assert result.loaded == []
    assert len(result.errors) == 1
    err_path, err_msg = result.errors[0]
    assert err_path == bad
    assert isinstance(err_msg, str) and err_msg


def test_loaded_plugin_records_scope_and_dir(tmp_path: Path) -> None:
    user = tmp_path / "user"
    user.mkdir()
    plugin = user / "demo"
    _write_manifest(plugin, {"name": "demo", "version": "1.0.0"})

    result = load_all_plugins({"project": [], "user": [user], "global": []})
    assert len(result.loaded) == 1
    record = result.loaded[0]
    assert record.scope == "user"
    assert record.plugin_dir == plugin
