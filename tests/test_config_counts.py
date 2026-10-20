"""Tests for mullet.hud.config_counts — pure filesystem helpers."""

import json
import pytest
from pathlib import Path

from mullet.hud.config_counts import (
    _sentinel_key,
    _read_json,
    _get_mcp_servers,
    _get_disabled_mcp_servers,
    _count_hooks,
    _count_rules_recursive,
    _get_output_style,
    _paths_equal,
    get_config_counts,
    ConfigCounts,
)


@pytest.fixture(autouse=True)
def clear_cache():
    import mullet.hud.config_counts as mod
    mod._CONFIG_CACHE.clear()
    yield
    mod._CONFIG_CACHE.clear()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

class TestSentinelKey:
    def test_returns_tuple_for_existing_file(self, tmp_path):
        f = tmp_path / "f.json"
        f.write_text("{}")
        key = _sentinel_key(f)
        assert isinstance(key, tuple)
        assert len(key) == 2

    def test_returns_none_for_missing_file(self, tmp_path):
        key = _sentinel_key(tmp_path / "nonexistent.json")
        assert key is None

    def test_different_files_have_different_keys(self, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text("a")
        b.write_text("bb")
        assert _sentinel_key(a) != _sentinel_key(b)


class TestReadJson:
    def test_valid_json(self, tmp_path):
        f = tmp_path / "cfg.json"
        f.write_text('{"key": "val"}')
        assert _read_json(f) == {"key": "val"}

    def test_missing_file_returns_empty(self, tmp_path):
        assert _read_json(tmp_path / "nope.json") == {}

    def test_invalid_json_returns_empty(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        assert _read_json(f) == {}

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("")
        assert _read_json(f) == {}


class TestGetMcpServers:
    def test_extracts_server_names(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"mcpServers": {"server_a": {}, "server_b": {}}}))
        assert _get_mcp_servers(f) == {"server_a", "server_b"}

    def test_empty_mcp_servers(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"mcpServers": {}}))
        assert _get_mcp_servers(f) == set()

    def test_missing_mcp_servers_key(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"other": "stuff"}))
        assert _get_mcp_servers(f) == set()

    def test_missing_file_returns_empty(self, tmp_path):
        assert _get_mcp_servers(tmp_path / "nope.json") == set()


class TestGetDisabledMcpServers:
    def test_extracts_disabled_list(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text(json.dumps({"disabledMcpServers": ["srv1", "srv2"]}))
        assert _get_disabled_mcp_servers(f, "disabledMcpServers") == {"srv1", "srv2"}

    def test_missing_key_returns_empty(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text(json.dumps({}))
        assert _get_disabled_mcp_servers(f, "disabledMcpServers") == set()

    def test_missing_file_returns_empty(self, tmp_path):
        assert _get_disabled_mcp_servers(tmp_path / "n.json", "disabledMcpServers") == set()


class TestCountHooks:
    def test_counts_hooks(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"hooks": {"UserPromptSubmit": [], "Stop": []}}))
        assert _count_hooks(f) == 2

    def test_no_hooks_returns_zero(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"other": "x"}))
        assert _count_hooks(f) == 0

    def test_missing_file_returns_zero(self, tmp_path):
        assert _count_hooks(tmp_path / "missing.json") == 0


class TestCountRulesRecursive:
    def test_counts_md_files(self, tmp_path):
        rules = tmp_path / "rules"
        rules.mkdir()
        (rules / "a.md").write_text("a")
        (rules / "b.md").write_text("b")
        (rules / "sub").mkdir()
        (rules / "sub" / "c.md").write_text("c")
        assert _count_rules_recursive(rules) == 3

    def test_ignores_non_md_files(self, tmp_path):
        rules = tmp_path / "rules"
        rules.mkdir()
        (rules / "a.md").write_text("")
        (rules / "b.txt").write_text("")
        assert _count_rules_recursive(rules) == 1

    def test_missing_dir_returns_zero(self, tmp_path):
        assert _count_rules_recursive(tmp_path / "nonexistent") == 0

    def test_empty_dir_returns_zero(self, tmp_path):
        rules = tmp_path / "rules"
        rules.mkdir()
        assert _count_rules_recursive(rules) == 0


class TestGetOutputStyle:
    def test_extracts_style(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"outputStyle": "compact"}))
        assert _get_output_style(f) == "compact"

    def test_empty_string_returns_none(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"outputStyle": ""}))
        assert _get_output_style(f) is None

    def test_missing_key_returns_none(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({}))
        assert _get_output_style(f) is None

    def test_whitespace_only_returns_none(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"outputStyle": "   "}))
        assert _get_output_style(f) is None


class TestPathsEqual:
    def test_same_path(self, tmp_path):
        assert _paths_equal(tmp_path, tmp_path) is True

    def test_different_paths(self, tmp_path):
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        assert _paths_equal(a, b) is False

    def test_symlink_equivalent(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        assert _paths_equal(real, link) is True


# ---------------------------------------------------------------------------
# get_config_counts integration
# ---------------------------------------------------------------------------

class TestGetConfigCounts:
    def _write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_empty_dirs_returns_zeroes(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        result = get_config_counts(str(cwd), home_dir=str(home))
        assert isinstance(result, ConfigCounts)
        assert result.mcp_count == 0
        assert result.hooks_count == 0
        assert result.rules_count == 0
        assert result.claude_md_count == 0
        assert result.output_style is None

    def test_counts_mcp_from_settings(self, tmp_path):
        home = tmp_path / "home"
        cwd = tmp_path / "project"
        cwd.mkdir()
        self._write_json(
            home / ".claude" / "settings.json",
            {"mcpServers": {"srv_a": {}, "srv_b": {}}},
        )
        result = get_config_counts(str(cwd), home_dir=str(home))
        assert result.mcp_count == 2

    def test_counts_hooks(self, tmp_path):
        home = tmp_path / "home"
        cwd = tmp_path / "project"
        cwd.mkdir()
        self._write_json(
            home / ".claude" / "settings.json",
            {"hooks": {"UserPromptSubmit": [], "Stop": []}},
        )
        result = get_config_counts(str(cwd), home_dir=str(home))
        assert result.hooks_count == 2

    def test_counts_rules_files(self, tmp_path):
        home = tmp_path / "home"
        cwd = tmp_path / "project"
        cwd.mkdir()
        rules_dir = home / ".claude" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "rule1.md").write_text("r1")
        (rules_dir / "rule2.md").write_text("r2")
        result = get_config_counts(str(cwd), home_dir=str(home))
        assert result.rules_count == 2

    def test_counts_claude_md(self, tmp_path):
        home = tmp_path / "home"
        cwd = tmp_path / "project"
        cwd.mkdir()
        (home / ".claude").mkdir(parents=True)
        (home / ".claude" / "CLAUDE.md").write_text("global")
        (cwd / "CLAUDE.md").write_text("project")
        result = get_config_counts(str(cwd), home_dir=str(home))
        assert result.claude_md_count >= 2

    def test_disabled_mcps_subtracted(self, tmp_path):
        home = tmp_path / "home"
        cwd = tmp_path / "project"
        cwd.mkdir()
        self._write_json(
            home / ".claude.json",
            {"mcpServers": {"a": {}, "b": {}, "c": {}}, "disabledMcpServers": ["b"]},
        )
        result = get_config_counts(str(cwd), home_dir=str(home))
        assert result.mcp_count == 2  # a, c (b disabled)

    def test_cache_returns_same_object(self, tmp_path):
        home = tmp_path / "home"
        cwd = tmp_path / "project"
        cwd.mkdir()
        r1 = get_config_counts(str(cwd), home_dir=str(home))
        r2 = get_config_counts(str(cwd), home_dir=str(home))
        assert r1 is r2

    def test_output_style_from_settings(self, tmp_path):
        home = tmp_path / "home"
        cwd = tmp_path / "project"
        cwd.mkdir()
        self._write_json(
            home / ".claude" / "settings.json",
            {"outputStyle": "verbose"},
        )
        result = get_config_counts(str(cwd), home_dir=str(home))
        assert result.output_style == "verbose"

    def test_project_mcp_servers_counted(self, tmp_path):
        home = tmp_path / "home"
        cwd = tmp_path / "project"
        cwd.mkdir()
        self._write_json(
            cwd / ".mcp.json",
            {"mcpServers": {"local_srv": {}}},
        )
        result = get_config_counts(str(cwd), home_dir=str(home))
        assert result.mcp_count >= 1
