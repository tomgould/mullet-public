"""Tests for HUD config counters (MCP/hooks/rules/CLAUDE.md)."""
import json
import pytest
from mullet.hud.config_counts import get_config_counts, _CONFIG_CACHE


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear config cache before and after each test."""
    _CONFIG_CACHE.clear()
    yield
    _CONFIG_CACHE.clear()


def test_all_missing_returns_zeros(tmp_path):
    """When all config files are missing, return zeroed counts."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()
    result = get_config_counts(str(cwd), home_dir=str(home))
    assert result.mcp_count == 0
    assert result.hooks_count == 0
    assert result.rules_count == 0
    assert result.claude_md_count == 0
    assert result.output_style is None


def test_single_user_settings_with_mcp_servers(tmp_path):
    """Count 3 MCP servers from ~/.claude/settings.json."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)

    settings = {
        "mcpServers": {
            "server_a": {},
            "server_b": {},
            "server_c": {},
        }
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings))

    cwd = tmp_path / "project"
    cwd.mkdir()

    result = get_config_counts(str(cwd), home_dir=str(home))
    assert result.mcp_count == 3


def test_mcp_dedup_across_user_and_project(tmp_path):
    """MCP servers counted separately per scope (no cross-scope dedup per TS logic)."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)

    # User has 2 MCPs
    user_settings = {"mcpServers": {"s1": {}, "s2": {}}}
    (claude_dir / "settings.json").write_text(json.dumps(user_settings))

    # Project has 2 MCPs (1 overlap, 1 new)
    cwd = tmp_path / "project"
    project_claude = cwd / ".claude"
    project_claude.mkdir(parents=True)
    project_settings = {"mcpServers": {"s2": {}, "s3": {}}}
    (project_claude / "settings.json").write_text(json.dumps(project_settings))

    result = get_config_counts(str(cwd), home_dir=str(home))
    # Per TS config-reader.ts line 407: "A server with the same name in both user
    # and project scope counts as 2 (separate configs)."
    # User: s1, s2 (2); Project: s2, s3 (2) → Total = 4
    assert result.mcp_count == 4


def test_disabled_mcp_servers_removes_entries(tmp_path):
    """disabledMcpServers array removes names from dedup."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)

    # User has 3 MCPs, but 1 is disabled
    settings = {
        "mcpServers": {"s1": {}, "s2": {}, "s3": {}},
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings))

    claude_json = {
        "disabledMcpServers": ["s2"],
    }
    (home / ".claude.json").write_text(json.dumps(claude_json))

    cwd = tmp_path / "project"
    cwd.mkdir()

    result = get_config_counts(str(cwd), home_dir=str(home))
    assert result.mcp_count == 2  # s1 + s3 (s2 disabled)


def test_hooks_count(tmp_path):
    """Count top-level keys under config['hooks']."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)

    settings = {
        "hooks": {
            "beforeExit": "some_command",
            "onIdle": "another_command",
            "userPromptSubmit": "hook_command",
        }
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings))

    cwd = tmp_path / "project"
    cwd.mkdir()

    result = get_config_counts(str(cwd), home_dir=str(home))
    assert result.hooks_count == 3


def test_rules_recursive_count(tmp_path):
    """Recursively count .md files under rules directory."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    rules_dir = claude_dir / "rules"
    rules_dir.mkdir(parents=True)

    # Create nested structure with .md files
    (rules_dir / "rule1.md").write_text("# Rule 1")
    (rules_dir / "rule2.md").write_text("# Rule 2")

    subdir = rules_dir / "subdir"
    subdir.mkdir()
    (subdir / "rule3.md").write_text("# Rule 3")
    (subdir / "rule4.md").write_text("# Rule 4")

    cwd = tmp_path / "project"
    cwd.mkdir()

    result = get_config_counts(str(cwd), home_dir=str(home))
    assert result.rules_count == 4


def test_claude_md_count_all_locations(tmp_path):
    """Count CLAUDE.md across ~/, cwd/, cwd/.claude/."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)

    # User-level CLAUDE.md
    (claude_dir / "CLAUDE.md").write_text("# User CLAUDE.md")

    cwd = tmp_path / "project"
    cwd.mkdir()

    # Project-level CLAUDE.md
    (cwd / "CLAUDE.md").write_text("# Project CLAUDE.md")

    # Project .claude/CLAUDE.md
    project_claude = cwd / ".claude"
    project_claude.mkdir()
    (project_claude / "CLAUDE.md").write_text("# Project .claude CLAUDE.md")

    result = get_config_counts(str(cwd), home_dir=str(home))
    assert result.claude_md_count == 3


def test_output_style_from_config(tmp_path):
    """Read outputStyle from settings.json."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)

    settings = {
        "outputStyle": "markdown",
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings))

    cwd = tmp_path / "project"
    cwd.mkdir()

    result = get_config_counts(str(cwd), home_dir=str(home))
    assert result.output_style == "markdown"


def test_cache_hit_on_unchanged_mtime(tmp_path):
    """Second call with unchanged mtime uses cache (verify via call counter)."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)

    settings = {"mcpServers": {"s1": {}}}
    (claude_dir / "settings.json").write_text(json.dumps(settings))

    cwd = tmp_path / "project"
    cwd.mkdir()

    # First call
    result1 = get_config_counts(str(cwd), home_dir=str(home))
    assert result1.mcp_count == 1

    # Cache should have one entry
    assert len(_CONFIG_CACHE) == 1

    # Second call with same state
    result2 = get_config_counts(str(cwd), home_dir=str(home))
    assert result2.mcp_count == 1

    # Cache size unchanged (same key reused)
    assert len(_CONFIG_CACHE) == 1


def test_malformed_json_swallowed_silently(tmp_path):
    """Malformed JSON is silently ignored, counts default to 0."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)

    # Write invalid JSON
    (claude_dir / "settings.json").write_text("{invalid json")

    cwd = tmp_path / "project"
    cwd.mkdir()

    # Should not raise, just return zeros
    result = get_config_counts(str(cwd), home_dir=str(home))
    assert result.mcp_count == 0
    assert result.hooks_count == 0


def test_whitespace_output_style_treated_as_none(tmp_path):
    """Empty/whitespace outputStyle is treated as None."""
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)

    settings = {
        "outputStyle": "   ",  # Just whitespace
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings))

    cwd = tmp_path / "project"
    cwd.mkdir()

    result = get_config_counts(str(cwd), home_dir=str(home))
    assert result.output_style is None


class TestConfigCountsMissingLines:
    """Cover edge-case lines in config_counts.py helpers."""

    def test_mcp_servers_non_dict_returns_empty(self, tmp_path):
        """_get_mcp_servers returns empty set when mcpServers is not a dict (line 49)."""
        from mullet.hud.config_counts import _get_mcp_servers
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"mcpServers": ["not-a-dict"]}))
        assert _get_mcp_servers(path) == set()

    def test_disabled_mcp_non_list_returns_empty(self, tmp_path):
        """_get_disabled_mcp_servers returns empty set when value is not a list (line 58)."""
        from mullet.hud.config_counts import _get_disabled_mcp_servers
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"disabledMcpServers": "not-a-list"}))
        assert _get_disabled_mcp_servers(path, "disabledMcpServers") == set()

    def test_hooks_non_dict_returns_zero(self, tmp_path):
        """_count_hooks returns 0 when hooks is not a dict (line 67)."""
        from mullet.hud.config_counts import _count_hooks
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"hooks": ["not-a-dict"]}))
        assert _count_hooks(path) == 0

    def test_rules_recursive_oserror_swallowed(self, tmp_path):
        """_count_rules_recursive swallows OSError on rglob (lines 79-80)."""
        from mullet.hud.config_counts import _count_rules_recursive
        import pathlib
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        (rules_dir / "file.md").write_text("# test")
        # Patch the concrete path class rglob to raise OSError
        orig_rglob = type(rules_dir).rglob
        def bad_rglob(self, pattern):
            raise OSError("simulated rglob failure")
        type(rules_dir).rglob = bad_rglob
        try:
            assert _count_rules_recursive(rules_dir) == 0
        finally:
            type(rules_dir).rglob = orig_rglob

    def test_output_style_non_str_returns_none(self, tmp_path):
        """_get_output_style returns None when outputStyle is not a string (line 91)."""
        from mullet.hud.config_counts import _get_output_style
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"outputStyle": 123}))
        assert _get_output_style(path) is None

    def test_paths_equal_oserror_returns_false(self, tmp_path):
        """_paths_equal returns False when resolve() raises OSError (lines 98-99)."""
        from mullet.hud.config_counts import _paths_equal
        import pathlib
        original_resolve = pathlib.Path.resolve
        def bad_resolve(self, strict=False):
            raise OSError("permission denied")
        pathlib.Path.resolve = bad_resolve
        try:
            assert _paths_equal(tmp_path / "a", tmp_path / "b") is False
        finally:
            pathlib.Path.resolve = original_resolve
