"""Tests for config_counts and mcp_client pure helpers."""

import json


from mullet.hud.config_counts import (
    ConfigCounts,
    _read_json,
    _get_mcp_servers,
    _get_disabled_mcp_servers,
    _count_hooks,
    _count_rules_recursive,
    _get_output_style,
    _sentinel_key,
    _paths_equal,
    get_config_counts,
)
from mullet.core.tools.mcp_client import (
    MCPServerConfig,
    MCPTool,
    MCPCallResult,
    _sanitize_error,
    to_mullet_schema,
    load_mcp_servers,
)


# ---------------------------------------------------------------------------
# ConfigCounts dataclass
# ---------------------------------------------------------------------------

class TestConfigCounts:
    def test_defaults_all_zero(self):
        c = ConfigCounts()
        assert c.mcp_count == 0
        assert c.hooks_count == 0
        assert c.rules_count == 0
        assert c.claude_md_count == 0
        assert c.output_style is None

    def test_explicit_values(self):
        c = ConfigCounts(mcp_count=3, hooks_count=2, rules_count=5, claude_md_count=1, output_style="auto")
        assert c.mcp_count == 3
        assert c.hooks_count == 2
        assert c.rules_count == 5
        assert c.claude_md_count == 1
        assert c.output_style == "auto"


# ---------------------------------------------------------------------------
# _read_json
# ---------------------------------------------------------------------------

class TestReadJson:
    def test_returns_dict_for_valid_json(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text('{"key": "value"}')
        result = _read_json(f)
        assert result == {"key": "value"}

    def test_nonexistent_file_returns_empty(self, tmp_path):
        result = _read_json(tmp_path / "missing.json")
        assert result == {}

    def test_malformed_json_returns_empty(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json}")
        result = _read_json(f)
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("")
        result = _read_json(f)
        assert result == {}

    def test_returns_dict(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"mcpServers": {}}')
        assert isinstance(_read_json(f), dict)


# ---------------------------------------------------------------------------
# _get_mcp_servers
# ---------------------------------------------------------------------------

class TestGetMcpServers:
    def test_returns_set_of_server_names(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"mcpServers": {"server1": {}, "server2": {}}}))
        result = _get_mcp_servers(f)
        assert result == {"server1", "server2"}

    def test_empty_mcp_servers_returns_empty_set(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"mcpServers": {}}))
        assert _get_mcp_servers(f) == set()

    def test_missing_mcp_servers_key_returns_empty(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"other": "value"}))
        assert _get_mcp_servers(f) == set()

    def test_nonexistent_file_returns_empty(self, tmp_path):
        assert _get_mcp_servers(tmp_path / "missing.json") == set()

    def test_returns_set(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text(json.dumps({"mcpServers": {"s1": {}}}))
        assert isinstance(_get_mcp_servers(f), set)


# ---------------------------------------------------------------------------
# _get_disabled_mcp_servers
# ---------------------------------------------------------------------------

class TestGetDisabledMcpServers:
    def test_returns_disabled_servers(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"disabledMcpServers": ["server1", "server2"]}))
        result = _get_disabled_mcp_servers(f, "disabledMcpServers")
        assert result == {"server1", "server2"}

    def test_missing_key_returns_empty(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({}))
        assert _get_disabled_mcp_servers(f, "disabledMcpServers") == set()

    def test_nonexistent_file_returns_empty(self, tmp_path):
        assert _get_disabled_mcp_servers(tmp_path / "missing.json", "disabledMcpServers") == set()


# ---------------------------------------------------------------------------
# _count_hooks
# ---------------------------------------------------------------------------

class TestCountHooks:
    def test_counts_hooks_keys(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"hooks": {"PreToolUse": {}, "PostToolUse": {}}}))
        assert _count_hooks(f) == 2

    def test_empty_hooks_returns_zero(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"hooks": {}}))
        assert _count_hooks(f) == 0

    def test_no_hooks_key_returns_zero(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"other": "value"}))
        assert _count_hooks(f) == 0

    def test_nonexistent_file_returns_zero(self, tmp_path):
        assert _count_hooks(tmp_path / "missing.json") == 0


# ---------------------------------------------------------------------------
# _count_rules_recursive
# ---------------------------------------------------------------------------

class TestCountRulesRecursive:
    def test_counts_md_files(self, tmp_path):
        rules = tmp_path / "rules"
        rules.mkdir()
        (rules / "rule1.md").write_text("rule 1")
        (rules / "rule2.md").write_text("rule 2")
        assert _count_rules_recursive(rules) == 2

    def test_nonexistent_dir_returns_zero(self, tmp_path):
        assert _count_rules_recursive(tmp_path / "missing_rules") == 0

    def test_empty_dir_returns_zero(self, tmp_path):
        rules = tmp_path / "rules"
        rules.mkdir()
        assert _count_rules_recursive(rules) == 0

    def test_nested_md_files_counted(self, tmp_path):
        rules = tmp_path / "rules"
        rules.mkdir()
        sub = rules / "sub"
        sub.mkdir()
        (rules / "r1.md").write_text("r1")
        (sub / "r2.md").write_text("r2")
        assert _count_rules_recursive(rules) == 2

    def test_non_md_files_not_counted(self, tmp_path):
        rules = tmp_path / "rules"
        rules.mkdir()
        (rules / "file.txt").write_text("not a rule")
        assert _count_rules_recursive(rules) == 0


# ---------------------------------------------------------------------------
# _get_output_style
# ---------------------------------------------------------------------------

class TestGetOutputStyle:
    def test_returns_style_string(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"outputStyle": "stream"}))
        assert _get_output_style(f) == "stream"

    def test_empty_style_returns_none(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"outputStyle": ""}))
        assert _get_output_style(f) is None

    def test_missing_key_returns_none(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({}))
        assert _get_output_style(f) is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        assert _get_output_style(tmp_path / "missing.json") is None


# ---------------------------------------------------------------------------
# _sentinel_key
# ---------------------------------------------------------------------------

class TestSentinelKey:
    def test_existing_file_returns_tuple(self, tmp_path):
        f = tmp_path / "file.json"
        f.write_text("content")
        result = _sentinel_key(f)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_nonexistent_file_returns_none(self, tmp_path):
        result = _sentinel_key(tmp_path / "missing.json")
        assert result is None

    def test_changed_file_different_sentinel(self, tmp_path):
        f = tmp_path / "file.json"
        f.write_text("original")
        s1 = _sentinel_key(f)
        f.write_text("modified content here that is longer")
        s2 = _sentinel_key(f)
        # Size changed, so sentinel should differ
        assert s1 != s2


# ---------------------------------------------------------------------------
# _paths_equal
# ---------------------------------------------------------------------------

class TestPathsEqual:
    def test_same_path_equal(self, tmp_path):
        assert _paths_equal(tmp_path, tmp_path) is True

    def test_different_paths_not_equal(self, tmp_path):
        other = tmp_path / "sub"
        other.mkdir()
        assert _paths_equal(tmp_path, other) is False


# ---------------------------------------------------------------------------
# get_config_counts
# ---------------------------------------------------------------------------

class TestGetConfigCounts:
    def test_returns_config_counts(self, tmp_path):
        result = get_config_counts(str(tmp_path), home_dir=str(tmp_path))
        assert isinstance(result, ConfigCounts)

    def test_empty_dir_all_zeros(self, tmp_path):
        result = get_config_counts(str(tmp_path), home_dir=str(tmp_path))
        assert result.mcp_count == 0
        assert result.hooks_count == 0
        assert result.rules_count == 0

    def test_mcp_servers_counted(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(
            json.dumps({"mcpServers": {"s1": {}, "s2": {}}})
        )
        result = get_config_counts(str(tmp_path / "project"), home_dir=str(tmp_path))
        assert result.mcp_count >= 2

    def test_hooks_counted(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(
            json.dumps({"hooks": {"PreToolUse": {}, "PostToolUse": {}}})
        )
        result = get_config_counts(str(tmp_path / "project"), home_dir=str(tmp_path))
        assert result.hooks_count >= 2

    def test_claude_md_counted(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("instructions")
        result = get_config_counts(str(tmp_path), home_dir=str(tmp_path))
        assert result.claude_md_count >= 1

    def test_output_style_read(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(
            json.dumps({"outputStyle": "compact"})
        )
        result = get_config_counts(str(tmp_path / "project"), home_dir=str(tmp_path))
        assert result.output_style == "compact"

    def test_result_cached(self, tmp_path):
        r1 = get_config_counts(str(tmp_path), home_dir=str(tmp_path))
        r2 = get_config_counts(str(tmp_path), home_dir=str(tmp_path))
        assert r1 is r2


# ---------------------------------------------------------------------------
# MCPServerConfig
# ---------------------------------------------------------------------------

class TestMCPServerConfig:
    def test_fields_stored(self):
        cfg = MCPServerConfig(name="filesystem", command="npx", args=["-y", "@mcp/filesystem"])
        assert cfg.name == "filesystem"
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "@mcp/filesystem"]

    def test_default_args_empty(self):
        cfg = MCPServerConfig(name="s", command="cmd")
        assert cfg.args == []

    def test_default_env_empty(self):
        cfg = MCPServerConfig(name="s", command="cmd")
        assert cfg.env == {}

    def test_timeout_has_default(self):
        cfg = MCPServerConfig(name="s", command="cmd")
        assert cfg.timeout > 0


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------

class TestMCPTool:
    def test_fields_stored(self):
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object"},
            server_name="filesystem",
        )
        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert tool.server_name == "filesystem"


# ---------------------------------------------------------------------------
# MCPCallResult
# ---------------------------------------------------------------------------

class TestMCPCallResult:
    def test_content_stored(self):
        r = MCPCallResult(content="result text")
        assert r.content == "result text"

    def test_is_error_defaults_false(self):
        r = MCPCallResult(content="ok")
        assert r.is_error is False

    def test_explicit_error(self):
        r = MCPCallResult(content="error msg", is_error=True)
        assert r.is_error is True


# ---------------------------------------------------------------------------
# _sanitize_error
# ---------------------------------------------------------------------------

class TestSanitizeError:
    def test_plain_text_unchanged(self):
        result = _sanitize_error("Connection timeout")
        assert result == "Connection timeout"

    def test_github_token_redacted(self):
        msg = "Failed with token ghp_abcdefghijklmnopqrstuvwxyz"
        result = _sanitize_error(msg)
        assert "ghp_" not in result
        assert "[REDACTED]" in result

    def test_openai_key_redacted(self):
        msg = "Error: sk-abcdefghijklmnopqrstuvwxyz is invalid"
        result = _sanitize_error(msg)
        assert "sk-abcdef" not in result

    def test_returns_string(self):
        assert isinstance(_sanitize_error("any message"), str)


# ---------------------------------------------------------------------------
# to_mullet_schema
# ---------------------------------------------------------------------------

class TestToMulletSchema:
    def test_returns_dict(self):
        tool = MCPTool(name="t", description="desc", input_schema={"type": "object"}, server_name="s")
        result = to_mullet_schema(tool)
        assert isinstance(result, dict)

    def test_has_description(self):
        tool = MCPTool(name="t", description="do something", input_schema={}, server_name="s")
        result = to_mullet_schema(tool)
        assert result["description"] == "do something"

    def test_has_parameters(self):
        tool = MCPTool(name="t", description="d", input_schema={"type": "object", "properties": {}}, server_name="s")
        result = to_mullet_schema(tool)
        assert "parameters" in result

    def test_empty_schema_gets_default(self):
        tool = MCPTool(name="t", description="d", input_schema={}, server_name="s")
        result = to_mullet_schema(tool)
        assert result["parameters"]["type"] == "object"

    def test_none_schema_gets_default(self):
        tool = MCPTool(name="t", description="d", input_schema=None, server_name="s")
        result = to_mullet_schema(tool)
        assert "parameters" in result


# ---------------------------------------------------------------------------
# load_mcp_servers
# ---------------------------------------------------------------------------

class TestLoadMcpServers:
    def test_returns_list(self):
        result = load_mcp_servers({})
        assert isinstance(result, list)

    def test_empty_config_empty_list(self):
        assert load_mcp_servers({}) == []

    def test_single_server_loaded(self):
        config = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@mcp/filesystem"],
                "env": {},
            }
        }
        result = load_mcp_servers(config)
        assert len(result) == 1

    def test_server_name_set(self):
        config = {"myserver": {"command": "python", "args": ["-m", "server"]}}
        result = load_mcp_servers(config)
        assert result[0].name == "myserver"

    def test_server_command_set(self):
        config = {"s": {"command": "node", "args": ["server.js"]}}
        result = load_mcp_servers(config)
        assert result[0].command == "node"

    def test_server_args_set(self):
        config = {"s": {"command": "npx", "args": ["-y", "mcp-server"]}}
        result = load_mcp_servers(config)
        assert result[0].args == ["-y", "mcp-server"]

    def test_timeout_loaded(self):
        config = {"s": {"command": "cmd", "timeout": 60}}
        result = load_mcp_servers(config)
        assert result[0].timeout == 60

    def test_multiple_servers(self):
        config = {
            "server1": {"command": "cmd1"},
            "server2": {"command": "cmd2"},
        }
        result = load_mcp_servers(config)
        assert len(result) == 2

    def test_all_are_mcp_server_config(self):
        config = {"s1": {"command": "c"}, "s2": {"command": "d"}}
        for cfg in load_mcp_servers(config):
            assert isinstance(cfg, MCPServerConfig)
