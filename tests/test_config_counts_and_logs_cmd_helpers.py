"""Tests for mullet.hud.config_counts and mullet.cli.logs_cmd pure helpers."""


from mullet.hud.config_counts import (
    ConfigCounts,
    _read_json,
    _get_mcp_servers,
    _get_disabled_mcp_servers,
    _count_hooks,
    _count_rules_recursive,
    _get_output_style,
    _paths_equal,
)
from mullet.cli.logs_cmd import _format_time, _format_latency


# ---------------------------------------------------------------------------
# ConfigCounts dataclass
# ---------------------------------------------------------------------------

class TestConfigCounts:
    def test_defaults_are_zero(self):
        cc = ConfigCounts()
        assert cc.mcp_count == 0
        assert cc.hooks_count == 0
        assert cc.rules_count == 0
        assert cc.claude_md_count == 0

    def test_default_output_style_none(self):
        cc = ConfigCounts()
        assert cc.output_style is None

    def test_custom_values_stored(self):
        cc = ConfigCounts(mcp_count=3, hooks_count=5, rules_count=2)
        assert cc.mcp_count == 3
        assert cc.hooks_count == 5
        assert cc.rules_count == 2


# ---------------------------------------------------------------------------
# _read_json
# ---------------------------------------------------------------------------

class TestReadJson:
    def test_nonexistent_file_returns_empty_dict(self, tmp_path):
        result = _read_json(tmp_path / "nonexistent.json")
        assert result == {}

    def test_valid_json_returned(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        assert _read_json(f) == {"key": "value"}

    def test_invalid_json_returns_empty_dict(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json {")
        assert _read_json(f) == {}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("")
        assert _read_json(f) == {}

    def test_nested_json_returned(self, tmp_path):
        f = tmp_path / "nested.json"
        f.write_text('{"a": {"b": 1}}')
        assert _read_json(f)["a"]["b"] == 1


# ---------------------------------------------------------------------------
# _get_mcp_servers
# ---------------------------------------------------------------------------

class TestGetMcpServers:
    def test_no_file_returns_empty_set(self, tmp_path):
        result = _get_mcp_servers(tmp_path / "missing.json")
        assert result == set()

    def test_extracts_server_names(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"mcpServers": {"mullet": {}, "github": {}}}')
        result = _get_mcp_servers(f)
        assert "mullet" in result
        assert "github" in result

    def test_no_mcp_servers_key_returns_empty(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"hooks": {}}')
        assert _get_mcp_servers(f) == set()

    def test_returns_set(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"mcpServers": {}}')
        assert isinstance(_get_mcp_servers(f), set)


# ---------------------------------------------------------------------------
# _get_disabled_mcp_servers
# ---------------------------------------------------------------------------

class TestGetDisabledMcpServers:
    def test_no_file_returns_empty_set(self, tmp_path):
        result = _get_disabled_mcp_servers(tmp_path / "missing.json", "disabledMcpServers")
        assert result == set()

    def test_extracts_disabled_servers(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"disabledMcpServers": ["server_a", "server_b"]}')
        result = _get_disabled_mcp_servers(f, "disabledMcpServers")
        assert "server_a" in result
        assert "server_b" in result

    def test_missing_key_returns_empty(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{}')
        result = _get_disabled_mcp_servers(f, "disabledMcpServers")
        assert result == set()


# ---------------------------------------------------------------------------
# _count_hooks
# ---------------------------------------------------------------------------

class TestCountHooks:
    def test_no_file_returns_zero(self, tmp_path):
        assert _count_hooks(tmp_path / "missing.json") == 0

    def test_counts_hook_types(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"hooks": {"PreToolUse": [], "PostToolUse": [], "UserPromptSubmit": []}}')
        assert _count_hooks(f) == 3

    def test_empty_hooks_returns_zero(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"hooks": {}}')
        assert _count_hooks(f) == 0

    def test_no_hooks_key_returns_zero(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"mcpServers": {}}')
        assert _count_hooks(f) == 0


# ---------------------------------------------------------------------------
# _count_rules_recursive
# ---------------------------------------------------------------------------

class TestCountRulesRecursive:
    def test_nonexistent_dir_returns_zero(self, tmp_path):
        result = _count_rules_recursive(tmp_path / "nonexistent")
        assert result == 0

    def test_counts_md_files(self, tmp_path):
        (tmp_path / "rule1.md").write_text("# Rule 1")
        (tmp_path / "rule2.md").write_text("# Rule 2")
        assert _count_rules_recursive(tmp_path) == 2

    def test_ignores_non_md_files(self, tmp_path):
        (tmp_path / "code.py").write_text("code")
        (tmp_path / "readme.txt").write_text("text")
        assert _count_rules_recursive(tmp_path) == 0

    def test_counts_recursively(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "rule1.md").write_text("r1")
        (subdir / "rule2.md").write_text("r2")
        assert _count_rules_recursive(tmp_path) == 2

    def test_empty_dir_returns_zero(self, tmp_path):
        assert _count_rules_recursive(tmp_path) == 0


# ---------------------------------------------------------------------------
# _get_output_style
# ---------------------------------------------------------------------------

class TestGetOutputStyle:
    def test_no_file_returns_none(self, tmp_path):
        assert _get_output_style(tmp_path / "missing.json") is None

    def test_reads_output_style(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"outputStyle": "auto"}')
        assert _get_output_style(f) == "auto"

    def test_missing_key_returns_none(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{}')
        assert _get_output_style(f) is None

    def test_empty_string_returns_none(self, tmp_path):
        f = tmp_path / "settings.json"
        f.write_text('{"outputStyle": ""}')
        assert _get_output_style(f) is None


# ---------------------------------------------------------------------------
# _paths_equal
# ---------------------------------------------------------------------------

class TestPathsEqual:
    def test_same_path_is_equal(self, tmp_path):
        assert _paths_equal(tmp_path, tmp_path) is True

    def test_different_paths_not_equal(self, tmp_path):
        other = tmp_path / "subdir"
        other.mkdir()
        assert _paths_equal(tmp_path, other) is False

    def test_same_path_different_representation(self, tmp_path):
        p1 = tmp_path
        p2 = tmp_path / "." / ".."  / tmp_path.name
        assert _paths_equal(p1, p2) is True


# ---------------------------------------------------------------------------
# _format_time
# ---------------------------------------------------------------------------

class TestFormatTime:
    def test_iso_timestamp_formatted(self):
        result = _format_time("2025-01-15T10:30:45Z")
        assert "10:30:45" in result

    def test_returns_string(self):
        assert isinstance(_format_time("2025-01-15T10:30:45Z"), str)

    def test_malformed_falls_back(self):
        result = _format_time("not-a-date")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _format_latency
# ---------------------------------------------------------------------------

class TestFormatLatency:
    def test_ms_below_1000_shows_ms(self):
        result = _format_latency(500)
        assert "ms" in result
        assert "500" in result

    def test_ms_above_1000_shows_seconds(self):
        result = _format_latency(2500)
        assert "s" in result
        assert "2.5" in result

    def test_exactly_1000_shows_seconds(self):
        result = _format_latency(1000)
        assert "s" in result

    def test_999_shows_ms(self):
        result = _format_latency(999)
        assert "ms" in result

    def test_zero_shows_ms(self):
        result = _format_latency(0)
        assert "ms" in result

    def test_returns_string(self):
        assert isinstance(_format_latency(100), str)
