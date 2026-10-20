"""Tests for mcp_client pure helpers and web_crawler pure functions."""


from mullet.core.tools.mcp_client import (
    MCPServerConfig,
    MCPTool,
    MCPCallResult,
    _sanitize_error,
    to_mullet_schema,
    load_mcp_servers,
    DEFAULT_TIMEOUT,
    DEFAULT_CONNECT_TIMEOUT,
)
from mullet.knowledge.sources.web_crawler import (
    WebSourceConfig,
    normalize_link,
    _is_in_scope,
    _extract_links,
    _url_to_filepath,
)


# ---------------------------------------------------------------------------
# MCPServerConfig
# ---------------------------------------------------------------------------

class TestMCPServerConfig:
    def test_name_stored(self):
        cfg = MCPServerConfig(name="fs", command="npx")
        assert cfg.name == "fs"

    def test_command_stored(self):
        cfg = MCPServerConfig(name="fs", command="npx")
        assert cfg.command == "npx"

    def test_args_default_empty(self):
        cfg = MCPServerConfig(name="fs", command="npx")
        assert cfg.args == []

    def test_env_default_empty(self):
        cfg = MCPServerConfig(name="fs", command="npx")
        assert cfg.env == {}

    def test_timeout_default(self):
        cfg = MCPServerConfig(name="fs", command="npx")
        assert cfg.timeout == DEFAULT_TIMEOUT

    def test_connect_timeout_default(self):
        cfg = MCPServerConfig(name="fs", command="npx")
        assert cfg.connect_timeout == DEFAULT_CONNECT_TIMEOUT

    def test_custom_args(self):
        cfg = MCPServerConfig(name="fs", command="npx", args=["-y", "pkg"])
        assert cfg.args == ["-y", "pkg"]

    def test_custom_env(self):
        cfg = MCPServerConfig(name="fs", command="npx", env={"FOO": "bar"})
        assert cfg.env["FOO"] == "bar"


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------

class TestMCPTool:
    def test_fields_stored(self):
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object"},
            server_name="fs",
        )
        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert tool.server_name == "fs"

    def test_input_schema_stored(self):
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        tool = MCPTool(name="t", description="d", input_schema=schema, server_name="s")
        assert tool.input_schema == schema


# ---------------------------------------------------------------------------
# MCPCallResult
# ---------------------------------------------------------------------------

class TestMCPCallResult:
    def test_content_stored(self):
        r = MCPCallResult(content="hello")
        assert r.content == "hello"

    def test_is_error_defaults_false(self):
        r = MCPCallResult(content="ok")
        assert r.is_error is False

    def test_is_error_can_be_true(self):
        r = MCPCallResult(content="err", is_error=True)
        assert r.is_error is True


# ---------------------------------------------------------------------------
# _sanitize_error
# ---------------------------------------------------------------------------

class TestSanitizeError:
    def test_plain_message_unchanged(self):
        result = _sanitize_error("connection refused")
        assert result == "connection refused"

    def test_github_token_redacted(self):
        msg = "ghp_" + "a" * 40 + " failed"
        result = _sanitize_error(msg)
        assert "ghp_" not in result
        assert "[REDACTED]" in result

    def test_openai_key_redacted(self):
        msg = "sk-" + "x" * 40 + " error"
        result = _sanitize_error(msg)
        assert "sk-" not in result
        assert "[REDACTED]" in result

    def test_returns_string(self):
        assert isinstance(_sanitize_error("msg"), str)

    def test_empty_string_unchanged(self):
        assert _sanitize_error("") == ""


# ---------------------------------------------------------------------------
# to_mullet_schema
# ---------------------------------------------------------------------------

class TestToMulletSchema:
    def test_description_included(self):
        tool = MCPTool(name="t", description="do thing", input_schema={}, server_name="s")
        result = to_mullet_schema(tool)
        assert result["description"] == "do thing"

    def test_parameters_included(self):
        schema = {"type": "object", "properties": {}}
        tool = MCPTool(name="t", description="d", input_schema=schema, server_name="s")
        result = to_mullet_schema(tool)
        assert result["parameters"] == schema

    def test_empty_schema_gets_default(self):
        tool = MCPTool(name="t", description="d", input_schema={}, server_name="s")
        result = to_mullet_schema(tool)
        # empty dict is falsy — should use default
        assert "type" in result["parameters"]

    def test_returns_dict(self):
        tool = MCPTool(name="t", description="d", input_schema={}, server_name="s")
        assert isinstance(to_mullet_schema(tool), dict)


# ---------------------------------------------------------------------------
# load_mcp_servers
# ---------------------------------------------------------------------------

class TestLoadMcpServers:
    def test_empty_config_returns_empty(self):
        result = load_mcp_servers({})
        assert result == []

    def test_single_server_loaded(self):
        config = {"fs": {"command": "npx", "args": ["-y", "pkg"]}}
        result = load_mcp_servers(config)
        assert len(result) == 1
        assert result[0].name == "fs"
        assert result[0].command == "npx"

    def test_multiple_servers_loaded(self):
        config = {
            "fs": {"command": "npx"},
            "gh": {"command": "node"},
        }
        result = load_mcp_servers(config)
        assert len(result) == 2
        names = {s.name for s in result}
        assert "fs" in names
        assert "gh" in names

    def test_args_passed_through(self):
        config = {"s": {"command": "cmd", "args": ["--flag"]}}
        result = load_mcp_servers(config)
        assert result[0].args == ["--flag"]

    def test_missing_command_defaults_empty(self):
        config = {"s": {}}
        result = load_mcp_servers(config)
        assert result[0].command == ""

    def test_env_passed_through(self):
        config = {"s": {"command": "c", "env": {"KEY": "val"}}}
        result = load_mcp_servers(config)
        assert result[0].env == {"KEY": "val"}


# ---------------------------------------------------------------------------
# WebSourceConfig
# ---------------------------------------------------------------------------

class TestWebSourceConfig:
    def test_from_dict_url(self):
        cfg = WebSourceConfig.from_dict({"url": "https://example.com/docs"})
        assert cfg.url == "https://example.com/docs"

    def test_from_dict_defaults(self):
        cfg = WebSourceConfig.from_dict({})
        assert cfg.mode == "crawl"
        assert cfg.stay_in_path is True
        assert cfg.max_pages == 500

    def test_domain_property(self):
        cfg = WebSourceConfig(url="https://www.example.com/path")
        assert cfg.domain == "example.com"

    def test_base_url_property(self):
        cfg = WebSourceConfig(url="https://example.com/docs/intro")
        assert cfg.base_url == "https://example.com"

    def test_starting_path_property(self):
        cfg = WebSourceConfig(url="https://example.com/docs/intro")
        assert cfg.starting_path == "/docs/intro"

    def test_starting_path_empty_becomes_slash(self):
        cfg = WebSourceConfig(url="https://example.com")
        assert cfg.starting_path == "/"


# ---------------------------------------------------------------------------
# normalize_link
# ---------------------------------------------------------------------------

class TestNormalizeLink:
    BASE = "https://example.com"

    def test_empty_link_returns_none(self):
        assert normalize_link("", "/page", self.BASE) is None

    def test_mailto_returns_none(self):
        assert normalize_link("mailto:user@example.com", "/", self.BASE) is None

    def test_javascript_returns_none(self):
        assert normalize_link("javascript:void(0)", "/", self.BASE) is None

    def test_external_link_returns_none(self):
        assert normalize_link("https://other.com/page", "/", self.BASE) is None

    def test_absolute_path_returned(self):
        result = normalize_link("/docs/page", "/", self.BASE)
        assert result == "/docs/page"

    def test_fragment_stripped(self):
        result = normalize_link("/docs#anchor", "/", self.BASE)
        assert "#anchor" not in (result or "")

    def test_query_stripped(self):
        result = normalize_link("/docs?q=1", "/", self.BASE)
        assert "?" not in (result or "")

    def test_relative_link_resolved(self):
        result = normalize_link("sub/page", "/docs/", self.BASE)
        assert result is not None
        assert result.startswith("/")

    def test_full_url_same_domain_normalized(self):
        result = normalize_link("https://example.com/docs", "/", self.BASE)
        assert result == "/docs"


# ---------------------------------------------------------------------------
# _is_in_scope
# ---------------------------------------------------------------------------

class TestIsInScope:
    def _cfg(self, url: str, stay: bool = True) -> WebSourceConfig:
        return WebSourceConfig(url=url, stay_in_path=stay)

    def test_stay_in_path_false_always_true(self):
        cfg = self._cfg("https://example.com/docs", stay=False)
        assert _is_in_scope("/other/page", cfg) is True

    def test_root_path_always_in_scope(self):
        cfg = self._cfg("https://example.com")
        assert _is_in_scope("/any/path", cfg) is True

    def test_exact_match_in_scope(self):
        cfg = self._cfg("https://example.com/docs")
        assert _is_in_scope("/docs", cfg) is True

    def test_sub_path_in_scope(self):
        cfg = self._cfg("https://example.com/docs")
        assert _is_in_scope("/docs/intro", cfg) is True

    def test_out_of_scope(self):
        cfg = self._cfg("https://example.com/docs")
        assert _is_in_scope("/blog/post", cfg) is False


# ---------------------------------------------------------------------------
# _extract_links
# ---------------------------------------------------------------------------

class TestExtractLinks:
    def test_single_link_extracted(self):
        html = '<a href="/about">About</a>'
        links = _extract_links(html)
        assert "/about" in links

    def test_multiple_links_extracted(self):
        html = '<a href="/a">A</a><a href="/b">B</a>'
        links = _extract_links(html)
        assert "/a" in links
        assert "/b" in links

    def test_no_links_returns_empty(self):
        links = _extract_links("<p>No links here</p>")
        assert links == []

    def test_returns_list(self):
        assert isinstance(_extract_links(""), list)

    def test_duplicate_links_deduplicated(self):
        html = '<a href="/p">1</a><a href="/p">2</a>'
        links = _extract_links(html)
        assert links.count("/p") == 1


# ---------------------------------------------------------------------------
# _url_to_filepath
# ---------------------------------------------------------------------------

class TestUrlToFilepath:
    def test_simple_path(self):
        result = _url_to_filepath("/docs/intro")
        assert "docs" in result
        assert "intro" in result

    def test_empty_path_returns_index(self):
        result = _url_to_filepath("")
        assert "index" in result

    def test_slash_only_returns_index(self):
        result = _url_to_filepath("/")
        assert "index" in result

    def test_returns_md_extension(self):
        result = _url_to_filepath("/docs/page")
        assert result.endswith(".md")

    def test_no_slashes_in_filename(self):
        result = _url_to_filepath("/a/b/c")
        assert "/" not in result.lstrip("/")

    def test_returns_string(self):
        assert isinstance(_url_to_filepath("/page"), str)
