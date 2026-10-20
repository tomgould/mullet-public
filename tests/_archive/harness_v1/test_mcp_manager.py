"""Tests for mullet.harness.mcp.manager — config loading + transport detection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mullet.harness.mcp.manager import MCPManager, MCPServerConfig

pytestmark = pytest.mark.harness


def test_manager_loads_empty_config(tmp_path: Path) -> None:
    """No config files → empty server list."""
    mgr = MCPManager(config_paths=[tmp_path / "missing.json"])
    assert mgr.load_config() == []


def test_manager_loads_project_mcp_json(tmp_path: Path) -> None:
    cfg = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
                "env": {"FOO": "bar"},
            }
        }
    }
    path = tmp_path / ".mcp.json"
    path.write_text(json.dumps(cfg))
    mgr = MCPManager(config_paths=[path])
    servers = mgr.load_config()
    assert len(servers) == 1
    s = servers[0]
    assert s.name == "filesystem"
    assert s.type == "stdio"
    assert s.command == "npx"
    assert s.args == ["-y", "@anthropic/mcp-server-filesystem", "/tmp"]
    assert s.env == {"FOO": "bar"}
    assert s.enabled is True


def test_manager_loads_legacy_servers_key(tmp_path: Path) -> None:
    """Older configs use 'servers' instead of 'mcpServers'."""
    cfg = {"servers": {"foo": {"command": "echo"}}}
    path = tmp_path / ".mcp.json"
    path.write_text(json.dumps(cfg))
    mgr = MCPManager(config_paths=[path])
    servers = mgr.load_config()
    assert len(servers) == 1
    assert servers[0].name == "foo"


def test_manager_merges_project_and_user_config(tmp_path: Path) -> None:
    project = tmp_path / ".mcp.json"
    user = tmp_path / "user_mcp.json"
    project.write_text(
        json.dumps(
            {"mcpServers": {"a": {"command": "cmd-a"}, "shared": {"command": "project"}}}
        )
    )
    user.write_text(
        json.dumps(
            {"mcpServers": {"b": {"command": "cmd-b"}, "shared": {"command": "user-override"}}}
        )
    )
    mgr = MCPManager(config_paths=[project, user])
    servers = {s.name: s for s in mgr.load_config()}
    assert set(servers.keys()) == {"a", "b", "shared"}
    # Later config (user) overrides project.
    assert servers["shared"].command == "user-override"


def test_manager_skips_unreadable_config(tmp_path: Path) -> None:
    bad = tmp_path / ".mcp.json"
    bad.write_text("{not json}")
    mgr = MCPManager(config_paths=[bad])
    assert mgr.load_config() == []


def test_manager_detects_transport_stdio_from_command_only() -> None:
    body = {"command": "npx", "args": ["-y", "x"]}
    assert MCPManager._detect_transport_type(body) == "stdio"


def test_manager_detects_transport_http_from_url() -> None:
    body = {"url": "https://example.com/mcp"}
    assert MCPManager._detect_transport_type(body) == "http"


def test_manager_detects_transport_explicit_type_field_wins() -> None:
    # Explicit type=http even though only command is set.
    body = {"type": "http", "command": "weird", "url": "https://example.com/mcp"}
    assert MCPManager._detect_transport_type(body) == "http"
    # Explicit type=stdio even though url is set.
    body2 = {"type": "stdio", "url": "https://example.com/mcp", "command": "x"}
    assert MCPManager._detect_transport_type(body2) == "stdio"


def test_manager_legacy_sse_maps_to_http() -> None:
    body = {"type": "sse", "url": "https://example.com/sse"}
    assert MCPManager._detect_transport_type(body) == "http"


def test_manager_streamable_http_alias() -> None:
    body = {"type": "streamable-http", "url": "https://example.com/mcp"}
    assert MCPManager._detect_transport_type(body) == "http"


def test_manager_default_falls_back_to_stdio_with_no_clues() -> None:
    body: dict[str, Any] = {}
    assert MCPManager._detect_transport_type(body) == "stdio"


def test_manager_qualified_tool_name_resolution() -> None:
    """list_tools and call routing namespace by server name."""
    mgr = MCPManager(config_paths=[])
    # Inject fake state directly: a "client" stub and tool_index entries.

    class FakeClient:
        initialized = True
        transport = SimpleNamespace()  # not used here

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls.append((name, arguments))
            return {"content": [{"type": "text", "text": f"{name}-ok"}], "isError": False}

        async def close(self) -> None:
            pass

    fs_client = FakeClient()
    gh_client = FakeClient()
    mgr.clients["filesystem"] = fs_client  # type: ignore[assignment]
    mgr.clients["github"] = gh_client  # type: ignore[assignment]
    mgr.tool_index["filesystem.read_file"] = (
        "filesystem",
        {"name": "read_file", "description": "Read a file"},
    )
    mgr.tool_index["github.read_file"] = (
        "github",
        {"name": "read_file", "description": "Read from a GH repo"},
    )

    listed = {t["name"]: t for t in mgr.list_tools()}
    assert "filesystem.read_file" in listed
    assert "github.read_file" in listed
    # Server-attribution tag for callers that want to display it.
    assert listed["filesystem.read_file"]["server"] == "filesystem"

    # Routing: call("github.read_file") goes to github client only.
    import asyncio

    asyncio.run(mgr.call("github.read_file", {"path": "x"}))
    assert gh_client.calls == [("read_file", {"path": "x"})]
    assert fs_client.calls == []


def test_manager_call_unknown_tool_raises() -> None:
    mgr = MCPManager(config_paths=[])
    import asyncio

    with pytest.raises(KeyError):
        asyncio.run(mgr.call("nope.nope", {}))


def test_manager_disconnected_server_call_raises() -> None:
    mgr = MCPManager(config_paths=[])
    mgr.tool_index["ghost.tool"] = ("ghost", {"name": "tool"})
    # ghost not in mgr.clients
    import asyncio

    with pytest.raises(KeyError):
        asyncio.run(mgr.call("ghost.tool", {}))


def test_manager_disabled_servers_appear_in_config() -> None:
    cfg = {"mcpServers": {"x": {"command": "cmd", "enabled": False}}}
    path = Path("/tmp/_mullet_disabled_test.json")
    path.write_text(json.dumps(cfg))
    try:
        mgr = MCPManager(config_paths=[path])
        servers = mgr.load_config()
        assert len(servers) == 1
        assert servers[0].enabled is False
    finally:
        path.unlink(missing_ok=True)


def test_manager_server_config_model_validates() -> None:
    cfg = MCPServerConfig(name="x", command="cmd")
    assert cfg.type == "stdio"
    assert cfg.enabled is True
    assert cfg.args == []
