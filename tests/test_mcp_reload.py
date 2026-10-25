"""Tests for MCP server hot reload."""

import json

import pytest

from mullet.server.mcp import handle_request


@pytest.fixture(autouse=True)
def _no_real_module_drop(monkeypatch):
    """Never drop the test process's real mullet modules; only test sentinels."""
    import sys
    import mullet.server.mcp as mcp

    def fake_drop():
        for name in [k for k in sys.modules if k.startswith("mullet._reload")]:
            del sys.modules[name]

    monkeypatch.setattr(mcp, "_drop_mullet_modules", fake_drop)



@pytest.fixture(autouse=True)
def _reset_reload_state(monkeypatch):
    """Save and restore reload-related module state around each test."""
    from mullet.server import mcp as mcp_module

    saved = {
        "MCP_TOOLS": mcp_module.MCP_TOOLS,
        "MCP_HANDLERS": mcp_module.MCP_HANDLERS,
        "_reload_last_fingerprint": mcp_module._reload_last_fingerprint,
        "_reload_last_check": mcp_module._reload_last_check,
        "_reload_pending_warning": mcp_module._reload_pending_warning,
    }
    mcp_module._reload_last_check = -100.0
    mcp_module._reload_last_fingerprint = 0.0
    mcp_module._reload_pending_warning = None
    yield
    for key, value in saved.items():
        setattr(mcp_module, key, value)


def test_initialize_advertises_list_changed():
    resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["capabilities"] == {"tools": {"listChanged": True}}


def test_unchanged_fingerprint_no_reload(monkeypatch):
    from mullet.server import mcp as mcp_module

    monkeypatch.setattr(mcp_module, "_code_fingerprint", lambda: 1.0)
    mcp_module._reload_last_fingerprint = 1.0

    reload_calls = []
    monkeypatch.setattr(
        mcp_module,
        "_load_registries",
        lambda: (reload_calls.append("reload") or mcp_module.MCP_TOOLS, mcp_module.MCP_HANDLERS),
    )
    notifications = []
    monkeypatch.setattr(mcp_module, "_write_response", notifications.append)

    handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})

    assert len(reload_calls) == 0
    assert len(notifications) == 0


def test_changed_fingerprint_reloads_and_notifies(monkeypatch):
    from mullet.server import mcp as mcp_module

    new_tools = [{"name": "new_tool", "description": "new", "inputSchema": {"type": "object"}}]
    new_handlers = {"new_tool": lambda _args: "new result"}
    reload_calls = []

    def fake_load():
        reload_calls.append("reload")
        return new_tools, new_handlers

    monkeypatch.setattr(mcp_module, "_load_registries", fake_load)
    monkeypatch.setattr(mcp_module, "_code_fingerprint", lambda: 1.0)
    mcp_module._reload_last_fingerprint = 0.0

    notifications = []
    monkeypatch.setattr(mcp_module, "_write_response", notifications.append)

    resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})

    assert len(reload_calls) == 1
    assert len(notifications) == 1
    assert notifications[0] == {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
    assert resp["result"]["tools"] == new_tools

    # A second call should not trigger another reload; the new registries stay in place.
    resp2 = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert len(reload_calls) == 1
    assert len(notifications) == 1
    assert resp2["result"]["tools"] == new_tools


def test_five_second_throttle(monkeypatch):
    from mullet.server import mcp as mcp_module

    reload_calls = []

    def fake_load():
        reload_calls.append("reload")
        return mcp_module.MCP_TOOLS, mcp_module.MCP_HANDLERS

    monkeypatch.setattr(mcp_module, "_load_registries", fake_load)
    monkeypatch.setattr(mcp_module, "_code_fingerprint", lambda: 1.0)
    monkeypatch.setattr(mcp_module, "_write_response", lambda _msg: None)

    clock = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])

    mcp_module._reload_last_fingerprint = 0.0
    assert mcp_module._maybe_reload() is None
    assert len(reload_calls) == 1

    # Still changed, but too soon: throttle should suppress the second reload.
    mcp_module._reload_last_fingerprint = 0.0
    clock[0] = 3.0
    assert mcp_module._maybe_reload() is None
    assert len(reload_calls) == 1

    # Past the throttle window: reload again.
    mcp_module._reload_last_fingerprint = 0.0
    clock[0] = 6.0
    assert mcp_module._maybe_reload() is None
    assert len(reload_calls) == 2


def test_failed_reload_keeps_old_handlers_and_warns_once(monkeypatch):
    from mullet.server import mcp as mcp_module

    original_tools = mcp_module.MCP_TOOLS
    original_handlers = mcp_module.MCP_HANDLERS
    reload_calls = []

    def fake_load():
        reload_calls.append("reload")
        raise RuntimeError("broken import")

    monkeypatch.setattr(mcp_module, "_load_registries", fake_load)
    monkeypatch.setattr(mcp_module, "_code_fingerprint", lambda: 1.0)
    mcp_module._reload_last_fingerprint = 0.0

    notifications = []
    monkeypatch.setattr(mcp_module, "_write_response", notifications.append)

    resp = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mullet_status", "arguments": {}},
        }
    )

    assert len(reload_calls) == 1
    assert len(notifications) == 0
    assert mcp_module.MCP_TOOLS is original_tools
    assert mcp_module.MCP_HANDLERS is original_handlers

    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert data["status"] == "ok"

    assert len(resp["result"]["content"]) == 2
    warning = resp["result"]["content"][1]["text"]
    assert warning.startswith("Mullet's code changed but the new version failed to load")
    assert "RuntimeError: broken import" in warning

    # The warning must appear only once.
    resp2 = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "mullet_status", "arguments": {}},
        }
    )
    assert len(resp2["result"]["content"]) == 1


def test_reload_disabled_by_env(monkeypatch):
    from mullet.server import mcp as mcp_module

    monkeypatch.setenv("MULLET_MCP_RELOAD", "off")
    monkeypatch.setattr(mcp_module, "_code_fingerprint", lambda: 1.0)
    mcp_module._reload_last_fingerprint = 0.0

    reload_calls = []
    monkeypatch.setattr(
        mcp_module,
        "_load_registries",
        lambda: (reload_calls.append("reload") or mcp_module.MCP_TOOLS, mcp_module.MCP_HANDLERS),
    )
    notifications = []
    monkeypatch.setattr(mcp_module, "_write_response", notifications.append)

    handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})

    assert len(reload_calls) == 0
    assert len(notifications) == 0


def test_failed_reload_restores_the_old_modules(monkeypatch):
    import sys
    import mullet.server.mcp as mcp
    sentinel = type(sys)("mullet._reload_sentinel")
    monkeypatch.setitem(sys.modules, "mullet._reload_sentinel", sentinel)

    def broken():
        sys.modules["mullet._reload_half_loaded"] = type(sys)("mullet._half_loaded")
        raise SyntaxError("bad edit")

    monkeypatch.setattr(mcp, "_load_registries", broken)
    monkeypatch.setattr(mcp, "_code_fingerprint", lambda: mcp._reload_last_fingerprint + 1)
    monkeypatch.setattr(mcp, "_reload_last_check", 0.0)
    monkeypatch.setattr(mcp, "_write_response", lambda msg: None)
    monkeypatch.delenv("MULLET_MCP_RELOAD", raising=False)
    assert "failed to load" in mcp._maybe_reload()
    assert sys.modules.get("mullet._reload_sentinel") is sentinel
    assert "mullet._reload_half_loaded" not in sys.modules


def test_importing_the_server_keeps_already_loaded_modules():
    import importlib
    import sys
    import mullet.core.paths as paths_before
    import mullet.server.mcp as mcp
    importlib.reload(mcp)
    assert sys.modules["mullet.core.paths"] is paths_before
