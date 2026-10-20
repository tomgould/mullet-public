"""Tests for mullet.server.mcp hardening — pure unit tests, no real I/O."""
import json
import sys
import threading
import time
from io import StringIO

import pytest

from mullet.server import mcp


# ---------------------------------------------------------------------------
# handle_request
# ---------------------------------------------------------------------------

class TestHandleRequest:
    @pytest.fixture(autouse=True)
    def _patch_tools(self, monkeypatch):
        monkeypatch.setattr(mcp, "MCP_TOOLS", [{"name": "test_tool"}])
        monkeypatch.setattr(mcp, "MCP_HANDLERS", {})

    def test_initialize(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "mullet"

    def test_notifications_initialized(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert resp is None

    def test_tools_list(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 2
        assert resp["result"]["tools"] == [{"name": "test_tool"}]

    def test_tools_call_success_string_result(self, monkeypatch):
        monkeypatch.setattr(mcp, "MCP_HANDLERS", {"my_tool": lambda args: "hello"})
        resp = mcp.handle_request({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "my_tool", "arguments": {}},
        })
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 3
        assert resp["result"]["content"][0]["text"] == "hello"
        assert "isError" not in resp["result"]

    def test_tools_call_success_dict_result(self, monkeypatch):
        monkeypatch.setattr(mcp, "MCP_HANDLERS", {"my_tool": lambda args: {"ok": True}})
        resp = mcp.handle_request({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "my_tool", "arguments": {}},
        })
        assert resp["result"]["content"][0]["text"] == json.dumps({"ok": True})

    def test_tools_call_unknown_tool(self):
        resp = mcp.handle_request({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        })
        assert resp["result"]["isError"] is True
        assert "Unknown tool: missing" in resp["result"]["content"][0]["text"]

    def test_tools_call_handler_raises(self, monkeypatch):
        def _boom(args):
            raise RuntimeError("boom")
        monkeypatch.setattr(mcp, "MCP_HANDLERS", {"my_tool": _boom})
        resp = mcp.handle_request({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "my_tool", "arguments": {}},
        })
        assert resp["result"]["isError"] is True
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert "boom" in payload["error"]

    def test_ping(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "id": 7, "method": "ping"})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 7
        assert resp["result"] == {}

    def test_unknown_method(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "id": 8, "method": "foo/bar"})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 8
        assert resp["error"]["code"] == -32601
        assert "foo/bar" in resp["error"]["message"]

    def test_request_without_id(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "method": "ping"})
        assert resp["id"] is None
        assert resp["result"] == {}

    def test_request_without_params(self):
        monkeypatch = pytest.MonkeyPatch()
        with monkeypatch.context() as m:
            m.setattr(mcp, "MCP_HANDLERS", {"my_tool": lambda args: "hi"})
            resp = mcp.handle_request({
                "jsonrpc": "2.0", "id": 9, "method": "tools/call",
                "params": {"name": "my_tool"},
            })
            assert resp["result"]["content"][0]["text"] == "hi"


# ---------------------------------------------------------------------------
# run_mcp_server
# ---------------------------------------------------------------------------

class TestRunMcpServer:
    @pytest.fixture(autouse=True)
    def _patch_tools(self, monkeypatch):
        monkeypatch.setattr(mcp, "MCP_TOOLS", [])
        monkeypatch.setattr(mcp, "MCP_HANDLERS", {})

    def _run_server(self, stdin_lines, timeout=2.0):
        """Feed lines to run_mcp_server and capture stdout/stderr."""
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        sys.stdin = StringIO("\n".join(stdin_lines) + "\n")
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # Start server in a thread so we can enforce a timeout
        exc = None
        def _target():
            nonlocal exc
            try:
                mcp.run_mcp_server()
            except Exception as e:
                exc = e

        t = threading.Thread(target=_target)
        t.start()
        t.join(timeout=timeout)

        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()

        sys.stdin = old_stdin
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        if t.is_alive():
            # The server is stuck waiting for more input; treat as timeout,
            # but still return captured output for assertions.
            pass

        if exc:
            raise exc

        return stdout, stderr

    def test_valid_request_processed(self):
        line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        stdout, stderr = self._run_server([line, ""])
        assert "mullet mcp: server ready" in stderr
        assert "mullet mcp: server stopped" in stderr
        responses = [l for l in stdout.splitlines() if l.strip()]
        assert len(responses) == 1
        resp = json.loads(responses[0])
        assert resp["id"] == 1
        assert resp["result"] == {}

    def test_invalid_json_skipped(self):
        stdout, stderr = self._run_server(["not json", ""])
        assert "invalid JSON ignored" in stderr
        assert stdout.strip() == ""

    def test_non_object_json_skipped(self):
        stdout, stderr = self._run_server([json.dumps("bare string"), ""])
        assert "non-object JSON ignored" in stderr
        assert stdout.strip() == ""

    def test_oserror_on_stdin_causes_clean_exit(self, monkeypatch):
        class BadStdin:
            def readline(self):
                raise OSError("bad fd")
        monkeypatch.setattr(sys, "stdin", BadStdin())
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)
        mcp.run_mcp_server()
        assert "stdin closed" in stderr.getvalue()
        assert "mullet mcp: server stopped" in stderr.getvalue()

    def test_eof_empty_readline_causes_clean_exit(self):
        stdout, stderr = self._run_server([""])
        assert "mullet mcp: server ready" in stderr
        assert "mullet mcp: server stopped" in stderr
        assert stdout.strip() == ""

    def test_startup_shutdown_messages_to_stderr(self):
        line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        stdout, stderr = self._run_server([line, ""])
        assert "mullet mcp: server ready (stdio transport)\n" in stderr
        assert "mullet mcp: server stopped\n" in stderr

    def test_keyboard_interrupt_clean_exit(self, monkeypatch):
        class InterruptStdin:
            def readline(self):
                raise KeyboardInterrupt()
        monkeypatch.setattr(sys, "stdin", InterruptStdin())
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)
        mcp.run_mcp_server()
        assert "mullet mcp: server stopped" in stderr.getvalue()

    def test_tools_call_dispatched_on_thread(self, monkeypatch):
        calls = []
        def _handler(args):
            calls.append(args)
            return "done"
        monkeypatch.setattr(mcp, "MCP_HANDLERS", {"my_tool": _handler})

        line = json.dumps({
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {"name": "my_tool", "arguments": {"x": 1}},
        })
        stdout, stderr = self._run_server([line, ""], timeout=3.0)
        # Give the worker thread a moment to finish
        time.sleep(0.3)
        assert len(calls) == 1
        assert calls[0] == {"x": 1}
        responses = [l for l in stdout.splitlines() if l.strip()]
        assert len(responses) == 1
        resp = json.loads(responses[0])
        assert resp["id"] == 10
        assert resp["result"]["content"][0]["text"] == "done"
