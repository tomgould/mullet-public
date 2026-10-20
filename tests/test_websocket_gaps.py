"""Tests for missing WebSocket coverage gaps."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from aiohttp import web  # noqa: F401
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")

if HAS_AIOHTTP:
    from mullet.server.websocket import WebSocketHandler, SESSION_TTL


@pytest.fixture
def app():
    from mullet.server.app import create_app
    import mullet.server.middleware as mw
    mw._rate_limiter._counters.clear()
    return create_app()


@pytest.fixture(autouse=True)
def _reset_handler_state():
    """Reset class-level state before every test."""
    if not HAS_AIOHTTP:
        return
    WebSocketHandler._sessions.clear()
    WebSocketHandler._cleanup_started = True


@pytest.mark.asyncio
async def test_cleanup_stale_sessions():
    """Stale sessions should be removed after one cleanup iteration."""
    now = time.time()
    WebSocketHandler._sessions["stale"] = {"last_seen": now - SESSION_TTL - 1}
    WebSocketHandler._sessions["fresh"] = {"last_seen": now - 10}

    call_count = 0

    async def _fake_sleep(delay):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError()

    with patch("mullet.server.websocket.asyncio.sleep", side_effect=_fake_sleep):
        try:
            await WebSocketHandler._cleanup_stale_sessions()
        except asyncio.CancelledError:
            pass

    assert "stale" not in WebSocketHandler._sessions
    assert "fresh" in WebSocketHandler._sessions


@pytest.mark.asyncio
async def test_ws_search_valid(aiohttp_client, app):
    """search with valid project+query should return search_results."""
    mock_km = MagicMock()
    mock_km.search.return_value = [{"chunk": "result"}]
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_km)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.core.knowledge.manager.knowledge_session", return_value=mock_ctx):
        await ws.send_json({
            "type": "search",
            "data": {"project": "myproj", "query": "test"},
        })
        resp = await ws.receive_json()

    assert resp["type"] == "search_results"
    assert resp["data"]["results"] == [{"chunk": "result"}]
    await ws.close()


@pytest.mark.asyncio
async def test_ws_search_exception(aiohttp_client, app):
    """search should return error when knowledge_session raises."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(side_effect=Exception("boom"))
    mock_ctx.__exit__ = MagicMock(return_value=False)

    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.core.knowledge.manager.knowledge_session", return_value=mock_ctx):
        await ws.send_json({
            "type": "search",
            "data": {"project": "myproj", "query": "test"},
        })
        resp = await ws.receive_json()

    assert resp["type"] == "error"
    assert resp["data"]["message"] == "Search failed"
    await ws.close()


@pytest.mark.asyncio
async def test_ws_kill_task_nonexistent(aiohttp_client, app):
    """kill_task with unknown task_id should not crash or respond."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    await ws.send_json({
        "type": "kill_task",
        "data": {"task_id": "nonexistent"},
    })

    # Verify connection is still alive by pinging
    await ws.send_json({"type": "ping"})
    resp = await ws.receive_json()
    assert resp["type"] == "pong"
    await ws.close()


@pytest.mark.asyncio
async def test_ws_start_task_streams_token_and_tool(aiohttp_client, app):
    """start_task should stream token_stream and tool_start events."""
    mock_result = MagicMock()
    mock_result.final_response = "done"
    mock_result.tokens_used.total_tokens = 5
    mock_result.tool_calls_made = 1

    def mock_build(prompt, opts, on_token=None, on_tool_call=None):
        mock_agent = MagicMock()

        def run(p):
            if on_token:
                on_token("hello")
            if on_tool_call:
                on_tool_call("test_tool", '{"x": 1}')
            # Small delay so the main loop has time to process the tool event
            # before agent_future becomes done.
            import time as _time
            _time.sleep(0.05)
            return mock_result

        mock_agent.run = run
        return mock_agent, "mock"

    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.api._build_server_agent", side_effect=mock_build):
        await ws.send_json({
            "type": "start_task",
            "data": {"prompt": "test", "project": "myproj"},
        })

        messages = []
        for _ in range(10):
            msg = await ws.receive_json(timeout=2)
            messages.append(msg)
            if msg.get("type") in ("task_complete", "task_error"):
                break

    types = [m["type"] for m in messages]
    assert "task_started" in types
    assert "token_stream" in types
    assert "tool_start" in types
    assert "task_complete" in types
    await ws.close()


# ---------------------------------------------------------------------------
# New tests for previously uncovered lines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ws_authenticate_verify_pat_success(aiohttp_client, app):
    """authenticate with db+token should succeed when verify_pat returns user."""
    mock_db = MagicMock()
    app["db"] = mock_db
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.auth.verify_pat", return_value={"id": 1, "name": "Test"}):
        await ws.send_json({"type": "authenticate", "data": {"token": "valid-token"}})
        resp = await ws.receive_json()

    assert resp["type"] == "authenticated"
    assert resp["data"] == {"id": 1, "name": "Test"}
    await ws.close()


@pytest.mark.asyncio
async def test_ws_authenticate_verify_pat_failure(aiohttp_client, app):
    """authenticate should return auth_failed when verify_pat returns None."""
    mock_db = MagicMock()
    app["db"] = mock_db
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.auth.verify_pat", return_value=None):
        await ws.send_json({"type": "authenticate", "data": {"token": "bad-token"}})
        resp = await ws.receive_json()

    assert resp["type"] == "error"
    assert resp["data"]["code"] == "auth_failed"
    await ws.close()


@pytest.mark.asyncio
async def test_ws_authenticate_solo_fallback(aiohttp_client, app):
    """authenticate should fall back to get_solo_user when db or token missing."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.auth.get_solo_user", return_value={"id": "solo"}):
        await ws.send_json({"type": "authenticate", "data": {}})
        resp = await ws.receive_json()

    assert resp["type"] == "authenticated"
    assert resp["data"] == {"id": "solo"}
    await ws.close()


@pytest.mark.asyncio
async def test_ws_list_projects_auth_required(aiohttp_client, app):
    """list_projects should require auth in non-solo mode."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.websocket.MulletMode.is_solo", return_value=False):
        await ws.send_json({"type": "list_projects"})
        resp = await ws.receive_json()

    assert resp["type"] == "error"
    assert resp["data"]["message"] == "Authentication required"
    await ws.close()


@pytest.mark.asyncio
async def test_ws_search_auth_required(aiohttp_client, app):
    """search should require auth in non-solo mode."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.websocket.MulletMode.is_solo", return_value=False):
        await ws.send_json({"type": "search", "data": {"project": "p", "query": "q"}})
        resp = await ws.receive_json()

    assert resp["type"] == "error"
    assert resp["data"]["message"] == "Authentication required"
    await ws.close()


@pytest.mark.asyncio
async def test_ws_start_task_with_providers(aiohttp_client, app):
    """start_task should pass providers from payload to _build_server_agent opts."""
    mock_result = MagicMock()
    mock_result.final_response = "done"
    mock_result.tokens_used.total_tokens = 3
    mock_result.tool_calls_made = 0

    captured = {}

    def mock_build(prompt, opts, on_token=None, on_tool_call=None):
        captured["opts"] = opts
        mock_agent = MagicMock()
        mock_agent.run = lambda p: mock_result
        return mock_agent, "mock"

    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.api._build_server_agent", side_effect=mock_build):
        await ws.send_json({
            "type": "start_task",
            "data": {"prompt": "test", "project": "myproj", "providers": ["openai"]},
        })

        messages = []
        for _ in range(10):
            msg = await ws.receive_json(timeout=2)
            messages.append(msg)
            if msg.get("type") in ("task_complete", "task_error"):
                break

    assert captured["opts"].get("providers") == ["openai"]
    await ws.close()


@pytest.mark.asyncio
async def test_ws_kill_task_existing(aiohttp_client, app):
    """kill_task with existing task_id should cancel it and send task_error."""

    def mock_build_slow(prompt, opts, on_token=None, on_tool_call=None):
        mock_agent = MagicMock()

        def run(p):
            import time as _time
            _time.sleep(0.5)
            return MagicMock()

        mock_agent.run = run
        return mock_agent, "mock"

    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.api._build_server_agent", side_effect=mock_build_slow):
        await ws.send_json({
            "type": "start_task",
            "data": {"prompt": "test", "project": "myproj"},
        })

        # Wait for task_started
        msg = await ws.receive_json(timeout=2)
        assert msg["type"] == "task_started"
        task_id = msg["data"]["task_id"]

        # Kill the task
        await ws.send_json({
            "type": "kill_task",
            "data": {"task_id": task_id},
        })

        # Collect until we see task_error
        found = False
        for _ in range(5):
            msg = await ws.receive_json(timeout=2)
            if msg.get("type") == "task_error" and msg.get("data", {}).get("error") == "Task cancelled":
                found = True
                break

    assert found
    await ws.close()


@pytest.mark.asyncio
async def test_ws_start_task_no_provider(aiohttp_client, app):
    """start_task should error when _build_server_agent returns no agent."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.api._build_server_agent", return_value=(None, None)):
        await ws.send_json({
            "type": "start_task",
            "data": {"prompt": "test", "project": "myproj"},
        })

        msg = await ws.receive_json(timeout=2)
        assert msg["type"] == "task_started"

        msg = await ws.receive_json(timeout=2)
        assert msg["type"] == "task_error"
        assert "No provider available" in msg["data"]["error"]

    await ws.close()


@pytest.mark.asyncio
async def test_ws_start_task_streams_tool_result(aiohttp_client, app):
    """start_task should stream tool_result events in loop and drain, plus task_complete."""
    mock_result = MagicMock()
    mock_result.final_response = "done"
    mock_result.tokens_used.total_tokens = 5
    mock_result.tool_calls_made = 2

    def mock_build(prompt, opts, on_token=None, on_tool_call=None):
        mock_agent = MagicMock()

        def run(p):
            # Queue an early tool_result so it is processed in the streaming loop
            if mock_agent.on_tool_result:
                mock_agent.on_tool_result("early", "preview_early")
            # Sleep so the main loop times out a couple of times
            import time as _time
            _time.sleep(0.25)
            # Queue tool_results first, then token, so token is drained
            if mock_agent.on_tool_result:
                for i in range(3):
                    mock_agent.on_tool_result(f"late{i}", f"preview{i}")
            if on_token:
                on_token("hello")
            return mock_result

        mock_agent.run = run
        mock_agent.on_tool_result = None
        return mock_agent, "mock"

    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.api._build_server_agent", side_effect=mock_build):
        await ws.send_json({
            "type": "start_task",
            "data": {"prompt": "test", "project": "myproj"},
        })

        messages = []
        for _ in range(20):
            msg = await ws.receive_json(timeout=2)
            messages.append(msg)
            if msg.get("type") in ("task_complete", "task_error"):
                break

    types = [m["type"] for m in messages]
    assert "tool_result" in types
    tool_results = [m for m in messages if m["type"] == "tool_result"]
    assert len(tool_results) >= 2
    assert "task_complete" in types
    await ws.close()


@pytest.mark.asyncio
async def test_execute_task_cancelled_error():
    """_execute_task should handle asyncio.CancelledError gracefully."""
    handler = WebSocketHandler()
    ws = MagicMock()
    ws.closed = False
    ws.send_json = AsyncMock()

    with patch("mullet.server.api._build_server_agent", side_effect=asyncio.CancelledError):
        await handler._execute_task(ws, "task-1", {"prompt": "test"})

    calls = [c.args[0] for c in ws.send_json.call_args_list]
    assert calls[0]["type"] == "task_started"
    assert calls[1]["type"] == "task_error"
    assert calls[1]["data"]["error"] == "Task cancelled"


@pytest.mark.asyncio
async def test_execute_task_generic_exception():
    """_execute_task should handle generic exceptions gracefully."""
    handler = WebSocketHandler()
    ws = MagicMock()
    ws.closed = False
    ws.send_json = AsyncMock()

    with patch("mullet.server.api._build_server_agent", side_effect=Exception("boom")):
        await handler._execute_task(ws, "task-1", {"prompt": "test"})

    calls = [c.args[0] for c in ws.send_json.call_args_list]
    assert calls[0]["type"] == "task_started"
    assert calls[1]["type"] == "task_error"
    assert "Task execution failed" in calls[1]["data"]["error"]


@pytest.mark.asyncio
async def test_ws_invalid_json(aiohttp_client, app):
    """Invalid JSON should return an error message."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_str("not json at all")
    resp = await ws.receive_json()
    assert resp["type"] == "error"
    assert "Invalid JSON" in resp["data"]["message"]
    await ws.close()


@pytest.mark.asyncio
async def test_ws_resume_session_success(aiohttp_client, app):
    """resume_session with a valid session_id should resume the session."""
    handler = WebSocketHandler()
    sid = "test-sid"
    handler._sessions[sid] = {"user": None, "last_seen": time.time()}

    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    await ws.send_json({"type": "resume_session", "data": {"session_id": sid}})
    resp = await ws.receive_json()

    assert resp["type"] == "session_resumed"
    assert resp["data"]["session_id"] == sid
    await ws.close()


@pytest.mark.asyncio
async def test_ws_resume_session_unknown(aiohttp_client, app):
    """resume_session with an unknown id should create a new session."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    await ws.send_json({"type": "resume_session", "data": {"session_id": "unknown"}})
    resp = await ws.receive_json()

    assert resp["type"] == "session_created"
    assert resp["data"]["session_id"] != "unknown"
    await ws.close()


@pytest.mark.asyncio
async def test_ws_list_projects_success(aiohttp_client, app):
    """list_projects should return a projects list."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    await ws.send_json({"type": "list_projects"})
    resp = await ws.receive_json()

    assert resp["type"] == "projects_list"
    assert isinstance(resp["data"]["projects"], list)
    await ws.close()


@pytest.mark.asyncio
async def test_ws_search_missing_params(aiohttp_client, app):
    """search without project/query should return an error."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    await ws.send_json({"type": "search", "data": {}})
    resp = await ws.receive_json()

    assert resp["type"] == "error"
    assert "required" in resp["data"]["message"]
    await ws.close()


@pytest.mark.asyncio
async def test_ws_start_task_auth_required(aiohttp_client, app):
    """start_task should require auth in non-solo mode."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    with patch("mullet.server.websocket.MulletMode.is_solo", return_value=False):
        await ws.send_json({"type": "start_task", "data": {"prompt": "test"}})
        resp = await ws.receive_json()

    assert resp["type"] == "error"
    assert resp["data"]["message"] == "Authentication required"
    await ws.close()


@pytest.mark.asyncio
async def test_ws_max_concurrent_tasks(aiohttp_client, app):
    """Starting more than MAX_TASKS_PER_CONNECTION should error."""
    from mullet.server.websocket import MAX_TASKS_PER_CONNECTION

    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    for i in range(MAX_TASKS_PER_CONNECTION):
        await ws.send_json({"type": "start_task", "data": {"prompt": f"task {i}"}})

    await ws.send_json({"type": "start_task", "data": {"prompt": "one too many"}})

    found_error = False
    for _ in range(20):
        try:
            resp = await ws.receive_json(timeout=2)
            if resp.get("type") == "error" and "concurrent" in resp.get("data", {}).get("message", "").lower():
                found_error = True
                break
        except Exception:
            break

    assert found_error
    await ws.close()


@pytest.mark.asyncio
async def test_ws_unknown_type(aiohttp_client, app):
    """Unknown message type should return an error."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "bogus_type"})
    resp = await ws.receive_json()
    assert resp["type"] == "error"
    assert "Unknown type" in resp["data"]["message"]
    await ws.close()


@pytest.mark.asyncio
async def test_start_cleanup_schedules_task():
    """start_cleanup should schedule the cleanup loop exactly once."""
    WebSocketHandler._cleanup_started = False
    WebSocketHandler._cleanup_task = None

    WebSocketHandler.start_cleanup()

    assert WebSocketHandler._cleanup_started is True
    assert WebSocketHandler._cleanup_task is not None

    WebSocketHandler._cleanup_task.cancel()
    try:
        await WebSocketHandler._cleanup_task
    except asyncio.CancelledError:
        pass
