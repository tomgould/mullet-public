"""WebSocket handler for real-time agent execution streaming.

Each connection can run multiple concurrent tasks. Events follow
the protocol defined in 06-phase-4-api-server.md.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from aiohttp import web, WSMsgType

from mullet.config.mode import MulletMode

logger = logging.getLogger(__name__)

MAX_TASKS_PER_CONNECTION = 3
SESSION_TTL = 300  # seconds


class WebSocketHandler:
    """Handle WebSocket connections for real-time agent execution."""

    _sessions: dict[str, dict[str, Any]] = {}
    _cleanup_started = False
    _cleanup_task: asyncio.Future | None = None  # strong ref so task isn't GC'd

    def __init__(self):
        pass  # Cleanup task started by app.on_startup — not here (no event loop at init time)

    @classmethod
    def start_cleanup(cls) -> None:
        """Schedule the stale-session cleanup loop. Call from an on_startup hook."""
        if not cls._cleanup_started:
            cls._cleanup_started = True
            # Keep a strong reference so the task isn't GC'd. RUF006.
            cls._cleanup_task = asyncio.ensure_future(cls._cleanup_stale_sessions())

    @classmethod
    async def _cleanup_stale_sessions(cls):
        while True:
            await asyncio.sleep(60)
            now = time.time()
            stale = [
                sid for sid, s in cls._sessions.items()
                if now - s.get("last_seen", 0) > SESSION_TTL
            ]
            for sid in stale:
                del cls._sessions[sid]
                logger.info("WebSocket session %s expired and cleaned up", sid)

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket connection and route incoming messages."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        active_tasks: dict[str, asyncio.Task] = {}
        current_user = None  # Track authenticated user for this connection
        session_id: str = str(uuid.uuid4())

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "data": {"message": "Invalid JSON"}})
                        continue

                    msg_type = data.get("type", "")
                    payload = data.get("data") or {}

                    if msg_type == "resume_session":
                        sid = payload.get("session_id", "")
                        if sid and sid in self._sessions:
                            session = self._sessions[sid]
                            current_user = session.get("user")
                            session_id = sid
                            session["last_seen"] = time.time()
                            logger.info("WebSocket session %s resumed", sid)
                            await ws.send_json({
                                "type": "session_resumed",
                                "data": {"session_id": sid},
                            })
                        else:
                            session_id = str(uuid.uuid4())
                            logger.info(
                                "WebSocket session %s started (resume failed for %s)",
                                session_id, sid or "<none>",
                            )
                            await ws.send_json({
                                "type": "session_created",
                                "data": {"session_id": session_id},
                            })
                        continue

                    if msg_type == "ping":
                        await ws.send_json({"type": "pong"})

                    elif msg_type == "authenticate":
                        # Validate token and return user info
                        token = payload.get("token", "")
                        db = request.app.get("db")
                        if db and token:
                            from mullet.server.auth import verify_pat
                            user = verify_pat(token, db)
                            if user:
                                current_user = user
                                await ws.send_json({"type": "authenticated", "data": user})
                            else:
                                await ws.send_json({"type": "error", "data": {"code": "auth_failed", "message": "Invalid token"}})
                        elif MulletMode.is_solo():
                            from mullet.server.auth import get_solo_user
                            current_user = get_solo_user()
                            await ws.send_json({"type": "authenticated", "data": get_solo_user()})
                        else:
                            await ws.send_json({"type": "error", "data": {"code": "auth_failed", "message": "Token required"}})

                    elif msg_type == "list_projects":
                        if current_user is None and not MulletMode.is_solo():
                            await ws.send_json({"type": "error", "data": {"message": "Authentication required"}})
                            continue
                        db = request.app.get("db")
                        projects = []
                        if db:
                            rows = db.execute("SELECT id, name, pool_name, description FROM projects").fetchall()
                            projects = [{"id": r[0], "name": r[1], "client": r[2], "description": r[3]} for r in rows]
                        await ws.send_json({"type": "projects_list", "data": {"projects": projects}})

                    elif msg_type == "search":
                        if current_user is None and not MulletMode.is_solo():
                            await ws.send_json({"type": "error", "data": {"message": "Authentication required"}})
                            continue
                        project = payload.get("project", "")
                        query = payload.get("query", "")
                        if project and query:
                            try:
                                from mullet.core.knowledge.manager import knowledge_session
                                with knowledge_session(project) as km:
                                    results = km.search(query, top_k=payload.get("top_k", 10))
                                await ws.send_json({"type": "search_results", "data": {"results": results}})
                            except Exception as e:
                                logger.warning("WebSocket search failed: %s", e)
                                await ws.send_json({"type": "error", "data": {"message": "Search failed"}})
                        else:
                            await ws.send_json({"type": "error", "data": {"message": "project and query required"}})

                    elif msg_type == "start_task":
                        if current_user is None and not MulletMode.is_solo():
                            await ws.send_json({"type": "error", "data": {"message": "Authentication required"}})
                            continue
                        if len(active_tasks) >= MAX_TASKS_PER_CONNECTION:
                            await ws.send_json({"type": "error", "data": {
                                "message": f"Max {MAX_TASKS_PER_CONNECTION} concurrent tasks",
                            }})
                            continue
                        task_id = str(uuid.uuid4())
                        task = asyncio.create_task(
                            self._execute_task(ws, task_id, payload)
                        )
                        active_tasks[task_id] = task

                    elif msg_type == "kill_task":
                        task_id = payload.get("task_id", "")
                        if task_id in active_tasks:
                            active_tasks[task_id].cancel()
                            del active_tasks[task_id]
                            await ws.send_json({"type": "task_error", "data": {
                                "task_id": task_id, "error": "Task cancelled",
                            }})

                    else:
                        await ws.send_json({"type": "error", "data": {
                            "message": f"Unknown type: {msg_type}",
                        }})

                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            for task in active_tasks.values():
                task.cancel()
            # Retain session metadata briefly so clients can resume after a blip
            self._sessions[session_id] = {
                "user": current_user,
                "last_seen": time.time(),
            }
            logger.info(
                "WebSocket session %s disconnected (%d task(s) cancelled)",
                session_id, len(active_tasks),
            )

        return ws

    async def _execute_task(
        self,
        ws: web.WebSocketResponse,
        task_id: str,
        payload: dict,
    ):
        """Execute a task and stream results back via WebSocket."""
        prompt = payload.get("prompt", "")
        mode = payload.get("mode", "standard")

        try:
            await ws.send_json({"type": "task_started", "data": {
                "task_id": task_id,
                "mode": mode,
            }})

            # Build agent using shared builder (same quality as CLI)
            from mullet.server.api import _build_server_agent

            opts = payload.get("options", {})
            opts["client"] = payload.get("project")
            opts["mode"] = mode
            # First-frame providers field (parity with REST /v1/chat/completions
            # `providers` body key). Accept CSV string or list; top-level wins
            # over options.providers when both are present so callers can
            # constrain a single task without crafting a nested options dict.
            _ws_providers = payload.get("providers")
            if _ws_providers is not None:
                opts["providers"] = _ws_providers

            async def send_token(content: str):
                """Send a token stream event to the WebSocket."""
                if not ws.closed:
                    await ws.send_json({"type": "token_stream", "data": {
                        "task_id": task_id, "content": content,
                    }})

            async def send_tool(name: str, preview: str):
                """Send a tool start event to the WebSocket."""
                if not ws.closed:
                    await ws.send_json({"type": "tool_start", "data": {
                        "task_id": task_id, "tool_name": name, "args_preview": preview,
                    }})

            # Sync callbacks wrapped for async sending
            token_queue: asyncio.Queue = asyncio.Queue()

            def on_token(delta: str):
                """Queue a token delta for async streaming."""
                token_queue.put_nowait(("token", delta))

            def on_tool(name: str, preview: str):
                """Queue a tool start event for async streaming."""
                token_queue.put_nowait(("tool", name, preview))

            def on_tool_result(name: str, preview: str):
                """Queue a tool result event for async streaming."""
                token_queue.put_nowait(("tool_result", name, preview))

            from mullet.core.env import DELEGATION_OFF_MESSAGE, delegation_enabled
            if not delegation_enabled():
                await ws.send_json({"type": "task_error", "data": {
                    "task_id": task_id, "error": DELEGATION_OFF_MESSAGE,
                }})
                return

            agent, provider = _build_server_agent(
                prompt, opts,
                on_token=on_token,
                on_tool_call=on_tool,
            )
            # Add tool_result callback (not part of _build_server_agent signature)
            if agent:
                agent.on_tool_result = on_tool_result
            if not agent:
                await ws.send_json({"type": "task_error", "data": {
                    "task_id": task_id, "error": "No provider available",
                }})
                return

            loop = asyncio.get_event_loop()
            agent_future = loop.run_in_executor(None, agent.run, prompt)

            # Stream events while agent runs
            while not agent_future.done():
                try:
                    event = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                    if event[0] == "token":
                        await send_token(event[1])
                    elif event[0] == "tool":
                        await send_tool(event[1], event[2])
                    elif event[0] == "tool_result" and not ws.closed:
                        await ws.send_json({"type": "tool_result", "data": {
                            "task_id": task_id, "tool_name": event[1], "preview": event[2],
                        }})
                except asyncio.TimeoutError:
                    continue

            # Drain remaining events
            while not token_queue.empty():
                event = token_queue.get_nowait()
                if event[0] == "token":
                    await send_token(event[1])
                elif event[0] == "tool_result" and not ws.closed:
                    await ws.send_json({"type": "tool_result", "data": {
                        "task_id": task_id, "tool_name": event[1], "preview": event[2],
                    }})

            result = agent_future.result()

            await ws.send_json({"type": "task_complete", "data": {
                "task_id": task_id,
                "result": result.final_response,
                "tokens": result.tokens_used.total_tokens,
                "tools_called": result.tool_calls_made,
            }})

        except asyncio.CancelledError:
            if not ws.closed:
                await ws.send_json({"type": "task_error", "data": {
                    "task_id": task_id, "error": "Task cancelled",
                }})
        except Exception as e:
            logger.exception("Task %s failed", task_id)
            if not ws.closed:
                await ws.send_json({"type": "task_error", "data": {
                    "task_id": task_id, "error": f"Task execution failed: {type(e).__name__}",
                }})
