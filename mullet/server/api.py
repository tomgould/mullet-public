"""REST API endpoints — OpenAI-compatible chat completions, health, models."""

import json
import logging
import os
import time
import uuid

from aiohttp import web

logger = logging.getLogger(__name__)

MAX_SESSIONS = 1000


def setup_api_routes(app: web.Application):
    """Register all REST API routes."""
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/api/v1/status", handle_status)
    app.router.add_get("/api/v1/info", handle_info)
    app.router.add_get("/api/v1/skills", handle_list_skills)
    app.router.add_get("/api/v1/tools", handle_list_tools)
    app.router.add_get("/api/v1/projects", handle_list_projects)
    app.router.add_post("/api/v1/search", handle_search)
    app.router.add_post("/api/v1/memory", handle_store_memory)
    app.router.add_get("/api/v1/memory", handle_search_memory)
    app.router.add_post("/api/v1/explain", handle_explain)
    app.router.add_get("/metrics", handle_metrics)


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — health check with provider status.

    Uses the non-blocking local-provider wrapper so the handler never waits
    on the LMStudio subprocess probe (Cycle 26). Cold cache → ``local=null``
    with a background refresh scheduled; warm cache → cached value returned
    immediately.
    """
    from mullet.core.providers.local_detection import get_local_provider_nonblocking
    local = await get_local_provider_nonblocking()
    return web.json_response({
        "status": "ok",
        "version": __import__("mullet").__version__,
        "timestamp": time.time(),
        "active_tasks": len(request.app.get("active_tasks", {})),
        "providers": {
            "local": {"available": local is not None, "model": local.model if local else None},
        },
    })


async def handle_models(request: web.Request) -> web.Response:
    """GET /v1/models — list available models (OpenAI-compatible format).

    Non-blocking local-provider lookup (Cycle 26): see :func:`handle_health`.
    """
    import os
    from mullet.core.providers.local_detection import get_local_provider_nonblocking

    models = [{"id": "mullet", "object": "model", "owned_by": "mullet"}]

    local = await get_local_provider_nonblocking()
    if local:
        models.append({"id": f"mullet/local/{local.model}", "object": "model", "owned_by": "local"})

    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append({"id": "mullet/anthropic/claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"})
        models.append({"id": "mullet/anthropic/claude-opus-4-8", "object": "model", "owned_by": "anthropic"})
        models.append({"id": "mullet/anthropic/claude-opus-4-7", "object": "model", "owned_by": "anthropic"})

    if os.environ.get("OPENROUTER_API_KEY"):
        models.append({"id": "mullet/openrouter/auto", "object": "model", "owned_by": "openrouter"})

    return web.json_response({"object": "list", "data": models})


async def handle_chat_completions(request: web.Request) -> web.Response:
    """POST /v1/chat/completions — OpenAI-compatible chat completions.

    Accepts standard OpenAI format. Routes through Mullet's agent engine.
    Supports both streaming (SSE) and non-streaming responses.
    """
    from mullet.server.metrics import get_metrics
    _metrics = get_metrics()
    _metrics.inc("mullet_requests_total", labels={"endpoint": "chat_completions"})
    _req_start = time.time()

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    from mullet.core.env import DELEGATION_OFF_MESSAGE, delegation_enabled
    if not delegation_enabled():
        return web.json_response({"error": DELEGATION_OFF_MESSAGE, "delegation": "off"}, status=503)

    messages = body.get("messages", [])
    stream = body.get("stream", False)
    mullet_opts = body.get("mullet", {})

    # Session continuity: X-Mullet-Session-Id header loads previous history
    session_id = request.headers.get("X-Mullet-Session-Id", "")
    if session_id:
        sessions = request.app.get("sessions", {})
        prev_messages = sessions.get(session_id, [])
        if prev_messages:
            messages = prev_messages + messages

    if not messages:
        return web.json_response({"error": "messages required"}, status=400)

    # Extract the last user message as the task
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        return web.json_response({"error": "No user message found"}, status=400)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

    if stream:
        return await _stream_response(request, completion_id, user_message, messages, mullet_opts)
    else:
        return await _sync_response(request, completion_id, user_message, messages, mullet_opts)


def _build_server_agent(user_message: str, opts: dict, on_token=None, on_tool_call=None):
    """Build a fully wired agent for API/WebSocket use.

    Uses the same agent builder logic as CLI to ensure consistent behavior.

    opts may include:
        providers: CSV string or list of provider names to constrain routing
            (e.g. "kimi,haiku"). Forwarded as allowed_providers to the router.
            Default None: no constraint.
    """
    from mullet.core.agent.runner import AgentRunner
    from mullet.cli.main import build_tool_registry, detect_local_provider
    from mullet.core.agent.prompts import build_base_prompt

    registry = build_tool_registry()

    # Parse optional provider allowlist from opts.
    allowed_providers: frozenset[str] | None = None
    _raw_allow = opts.get("providers") or opts.get("allowed_providers")
    if isinstance(_raw_allow, str):
        _parts = [p.strip() for p in _raw_allow.split(",") if p.strip()]
        allowed_providers = frozenset(_parts) if _parts else None
    elif isinstance(_raw_allow, (list, tuple, set, frozenset)):
        _parts = [str(p).strip() for p in _raw_allow if str(p).strip()]
        allowed_providers = frozenset(_parts) if _parts else None

    # Resolve provider via Router
    try:
        from mullet.core.router.resolver import build_default_router
        router, resolver = build_default_router()
        provider, decision = resolver.route_and_resolve(
            task=user_message,
            force_tier=opts.get("force_tier"),
            allowed_providers=allowed_providers,
        )
    except Exception:
        provider = detect_local_provider()

    if not provider:
        return None, None

    model_name = getattr(provider, "model", "")
    system_prompt = build_base_prompt(model=model_name)

    # Load knowledge if client specified
    client = opts.get("client")
    toolsets = ["terminal", "file", "web", "code", "browser", "delegation", "session", "git", "testing", "vision", "docker", "lsp"]
    if client:
        try:
            from mullet.core.knowledge.manager import build_knowledge_environment

            _km, _context, system_prompt = build_knowledge_environment(
                client, user_message, registry, system_prompt,
            )
            toolsets.append("trove")
        except Exception as _exc:
            logger.warning("api: knowledge manager init failed: %s", _exc)

    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=toolsets,
        system_prompt=system_prompt,
        max_iterations=int(opts.get("max_iterations", 90)),
        on_token=on_token,
        on_tool_call=on_tool_call,
        allowed_providers=allowed_providers,
    )
    return agent, provider


async def _sync_response(
    request: web.Request,
    completion_id: str,
    user_message: str,
    messages: list[dict],
    opts: dict,
) -> web.Response:
    """Non-streaming response — run agent and return complete result."""
    import asyncio

    agent, provider = _build_server_agent(user_message, opts)
    if not agent:
        return web.json_response({"error": "No provider available"}, status=503)

    mode = opts.get("mode", "standard")

    # Run in orchestration mode if requested
    def _execute():
        if mode == "autopilot":
            from mullet.core.orchestration.autopilot import AutopilotMode
            ap = AutopilotMode()
            r = ap.execute(user_message, agent)
            return r.summary, r.completed, r.total_tokens
        elif mode == "ralph":
            from mullet.core.orchestration.ralph import RalphMode
            rm = RalphMode()
            r = rm.execute(user_message, agent)
            return r.summary, r.completed, r.total_tokens
        elif mode == "ultrawork":
            from mullet.core.orchestration.ultrawork import UltraworkMode
            um = UltraworkMode()
            r = um.execute(user_message, agent, provider)
            return r.final_response, r.completed, r.tokens_used
        else:
            # Pass prior conversation as history for multi-turn support
            history = [m for m in messages if m.get("role") != "user" or m.get("content") != user_message]
            # Keep only system + prior turns (not the current user message)
            prior = [m for m in history if m != messages[-1]] if len(messages) > 1 else None
            r = agent.run(user_message, conversation_history=prior if prior else None)
            return r.final_response, r.completed, r.tokens_used

    try:
        response_text, completed, tokens = await asyncio.get_event_loop().run_in_executor(None, _execute)
    except Exception as e:
        logger.error("Agent execution failed: %s", e)
        return web.json_response({
            "error": {"message": f"Agent execution failed: {e}", "type": "server_error"},
        }, status=500)

    # Save session history for continuation (X-Mullet-Session-Id)
    session_id = request.headers.get("X-Mullet-Session-Id", "")
    if session_id:
        if "sessions" not in request.app:
            request.app["sessions"] = {}
        sessions = request.app["sessions"]
        # Evict oldest sessions if at capacity
        if session_id not in sessions and len(sessions) >= MAX_SESSIONS:
            oldest_key = next(iter(sessions))
            del sessions[oldest_key]
        # Append the user message and assistant response
        session_messages = sessions.get(session_id, [])
        session_messages.extend(messages)  # Already includes previous + new
        session_messages.append({"role": "assistant", "content": response_text})
        sessions[session_id] = session_messages[-50:]  # Keep last 50 messages
    else:
        # Generate a session ID for the client to use in subsequent requests
        session_id = uuid.uuid4().hex[:16]

    return web.json_response({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"mullet/{provider.name}/{provider.model}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop" if completed else "length",
        }],
        "usage": {
            "prompt_tokens": getattr(tokens, "input_tokens", 0),
            "completion_tokens": getattr(tokens, "output_tokens", 0),
            "total_tokens": getattr(tokens, "total_tokens", 0),
        },
        "mullet": {
            "mode": mode,
            "provider_used": provider.name,
            "session_id": session_id,
        },
    })


async def _stream_response(
    request: web.Request,
    completion_id: str,
    user_message: str,
    messages: list[dict],
    opts: dict,
) -> web.StreamResponse:
    """SSE streaming response using asyncio.Queue (non-blocking)."""
    import asyncio

    stream_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_token(delta: str):
        """Forward an LLM token delta to the SSE stream queue."""
        loop.call_soon_threadsafe(stream_q.put_nowait, delta)

    def on_tool(name: str, preview: str):
        """Forward a tool-call notification to the SSE stream queue."""
        loop.call_soon_threadsafe(stream_q.put_nowait, f"\n`{name}`\n")

    agent, provider = _build_server_agent(user_message, opts, on_token=on_token, on_tool_call=on_tool)
    if not agent:
        return web.json_response({"error": "No provider available"}, status=503)

    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    # Run agent in background thread
    agent_future = loop.run_in_executor(None, agent.run, user_message)

    while not agent_future.done() or not stream_q.empty():
        try:
            delta = await asyncio.wait_for(stream_q.get(), timeout=0.1)
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": delta}, "index": 0}],
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        except asyncio.TimeoutError:
            continue

    await response.write(b"data: [DONE]\n\n")
    return response


async def handle_status(request: web.Request) -> web.Response:
    """GET /api/v1/status — current tasks, providers, modes, budget.

    Non-blocking local-provider lookup (Cycle 26): see :func:`handle_health`.
    """
    import os
    from mullet.core.providers.local_detection import get_local_provider_nonblocking
    local = await get_local_provider_nonblocking()

    # Check for active orchestration modes
    active_modes = []
    from pathlib import Path
    state_dir = Path(".mullet") / "state"
    if state_dir.exists():
        for f in state_dir.glob("*.json"):
            active_modes.append(f.stem)

    return web.json_response({
        "status": "ok",
        "version": __import__("mullet").__version__,
        "active_tasks": len(request.app.get("active_tasks", {})),
        "local_provider": {
            "name": local.name if local else None,
            "model": getattr(local, "model", None) if local else None,
        },
        "providers": {
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
            "local": local is not None,
        },
        "active_modes": active_modes,
    })


async def handle_info(request: web.Request) -> web.Response:
    """GET /api/v1/info — comprehensive system information.

    Non-blocking local-provider lookup (Cycle 26): see :func:`handle_health`.
    """
    import platform
    import mullet
    from mullet.cli.main import build_tool_registry
    from mullet.core.providers.local_detection import get_local_provider_nonblocking
    from mullet.config.mode import MulletMode

    local = await get_local_provider_nonblocking()
    registry = build_tool_registry()

    info = {
        "version": mullet.__version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "mode": MulletMode.current(),
        "providers": {
            "local": {"available": local is not None, "model": local.model if local else None},
            "anthropic": {"available": bool(os.environ.get("ANTHROPIC_API_KEY"))},
            "openrouter": {"available": bool(os.environ.get("OPENROUTER_API_KEY"))},
        },
        "tools": sorted(registry.list_tools()),
        "toolsets": sorted(registry.list_toolsets()),
        "profiles": sorted(registry.list_profiles()),
        "tool_count": len(registry.list_tools()),
    }

    try:
        from mullet.core.skills.loader import SkillLoader
        from mullet.core.skills.registry import SkillRegistry
        loader = SkillLoader()
        sr = SkillRegistry(loader=loader)
        sr.load_all()
        info["skills"] = [s["name"] for s in sr.list_skills()]
        info["skill_count"] = len(info["skills"])
    except Exception:
        info["skills"] = []
        info["skill_count"] = 0

    return web.json_response(info)


async def handle_list_projects(request: web.Request) -> web.Response:
    """GET /api/v1/projects — list projects user has access to."""
    user = request.get("user", {})
    db = request.app.get("db")
    if not db:
        return web.json_response({"projects": []})

    if user.get("role") == "admin":
        rows = db.execute("SELECT id, name, pool_name, description FROM projects").fetchall()
    elif user.get("id"):
        rows = db.execute(
            "SELECT p.id, p.name, p.pool_name, p.description FROM user_projects up "
            "JOIN projects p ON up.project_id = p.id WHERE up.user_id = ?",
            (user["id"],),
        ).fetchall()
    else:
        return web.json_response({"projects": [], "error": "Authentication required"}, status=401)

    projects = [{"id": r[0], "name": r[1], "pool": r[2], "description": r[3]} for r in rows]
    return web.json_response({"projects": projects})


async def handle_search(request: web.Request) -> web.Response:
    """POST /api/v1/search — search project knowledge."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    client = body.get("client") or body.get("project")
    query = body.get("query", "")

    if not client or not query:
        return web.json_response({"error": "client and query required"}, status=400)

    try:
        from mullet.core.knowledge.manager import knowledge_session
        with knowledge_session(client) as km:
            results = km.search(query, top_k=body.get("top_k", 10))
        return web.json_response({"results": results})
    except Exception as e:
        logger.warning("Search failed: %s", e)
        return web.json_response({"error": "Search operation failed"}, status=500)


async def handle_store_memory(request: web.Request) -> web.Response:
    """POST /api/v1/memory — store a memory."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    client = body.get("client")
    key = body.get("key", "")
    content = body.get("content", "")

    if not client or not key or not content:
        return web.json_response({"error": "client, key, and content required"}, status=400)

    try:
        from mullet.core.knowledge.manager import knowledge_session
        with knowledge_session(client) as km:
            km.remember(key, content, namespace=body.get("namespace", "api"))
        return web.json_response({"stored": True, "key": key})
    except Exception as e:
        logger.warning("Memory store failed: %s", e)
        return web.json_response({"error": "Memory store failed"}, status=500)


async def handle_search_memory(request: web.Request) -> web.Response:
    """GET /api/v1/memory — search memories."""
    client = request.query.get("client", "")
    query = request.query.get("query", "")

    if not client:
        return web.json_response({"error": "client query param required"}, status=400)

    try:
        from mullet.core.knowledge.manager import knowledge_session
        with knowledge_session(client) as km:
            results = km.recall(query or "recent", namespace=request.query.get("namespace"))
        return web.json_response({"memories": results})
    except Exception as e:
        logger.warning("Memory search failed: %s", e)
        return web.json_response({"error": "Memory search failed"}, status=500)


async def handle_list_skills(request: web.Request) -> web.Response:
    """GET /api/v1/skills — list available skills."""
    try:
        from mullet.core.skills.loader import SkillLoader
        from mullet.core.skills.registry import SkillRegistry
        loader = SkillLoader()
        registry = SkillRegistry(loader=loader)
        registry.load_all()
        skills = registry.list_skills()
        return web.json_response({"skills": skills})
    except Exception as e:
        logger.warning("Skills listing failed: %s", e)
        return web.json_response({"skills": [], "error": "Failed to load skills"})


async def handle_list_tools(request: web.Request) -> web.Response:
    """GET /api/v1/tools — list available tools and toolsets."""
    from mullet.cli.main import build_tool_registry
    registry = build_tool_registry()
    return web.json_response({
        "tools": sorted(registry.list_tools()),
        "toolsets": sorted(registry.list_toolsets()),
        "profiles": sorted(registry.list_profiles()),
    })


async def handle_explain(request: web.Request) -> web.Response:
    """POST /api/v1/explain — analyze task complexity and routing decision."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    task = body.get("task", "")
    if not task:
        return web.json_response({"error": "task required"}, status=400)

    from mullet.core.router.complexity import (
        extract_signals, score_complexity, score_to_tier, is_simple_message,
    )

    is_simple = is_simple_message(task)
    signals = extract_signals(task)
    score = score_complexity(signals)
    tier = score_to_tier(score)

    # Check routing learning
    learned_suggestion = None
    try:
        from mullet.core.router.learning import suggest_tier
        learned_suggestion = suggest_tier(task)
    except Exception as _exc:
        logger.warning("api: learned tier suggestion failed: %s", _exc)

    # Check project detection
    project = None
    try:
        from mullet.core.agent.project_detect import detect_project
        info = detect_project()
        if info.language != "unknown":
            project = {"language": info.language, "framework": info.framework, "test_command": info.test_command}
    except Exception as _exc:
        logger.warning("api: project detection failed: %s", _exc)

    return web.json_response({
        "task": task[:200],
        "is_simple": is_simple,
        "complexity_score": round(score, 1),
        "tier": tier.value,
        "signals": {
            "lexical": {
                "word_count": signals.lexical.word_count,
                "architecture_keywords": signals.lexical.has_architecture_keywords,
                "risk_keywords": signals.lexical.has_risk_keywords,
                "debugging_keywords": signals.lexical.has_debugging_keywords,
                "simple_keywords": signals.lexical.has_simple_keywords,
            },
            "structural": {
                "subtasks": signals.structural.estimated_subtasks,
                "domain": signals.structural.domain_specificity,
                "reversibility": signals.structural.reversibility,
                "impact_scope": signals.structural.impact_scope,
            },
        },
        "learned_suggestion": learned_suggestion,
        "project": project,
    })


async def handle_metrics(request: web.Request) -> web.Response:
    """GET /metrics — Prometheus-compatible metrics endpoint."""
    from mullet.server.metrics import get_metrics
    metrics = get_metrics()

    # Add current state gauges
    active = len(request.app.get("active_tasks", {}))
    metrics.set_gauge("mullet_active_tasks", active)

    return web.Response(
        text=metrics.format_prometheus(),
        content_type="text/plain",
    )
