"""Tests for mullet/server/api.py — endpoint handlers.

Covers: handle_health, handle_models, handle_status, handle_info,
        handle_list_skills, handle_list_tools, handle_list_projects,
        handle_search, handle_store_memory, handle_search_memory,
        handle_explain, handle_metrics, setup_api_routes.

Skips: handle_chat_completions (too complex — covered separately).
"""

import json
import os
import pytest
from aiohttp import web
from unittest.mock import MagicMock, patch, AsyncMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_local_mock(model="qwen-2.5"):
    m = MagicMock()
    m.model = model
    m.name = "local"
    return m


async def _build_client(aiohttp_client, handler, path="/test", method="GET"):
    """Convenience: wire a single handler into a minimal app and return client."""
    app = web.Application()
    app["active_tasks"] = {}
    if method == "GET":
        app.router.add_get(path, handler)
    else:
        app.router.add_post(path, handler)
    return await aiohttp_client(app)


# ---------------------------------------------------------------------------
# handle_health
# ---------------------------------------------------------------------------

async def test_handle_health_with_local(aiohttp_client):
    """Local provider available → providers.local populated in response."""
    from mullet.server.api import handle_health
    mock_local = _make_local_mock("qwen-2.5")

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=mock_local),
    ):
        client = await _build_client(aiohttp_client, handle_health, "/health")
        resp = await client.get("/health")
        data = await resp.json()

    assert resp.status == 200
    assert data["status"] == "ok"
    assert data["providers"]["local"]["available"] is True
    assert data["providers"]["local"]["model"] == "qwen-2.5"
    assert "version" in data
    assert "timestamp" in data


async def test_handle_health_no_local(aiohttp_client):
    """No local provider → available=False, model=None."""
    from mullet.server.api import handle_health

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=None),
    ):
        client = await _build_client(aiohttp_client, handle_health, "/health")
        resp = await client.get("/health")
        data = await resp.json()

    assert resp.status == 200
    assert data["providers"]["local"]["available"] is False
    assert data["providers"]["local"]["model"] is None


async def test_handle_health_reports_active_tasks(aiohttp_client):
    """active_tasks count reflects app state."""
    from mullet.server.api import handle_health

    app = web.Application()
    app["active_tasks"] = {"task1": {}, "task2": {}}
    app.router.add_get("/health", handle_health)

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=None),
    ):
        client = await aiohttp_client(app)
        resp = await client.get("/health")
        data = await resp.json()

    assert data["active_tasks"] == 2


# ---------------------------------------------------------------------------
# handle_models
# ---------------------------------------------------------------------------

async def test_handle_models_base_only(aiohttp_client):
    """No env keys, no local → only the base 'mullet' model returned."""
    from mullet.server.api import handle_models

    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")}
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENROUTER_API_KEY", None)

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=None),
    ), patch.dict(os.environ, {}, clear=True):
        client = await _build_client(aiohttp_client, handle_models, "/v1/models")
        resp = await client.get("/v1/models")
        data = await resp.json()

    assert resp.status == 200
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "mullet" in ids
    assert not any("anthropic" in i for i in ids)
    assert not any("openrouter" in i for i in ids)
    assert not any("local" in i for i in ids)


async def test_handle_models_with_local(aiohttp_client):
    """Local provider available → local model in list."""
    from mullet.server.api import handle_models
    mock_local = _make_local_mock("tinyllama")

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=mock_local),
    ), patch.dict(os.environ, {}, clear=True):
        client = await _build_client(aiohttp_client, handle_models, "/v1/models")
        resp = await client.get("/v1/models")
        data = await resp.json()

    ids = [m["id"] for m in data["data"]]
    assert any("local/tinyllama" in i for i in ids)


async def test_handle_models_with_anthropic_key(aiohttp_client):
    """ANTHROPIC_API_KEY set → claude models included."""
    from mullet.server.api import handle_models

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=None),
    ), patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-abc"}, clear=True):
        client = await _build_client(aiohttp_client, handle_models, "/v1/models")
        resp = await client.get("/v1/models")
        data = await resp.json()

    ids = [m["id"] for m in data["data"]]
    assert any("anthropic" in i for i in ids)


async def test_handle_models_with_openrouter_key(aiohttp_client):
    """OPENROUTER_API_KEY set → openrouter model included."""
    from mullet.server.api import handle_models

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=None),
    ), patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
        client = await _build_client(aiohttp_client, handle_models, "/v1/models")
        resp = await client.get("/v1/models")
        data = await resp.json()

    ids = [m["id"] for m in data["data"]]
    assert any("openrouter" in i for i in ids)


# ---------------------------------------------------------------------------
# handle_status
# ---------------------------------------------------------------------------

async def test_handle_status_basic(aiohttp_client):
    """handle_status returns status, version, providers dict."""
    from mullet.server.api import handle_status

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=None),
    ), patch.dict(os.environ, {}, clear=True):
        client = await _build_client(aiohttp_client, handle_status, "/api/v1/status")
        resp = await client.get("/api/v1/status")
        data = await resp.json()

    assert resp.status == 200
    assert data["status"] == "ok"
    assert "version" in data
    assert "providers" in data
    assert "local" in data["providers"]
    assert "anthropic" in data["providers"]
    assert "openrouter" in data["providers"]


async def test_handle_status_with_local_provider(aiohttp_client):
    """Local provider → local_provider and providers.local both reflect it."""
    from mullet.server.api import handle_status
    mock_local = _make_local_mock("phi-3")

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=mock_local),
    ), patch.dict(os.environ, {}, clear=True):
        client = await _build_client(aiohttp_client, handle_status, "/api/v1/status")
        resp = await client.get("/api/v1/status")
        data = await resp.json()

    assert data["providers"]["local"] is True
    assert data["local_provider"]["model"] == "phi-3"


async def test_handle_status_active_tasks(aiohttp_client):
    """active_tasks count is surfaced in response."""
    from mullet.server.api import handle_status

    app = web.Application()
    app["active_tasks"] = {"a": 1, "b": 2, "c": 3}
    app.router.add_get("/api/v1/status", handle_status)

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=None),
    ), patch.dict(os.environ, {}, clear=True):
        client = await aiohttp_client(app)
        resp = await client.get("/api/v1/status")
        data = await resp.json()

    assert data["active_tasks"] == 3


# ---------------------------------------------------------------------------
# handle_info
# ---------------------------------------------------------------------------

async def test_handle_info_basic(aiohttp_client):
    """handle_info returns version, python, platform, tools, toolsets."""
    from mullet.server.api import handle_info

    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = ["tool_a", "tool_b"]
    mock_registry.list_toolsets.return_value = ["ts1"]
    mock_registry.list_profiles.return_value = ["default"]

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=None),
    ), patch(
        "mullet.cli.main.build_tool_registry",
        return_value=mock_registry,
    ), patch(
        "mullet.config.mode.MulletMode.current",
        return_value="solo",
    ), patch.dict(os.environ, {}, clear=True):
        client = await _build_client(aiohttp_client, handle_info, "/api/v1/info")
        resp = await client.get("/api/v1/info")
        data = await resp.json()

    assert resp.status == 200
    assert "version" in data
    assert "python" in data
    assert "platform" in data
    assert "tools" in data
    assert "tool_count" in data


async def test_handle_info_skills_fallback_on_error(aiohttp_client):
    """If skills loading raises, info still returns with skills=[]."""
    from mullet.server.api import handle_info

    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = []
    mock_registry.list_toolsets.return_value = []
    mock_registry.list_profiles.return_value = []

    with patch(
        "mullet.core.providers.local_detection.get_local_provider_nonblocking",
        new=AsyncMock(return_value=None),
    ), patch(
        "mullet.cli.main.build_tool_registry",
        return_value=mock_registry,
    ), patch(
        "mullet.config.mode.MulletMode.current",
        return_value="solo",
    ), patch(
        "mullet.core.skills.loader.SkillLoader",
        side_effect=RuntimeError("skill load boom"),
    ), patch.dict(os.environ, {}, clear=True):
        client = await _build_client(aiohttp_client, handle_info, "/api/v1/info")
        resp = await client.get("/api/v1/info")
        data = await resp.json()

    assert resp.status == 200
    assert data["skills"] == []
    assert data["skill_count"] == 0


# ---------------------------------------------------------------------------
# handle_list_skills
# ---------------------------------------------------------------------------

async def test_handle_list_skills_success(aiohttp_client):
    """handle_list_skills returns list of skills from registry."""
    from mullet.server.api import handle_list_skills

    fake_skills = [{"name": "autopilot"}, {"name": "brainstorm"}]
    mock_registry = MagicMock()
    mock_registry.list_skills.return_value = fake_skills

    with patch("mullet.core.skills.registry.SkillRegistry", return_value=mock_registry), \
         patch("mullet.core.skills.loader.SkillLoader", return_value=MagicMock()):
        client = await _build_client(aiohttp_client, handle_list_skills, "/api/v1/skills")
        resp = await client.get("/api/v1/skills")
        data = await resp.json()

    assert resp.status == 200
    assert "skills" in data


async def test_handle_list_skills_error_returns_empty(aiohttp_client):
    """If registry raises, returns skills=[] with an error message (not 500)."""
    from mullet.server.api import handle_list_skills

    with patch(
        "mullet.core.skills.loader.SkillLoader",
        side_effect=Exception("boom"),
    ):
        client = await _build_client(aiohttp_client, handle_list_skills, "/api/v1/skills")
        resp = await client.get("/api/v1/skills")
        data = await resp.json()

    # Handler catches exception and returns 200 with empty skills
    assert resp.status == 200
    assert data["skills"] == []
    assert "error" in data


# ---------------------------------------------------------------------------
# handle_list_tools
# ---------------------------------------------------------------------------

async def test_handle_list_tools_returns_tools_and_toolsets(aiohttp_client):
    """handle_list_tools returns tools, toolsets, profiles keys."""
    from mullet.server.api import handle_list_tools

    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = ["terminal", "file_ops"]
    mock_registry.list_toolsets.return_value = ["core"]
    mock_registry.list_profiles.return_value = ["default"]

    with patch("mullet.cli.main.build_tool_registry", return_value=mock_registry):
        client = await _build_client(aiohttp_client, handle_list_tools, "/api/v1/tools")
        resp = await client.get("/api/v1/tools")
        data = await resp.json()

    assert resp.status == 200
    assert "tools" in data
    assert "toolsets" in data
    assert "profiles" in data
    assert isinstance(data["tools"], list)


# ---------------------------------------------------------------------------
# handle_list_projects
# ---------------------------------------------------------------------------

async def test_handle_list_projects_no_db(aiohttp_client):
    """No db in app → returns empty projects list."""
    from mullet.server.api import handle_list_projects

    app = web.Application()
    app["active_tasks"] = {}
    # No "db" key
    app.router.add_get("/api/v1/projects", handle_list_projects)
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/projects")
    data = await resp.json()

    assert resp.status == 200
    assert data["projects"] == []


async def test_handle_list_projects_no_auth(aiohttp_client):
    """With db but no user → 401 authentication required."""
    from mullet.server.api import handle_list_projects

    mock_db = MagicMock()

    app = web.Application()
    app["active_tasks"] = {}
    app["db"] = mock_db
    app.router.add_get("/api/v1/projects", handle_list_projects)
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/projects")
    data = await resp.json()

    assert resp.status == 401
    assert "error" in data


async def test_handle_list_projects_admin_sees_all(aiohttp_client):
    """Admin user → all projects via SELECT * query."""
    from mullet.server.api import handle_list_projects

    fake_rows = [(1, "Alpha", "pool_alpha", "Desc A"), (2, "Beta", "pool_beta", "Desc B")]
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = fake_rows
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_cursor

    app = web.Application()
    app["active_tasks"] = {}
    app["db"] = mock_db

    async def _inject_user(request):
        request["user"] = {"role": "admin", "id": 1}
        return await handle_list_projects(request)

    app.router.add_get("/api/v1/projects", _inject_user)
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/projects")
    data = await resp.json()

    assert resp.status == 200
    assert len(data["projects"]) == 2
    assert data["projects"][0]["name"] == "Alpha"


async def test_handle_list_projects_regular_user(aiohttp_client):
    """Non-admin user with id → JOIN query, sees own projects."""
    from mullet.server.api import handle_list_projects

    fake_rows = [(3, "MyProject", "pool_mine", "My desc")]
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = fake_rows
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_cursor

    app = web.Application()
    app["active_tasks"] = {}
    app["db"] = mock_db

    async def _inject_user(request):
        request["user"] = {"role": "user", "id": 42}
        return await handle_list_projects(request)

    app.router.add_get("/api/v1/projects", _inject_user)
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/projects")
    data = await resp.json()

    assert resp.status == 200
    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "MyProject"


# ---------------------------------------------------------------------------
# handle_search
# ---------------------------------------------------------------------------

async def test_handle_search_missing_client_and_query(aiohttp_client):
    """Empty body → 400 with error message."""
    from mullet.server.api import handle_search

    client = await _build_client(aiohttp_client, handle_search, "/api/v1/search", "POST")
    resp = await client.post("/api/v1/search", json={})
    data = await resp.json()

    assert resp.status == 400
    assert "error" in data


async def test_handle_search_missing_query(aiohttp_client):
    """client present but no query → 400."""
    from mullet.server.api import handle_search

    client = await _build_client(aiohttp_client, handle_search, "/api/v1/search", "POST")
    resp = await client.post("/api/v1/search", json={"client": "myproject"})
    data = await resp.json()

    assert resp.status == 400


async def test_handle_search_invalid_json(aiohttp_client):
    """Non-JSON body → 400."""
    from mullet.server.api import handle_search

    client = await _build_client(aiohttp_client, handle_search, "/api/v1/search", "POST")
    resp = await client.post("/api/v1/search", data="not-json", headers={"Content-Type": "application/json"})

    assert resp.status == 400


async def test_handle_search_knowledge_session_raises(aiohttp_client):
    """knowledge_session raises → 500 error response."""
    from mullet.server.api import handle_search

    with patch(
        "mullet.core.knowledge.manager.knowledge_session",
        side_effect=RuntimeError("db gone"),
    ):
        client = await _build_client(aiohttp_client, handle_search, "/api/v1/search", "POST")
        resp = await client.post("/api/v1/search", json={"client": "proj", "query": "find me"})
        data = await resp.json()

    assert resp.status == 500
    assert "error" in data


async def test_handle_search_success(aiohttp_client):
    """Successful search → results list returned."""
    from mullet.server.api import handle_search

    fake_results = [{"id": "doc1", "score": 0.95, "content": "hello"}]

    mock_km = MagicMock()
    mock_km.search.return_value = fake_results
    mock_km.__enter__ = MagicMock(return_value=mock_km)
    mock_km.__exit__ = MagicMock(return_value=False)

    with patch("mullet.core.knowledge.manager.knowledge_session", return_value=mock_km):
        client = await _build_client(aiohttp_client, handle_search, "/api/v1/search", "POST")
        resp = await client.post("/api/v1/search", json={"client": "proj", "query": "find"})
        data = await resp.json()

    assert resp.status == 200
    assert data["results"] == fake_results


# ---------------------------------------------------------------------------
# handle_store_memory
# ---------------------------------------------------------------------------

async def test_handle_store_memory_missing_required_fields(aiohttp_client):
    """Missing client/key/content → 400."""
    from mullet.server.api import handle_store_memory

    client = await _build_client(aiohttp_client, handle_store_memory, "/api/v1/memory", "POST")

    # No fields at all
    resp = await client.post("/api/v1/memory", json={})
    assert resp.status == 400

    # Missing content
    resp = await client.post("/api/v1/memory", json={"client": "p", "key": "k"})
    assert resp.status == 400

    # Missing key
    resp = await client.post("/api/v1/memory", json={"client": "p", "content": "c"})
    assert resp.status == 400


async def test_handle_store_memory_invalid_json(aiohttp_client):
    """Non-JSON body → 400."""
    from mullet.server.api import handle_store_memory

    client = await _build_client(aiohttp_client, handle_store_memory, "/api/v1/memory", "POST")
    resp = await client.post("/api/v1/memory", data="bad", headers={"Content-Type": "application/json"})
    assert resp.status == 400


async def test_handle_store_memory_success(aiohttp_client):
    """Valid body → stored=True with the key echoed."""
    from mullet.server.api import handle_store_memory

    mock_km = MagicMock()
    mock_km.__enter__ = MagicMock(return_value=mock_km)
    mock_km.__exit__ = MagicMock(return_value=False)

    with patch("mullet.core.knowledge.manager.knowledge_session", return_value=mock_km):
        client = await _build_client(aiohttp_client, handle_store_memory, "/api/v1/memory", "POST")
        resp = await client.post(
            "/api/v1/memory",
            json={"client": "proj", "key": "insight-1", "content": "some fact"},
        )
        data = await resp.json()

    assert resp.status == 200
    assert data["stored"] is True
    assert data["key"] == "insight-1"


async def test_handle_store_memory_exception_returns_500(aiohttp_client):
    """knowledge_session raises → 500."""
    from mullet.server.api import handle_store_memory

    with patch(
        "mullet.core.knowledge.manager.knowledge_session",
        side_effect=Exception("disk full"),
    ):
        client = await _build_client(aiohttp_client, handle_store_memory, "/api/v1/memory", "POST")
        resp = await client.post(
            "/api/v1/memory",
            json={"client": "p", "key": "k", "content": "v"},
        )
        data = await resp.json()

    assert resp.status == 500
    assert "error" in data


# ---------------------------------------------------------------------------
# handle_search_memory
# ---------------------------------------------------------------------------

async def test_handle_search_memory_missing_client(aiohttp_client):
    """No client query param → 400."""
    from mullet.server.api import handle_search_memory

    client = await _build_client(aiohttp_client, handle_search_memory, "/api/v1/memory")
    resp = await client.get("/api/v1/memory")
    data = await resp.json()

    assert resp.status == 400
    assert "error" in data


async def test_handle_search_memory_success(aiohttp_client):
    """Valid client param → memories list returned."""
    from mullet.server.api import handle_search_memory

    fake_memories = [{"key": "fact-1", "content": "the answer is 42"}]
    mock_km = MagicMock()
    mock_km.recall.return_value = fake_memories
    mock_km.__enter__ = MagicMock(return_value=mock_km)
    mock_km.__exit__ = MagicMock(return_value=False)

    with patch("mullet.core.knowledge.manager.knowledge_session", return_value=mock_km):
        client = await _build_client(aiohttp_client, handle_search_memory, "/api/v1/memory")
        resp = await client.get("/api/v1/memory", params={"client": "proj", "query": "answer"})
        data = await resp.json()

    assert resp.status == 200
    assert data["memories"] == fake_memories


async def test_handle_search_memory_exception_returns_500(aiohttp_client):
    """knowledge_session raises → 500."""
    from mullet.server.api import handle_search_memory

    with patch(
        "mullet.core.knowledge.manager.knowledge_session",
        side_effect=Exception("index broken"),
    ):
        client = await _build_client(aiohttp_client, handle_search_memory, "/api/v1/memory")
        resp = await client.get("/api/v1/memory", params={"client": "proj"})
        data = await resp.json()

    assert resp.status == 500
    assert "error" in data


async def test_handle_search_memory_no_query_uses_recent(aiohttp_client):
    """client present but no query → calls km.recall('recent')."""
    from mullet.server.api import handle_search_memory

    mock_km = MagicMock()
    mock_km.recall.return_value = []
    mock_km.__enter__ = MagicMock(return_value=mock_km)
    mock_km.__exit__ = MagicMock(return_value=False)

    with patch("mullet.core.knowledge.manager.knowledge_session", return_value=mock_km):
        client = await _build_client(aiohttp_client, handle_search_memory, "/api/v1/memory")
        resp = await client.get("/api/v1/memory", params={"client": "proj"})

    assert resp.status == 200
    # Verify recall was called with "recent" as the fallback
    mock_km.recall.assert_called_once_with("recent", namespace=None)


# ---------------------------------------------------------------------------
# handle_explain
# ---------------------------------------------------------------------------

async def test_handle_explain_missing_task(aiohttp_client):
    """No task field → 400."""
    from mullet.server.api import handle_explain

    client = await _build_client(aiohttp_client, handle_explain, "/api/v1/explain", "POST")
    resp = await client.post("/api/v1/explain", json={})
    data = await resp.json()

    assert resp.status == 400
    assert "error" in data


async def test_handle_explain_invalid_json(aiohttp_client):
    """Non-JSON body → 400."""
    from mullet.server.api import handle_explain

    client = await _build_client(aiohttp_client, handle_explain, "/api/v1/explain", "POST")
    resp = await client.post("/api/v1/explain", data="bad", headers={"Content-Type": "application/json"})
    assert resp.status == 400


async def test_handle_explain_success(aiohttp_client):
    """Valid task → routing analysis in response."""
    from mullet.server.api import handle_explain

    client = await _build_client(aiohttp_client, handle_explain, "/api/v1/explain", "POST")
    resp = await client.post("/api/v1/explain", json={"task": "refactor the authentication module"})
    data = await resp.json()

    assert resp.status == 200
    assert "task" in data
    assert "is_simple" in data
    assert "complexity_score" in data
    assert "tier" in data
    assert "signals" in data
    assert "lexical" in data["signals"]
    assert "structural" in data["signals"]


async def test_handle_explain_task_truncated_to_200(aiohttp_client):
    """Task longer than 200 chars is truncated in the response."""
    from mullet.server.api import handle_explain

    long_task = "x" * 300
    client = await _build_client(aiohttp_client, handle_explain, "/api/v1/explain", "POST")
    resp = await client.post("/api/v1/explain", json={"task": long_task})
    data = await resp.json()

    assert resp.status == 200
    assert len(data["task"]) == 200


async def test_handle_explain_learned_suggestion_error_graceful(aiohttp_client):
    """Learned suggestion failure is silent — response still succeeds."""
    from mullet.server.api import handle_explain

    with patch(
        "mullet.core.router.learning.suggest_tier",
        side_effect=Exception("learning db gone"),
    ):
        client = await _build_client(aiohttp_client, handle_explain, "/api/v1/explain", "POST")
        resp = await client.post("/api/v1/explain", json={"task": "write tests"})
        data = await resp.json()

    assert resp.status == 200
    assert data.get("learned_suggestion") is None


# ---------------------------------------------------------------------------
# handle_metrics
# ---------------------------------------------------------------------------

async def test_handle_metrics_returns_text(aiohttp_client):
    """handle_metrics returns plain text prometheus-like output."""
    from mullet.server.api import handle_metrics

    client = await _build_client(aiohttp_client, handle_metrics, "/metrics")
    resp = await client.get("/metrics")

    assert resp.status == 200
    assert "text/plain" in resp.content_type
    text = await resp.text()
    assert "mullet" in text


async def test_handle_metrics_includes_active_tasks_gauge(aiohttp_client):
    """Gauge for active tasks is set from app state."""
    from mullet.server.api import handle_metrics

    app = web.Application()
    app["active_tasks"] = {"t1": {}, "t2": {}}
    app.router.add_get("/metrics", handle_metrics)
    client = await aiohttp_client(app)

    resp = await client.get("/metrics")
    text = await resp.text()

    assert "mullet_active_tasks" in text


async def test_handle_metrics_uptime_present(aiohttp_client):
    """Metrics output contains uptime_seconds line."""
    from mullet.server.api import handle_metrics

    client = await _build_client(aiohttp_client, handle_metrics, "/metrics")
    resp = await client.get("/metrics")
    text = await resp.text()

    assert "mullet_uptime_seconds" in text


# ---------------------------------------------------------------------------
# setup_api_routes
# ---------------------------------------------------------------------------

def test_setup_api_routes_registers_expected_paths():
    """setup_api_routes registers all documented endpoint paths."""
    from mullet.server.api import setup_api_routes

    app = web.Application()
    setup_api_routes(app)

    resource_paths = {r.canonical for r in app.router.resources()}
    assert "/health" in resource_paths
    assert "/v1/models" in resource_paths
    assert "/v1/chat/completions" in resource_paths
    assert "/api/v1/status" in resource_paths
    assert "/api/v1/info" in resource_paths
    assert "/api/v1/skills" in resource_paths
    assert "/api/v1/tools" in resource_paths
    assert "/api/v1/projects" in resource_paths
    assert "/api/v1/search" in resource_paths
    assert "/api/v1/memory" in resource_paths
    assert "/api/v1/explain" in resource_paths
    assert "/metrics" in resource_paths
