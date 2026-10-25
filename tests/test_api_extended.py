"""Extended API endpoint tests."""

import pytest

try:
    from aiohttp import web  # noqa: F401  (importability probe)
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


async def test_skills_endpoint(aiohttp_client):
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())

    resp = await client.get("/api/v1/skills")
    assert resp.status == 200
    data = await resp.json()
    assert "skills" in data
    names = [s["name"] for s in data["skills"]]
    assert "brainstorm" in names
    assert "mullet-job" in names


async def test_tools_endpoint(aiohttp_client):
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())

    resp = await client.get("/api/v1/tools")
    assert resp.status == 200
    data = await resp.json()
    assert "tools" in data
    assert "toolsets" in data
    assert "profiles" in data
    assert "terminal" in data["tools"]
    assert len(data["tools"]) >= 13


async def test_health_dynamic_version(aiohttp_client):
    from mullet.server.app import create_app
    import mullet
    client = await aiohttp_client(create_app())

    resp = await client.get("/health")
    data = await resp.json()
    assert data["version"] == mullet.__version__


async def test_info_endpoint(aiohttp_client):
    """Info endpoint returns comprehensive system info."""
    from mullet.server.app import create_app
    import mullet
    client = await aiohttp_client(create_app())

    resp = await client.get("/api/v1/info")
    assert resp.status == 200
    data = await resp.json()
    assert data["version"] == mullet.__version__
    assert "tools" in data
    assert "skills" in data
    assert "providers" in data
    assert data["tool_count"] >= 13
    assert data["skill_count"] >= 6
    assert data["mode"] in ("solo", "enterprise")


async def test_explain_endpoint(aiohttp_client):
    """Explain endpoint returns routing analysis."""
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())

    resp = await client.post("/api/v1/explain", json={"task": "refactor auth for OAuth2"})
    assert resp.status == 200
    data = await resp.json()
    assert "complexity_score" in data
    assert "tier" in data
    assert data["tier"] in ("LOW", "STANDARD", "ADVANCED")
    assert "signals" in data
    assert "lexical" in data["signals"]
    assert "structural" in data["signals"]


async def test_explain_endpoint_empty_task(aiohttp_client):
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())

    resp = await client.post("/api/v1/explain", json={})
    assert resp.status == 400
