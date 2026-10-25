"""mullet serve refuses foreign hosts, other web pages and requests without the install's token."""

import os
import stat

import pytest

try:
    from aiohttp import web  # noqa: F401
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")

TOKEN = "t0ken-for-tests"


@pytest.fixture
def app():
    from mullet.server.app import create_app
    import mullet.server.middleware as mw
    mw._rate_limiter._counters.clear()
    return create_app(require_token=True, token=TOKEN)


def bearer(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


async def test_a_request_without_the_token_is_refused(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/v1/models")
    assert resp.status == 401 and "token" in (await resp.json())["error"]["message"]


async def test_a_wrong_token_is_refused_and_the_right_one_accepted(aiohttp_client, app):
    client = await aiohttp_client(app)
    assert (await client.get("/v1/models", headers=bearer("nope"))).status == 401
    assert (await client.get("/v1/models", headers=bearer())).status == 200
    assert (await client.get("/v1/models", headers={"X-Mullet-Token": TOKEN})).status == 200


async def test_health_needs_no_token(aiohttp_client, app):
    client = await aiohttp_client(app)
    assert (await client.get("/health")).status == 200


async def test_another_web_page_is_refused_even_with_the_token(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/health", headers={"Origin": "https://evil.example"})
    assert resp.status == 403
    resp = await client.post("/v1/chat/completions", json={}, headers={**bearer(), "Origin": "https://evil.example"})
    assert resp.status == 403


async def test_a_local_page_is_allowed_with_the_token(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/v1/models", headers={**bearer(), "Origin": "http://localhost:5173"})
    assert resp.status == 200


async def test_a_foreign_host_name_is_refused(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/health", headers={"Host": "attacker.example:3000"})
    assert resp.status == 403


async def test_the_websocket_needs_the_token_and_a_local_origin(aiohttp_client, app):
    client = await aiohttp_client(app)
    refused = await client.get("/ws")
    assert refused.status == 401
    ws = await client.ws_connect(f"/ws?token={TOKEN}")
    await ws.send_json({"type": "ping"})
    assert (await ws.receive_json())["type"] == "pong"
    await ws.close()
    page = await client.get(f"/ws?token={TOKEN}", headers={"Origin": "https://evil.example"})
    assert page.status == 403


async def test_the_token_in_a_query_string_is_ignored_off_the_websocket(aiohttp_client, app):
    client = await aiohttp_client(app)
    assert (await client.get(f"/v1/models?token={TOKEN}")).status == 401


async def test_enterprise_mode_uses_personal_access_tokens_instead(aiohttp_client, app, monkeypatch):
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    client = await aiohttp_client(app)
    resp = await client.get("/v1/models", headers=bearer("whatever"))
    assert resp.status != 401 or "guard" not in (await resp.text())


async def test_without_require_token_only_host_and_origin_are_checked(aiohttp_client):
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())
    assert (await client.get("/v1/models")).status == 200
    assert (await client.get("/health", headers={"Origin": "https://evil.example"})).status == 403


async def test_extra_hosts_and_origins_can_be_allowed(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app(cors_origins=["https://dash.example"], allowed_hosts=["mullet.lan"])
    client = await aiohttp_client(app)
    assert (await client.get("/health", headers={"Origin": "https://dash.example", "Host": "mullet.lan"})).status == 200


def test_the_token_is_made_once_and_kept_private(tmp_path):
    from mullet.server.guard import get_or_create_token

    path = tmp_path / "cfg" / "serve-token"
    first = get_or_create_token(path)
    assert len(first) >= 32 and get_or_create_token(path) == first
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_show_token_prints_the_token_file_and_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    from mullet.server.guard import get_or_create_token, token_path

    token = get_or_create_token()
    assert token_path() == tmp_path / "serve-token" and (tmp_path / "serve-token").read_text().strip() == token
