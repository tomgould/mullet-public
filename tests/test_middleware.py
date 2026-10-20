"""Tests for server middleware."""

import pytest

try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")

from mullet.server.middleware import RateLimiter


def test_rate_limiter_allows():
    rl = RateLimiter(max_requests=5, window_seconds=60)
    assert rl.is_allowed("client1") is True
    assert rl.remaining("client1") == 4


def test_rate_limiter_blocks():
    rl = RateLimiter(max_requests=2, window_seconds=60)
    rl.is_allowed("client1")
    rl.is_allowed("client1")
    assert rl.is_allowed("client1") is False


def test_rate_limiter_per_client():
    rl = RateLimiter(max_requests=1, window_seconds=60)
    rl.is_allowed("client1")
    assert rl.is_allowed("client2") is True  # Different client


def test_rate_limiter_remaining():
    rl = RateLimiter(max_requests=10, window_seconds=60)
    for _ in range(3):
        rl.is_allowed("c1")
    assert rl.remaining("c1") == 7


async def test_error_middleware_catches_exceptions(aiohttp_client):
    """Error middleware returns JSON for unhandled exceptions."""
    from mullet.server.middleware import error_handling_middleware

    async def bad_handler(request):
        raise ValueError("test error")

    app = web.Application()
    app.middlewares.append(error_handling_middleware)
    app.router.add_get("/test", bad_handler)

    client = await aiohttp_client(app)
    resp = await client.get("/test")
    assert resp.status == 500
    data = await resp.json()
    assert "error" in data
    assert data["error"]["type"] == "server_error"


async def test_error_middleware_passes_http_exception(aiohttp_client):
    """HTTPException should propagate through error middleware unchanged."""
    from mullet.server.middleware import error_handling_middleware

    async def handler(request):
        raise web.HTTPNotFound(reason="nothing here")

    app = web.Application()
    app.middlewares.append(error_handling_middleware)
    app.router.add_get("/missing", handler)

    client = await aiohttp_client(app)
    resp = await client.get("/missing")
    assert resp.status == 404


async def test_rate_limit_middleware_adds_headers(aiohttp_client):
    """rate_limit_middleware adds X-RateLimit-* headers to responses."""
    from mullet.server.middleware import rate_limit_middleware, RateLimiter
    import mullet.server.middleware as mw_mod

    old = mw_mod._rate_limiter
    mw_mod._rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

    async def handler(request):
        return web.Response(text="ok")

    app = web.Application()
    app.middlewares.append(rate_limit_middleware)
    app.router.add_get("/ok", handler)

    client = await aiohttp_client(app)
    resp = await client.get("/ok")
    assert resp.status == 200
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Limit" in resp.headers
    mw_mod._rate_limiter = old


async def test_rate_limit_middleware_blocks_excess(aiohttp_client):
    """rate_limit_middleware returns 429 when limit exceeded."""
    from mullet.server.middleware import rate_limit_middleware, RateLimiter
    import mullet.server.middleware as mw_mod

    old = mw_mod._rate_limiter
    mw_mod._rate_limiter = RateLimiter(max_requests=1, window_seconds=60)

    async def handler(request):
        return web.Response(text="ok")

    app = web.Application()
    app.middlewares.append(rate_limit_middleware)
    app.router.add_get("/ok", handler)

    client = await aiohttp_client(app)
    await client.get("/ok")  # consume the 1 allowed request
    resp = await client.get("/ok")  # this should be blocked
    assert resp.status == 429
    assert resp.headers.get("Retry-After") == "60"
    mw_mod._rate_limiter = old


def test_rate_limiter_stale_cleanup():
    """Periodic cleanup removes entries for idle clients."""
    import time
    rl = RateLimiter(max_requests=100, window_seconds=1)
    rl.is_allowed("idle-client")
    # Force the cleanup by faking _last_cleanup far in the past
    rl._last_cleanup = time.time() - 10
    # Advance apparent time past window by backdating entries
    rl._counters["idle-client"] = [time.time() - 5]  # expired
    # Trigger cleanup via is_allowed for a different client
    rl.is_allowed("trigger-client")
    assert "idle-client" not in rl._counters
