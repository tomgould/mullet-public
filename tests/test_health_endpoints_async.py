"""Cycle 26 — `/health`, `/v1/models`, `/api/v1/info` must never block on the
LMStudio subprocess probe.

The sync `detect_local_provider()` can shell out to `lms load` which takes
several seconds to fail. These handlers run inside the aiohttp event loop, so
the probe MUST be non-blocking:

- Cold cache → return immediately with `local=None`, schedule background refresh.
- Warm cache → return cached value immediately (already the pre-existing path).

Each test docstring names the expected failure mode so a RED run is easy to
diagnose.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

pytest.importorskip("aiohttp")


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a cleared local-provider cache + no pending refresh."""
    from mullet.core.providers import local_detection as ld
    ld.clear_local_provider_cache()
    # Reset async bookkeeping state added by Cycle 26
    if hasattr(ld, "_refresh_task"):
        ld._refresh_task = None  # type: ignore[attr-defined]
    yield
    ld.clear_local_provider_cache()
    if hasattr(ld, "_refresh_task"):
        ld._refresh_task = None  # type: ignore[attr-defined]


def _slow_probe(_delay: float = 5.0):
    """Factory for a sync detect stub that sleeps to simulate the lms subprocess."""
    def _probe():
        time.sleep(_delay)
        return None
    return _probe


class _StubProvider:
    """Minimal provider duck-type that matches what the handlers read."""
    name = "local"
    model = "stub-model-1"

    def health_check(self) -> bool:  # pragma: no cover — cache hot path
        return True


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


async def test_health_returns_fast_when_cache_cold(aiohttp_client):
    """Cold cache + slow probe → handler must return in <200ms, not wait 5s.

    FAIL MODE: handler awaits the sync probe inline and blocks ~5s.
    """
    from mullet.server.app import create_app
    from mullet.core.providers import local_detection as ld

    app = create_app()
    client = await aiohttp_client(app)

    with patch.object(ld, "detect_local_provider", side_effect=_slow_probe(5.0)):
        start = asyncio.get_event_loop().time()
        resp = await client.get("/health")
        elapsed = asyncio.get_event_loop().time() - start

    assert resp.status == 200
    assert elapsed < 0.2, f"/health blocked for {elapsed:.3f}s (expected <0.2s)"
    data = await resp.json()
    # Cold cache → local should be null on the fast path
    assert data["providers"]["local"]["available"] is False


async def test_health_returns_cached_value_immediately(aiohttp_client):
    """Warm cache → handler returns the stub's model within 50ms.

    FAIL MODE: handler re-probes instead of reading from cache.
    """
    from mullet.server.app import create_app
    from mullet.core.providers import local_detection as ld

    app = create_app()
    client = await aiohttp_client(app)

    # Pre-populate the cache with a fresh stub
    ld._cached_local_provider = _StubProvider()  # type: ignore[assignment]
    ld._cache_timestamp = time.monotonic()

    start = asyncio.get_event_loop().time()
    resp = await client.get("/health")
    elapsed = asyncio.get_event_loop().time() - start

    assert resp.status == 200
    assert elapsed < 0.05, f"/health took {elapsed:.3f}s with warm cache"
    data = await resp.json()
    assert data["providers"]["local"]["available"] is True
    assert data["providers"]["local"]["model"] == "stub-model-1"


async def test_health_schedules_background_refresh_when_cold(aiohttp_client):
    """Cold cache → handler must schedule a background probe (fire-and-forget).

    FAIL MODE: handler returns fast but never populates the cache, so every
    request triggers a new (wasted) probe.
    """
    from mullet.server.app import create_app
    from mullet.core.providers import local_detection as ld

    app = create_app()
    client = await aiohttp_client(app)

    probe_calls = {"n": 0}

    def _fast_probe():
        probe_calls["n"] += 1
        return None

    with patch.object(ld, "detect_local_provider", side_effect=_fast_probe):
        resp = await client.get("/health")
        assert resp.status == 200
        # Give the scheduled background task a beat to run.
        for _ in range(20):
            await asyncio.sleep(0.01)
            if probe_calls["n"] >= 1:
                break

    assert probe_calls["n"] >= 1, "expected background refresh to invoke probe at least once"


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


async def test_v1_models_does_not_block(aiohttp_client):
    """Cold cache + slow probe → /v1/models returns in <200ms.

    FAIL MODE: the handler inlines `detect_local_provider()` and waits.
    """
    from mullet.server.app import create_app
    from mullet.core.providers import local_detection as ld

    app = create_app()
    client = await aiohttp_client(app)

    with patch.object(ld, "detect_local_provider", side_effect=_slow_probe(5.0)):
        start = asyncio.get_event_loop().time()
        resp = await client.get("/v1/models")
        elapsed = asyncio.get_event_loop().time() - start

    assert resp.status == 200
    assert elapsed < 0.2, f"/v1/models blocked for {elapsed:.3f}s"
    data = await resp.json()
    # Always has the abstract `mullet` id even with no local backend
    ids = [m["id"] for m in data["data"]]
    assert "mullet" in ids


async def test_v1_models_includes_local_when_cache_warm(aiohttp_client):
    """Warm cache → local model is included in the /v1/models list."""
    from mullet.server.app import create_app
    from mullet.core.providers import local_detection as ld

    app = create_app()
    client = await aiohttp_client(app)

    ld._cached_local_provider = _StubProvider()  # type: ignore[assignment]
    ld._cache_timestamp = time.monotonic()

    resp = await client.get("/v1/models")
    assert resp.status == 200
    data = await resp.json()
    ids = [m["id"] for m in data["data"]]
    assert any("stub-model-1" in i for i in ids)


# ---------------------------------------------------------------------------
# /api/v1/info
# ---------------------------------------------------------------------------


async def test_api_v1_info_does_not_block(aiohttp_client):
    """Cold cache + slow probe → /api/v1/info returns in <200ms.

    FAIL MODE: `detect_local_provider()` runs inline and blocks the loop.

    (Note: this endpoint also builds the tool registry; that work is cheap
    enough to stay inline for now. The budget is 200ms specifically for
    the provider-detection portion.)
    """
    from mullet.server.app import create_app
    from mullet.core.providers import local_detection as ld

    app = create_app()
    client = await aiohttp_client(app)

    with patch.object(ld, "detect_local_provider", side_effect=_slow_probe(5.0)):
        start = asyncio.get_event_loop().time()
        resp = await client.get("/api/v1/info")
        elapsed = asyncio.get_event_loop().time() - start

    assert resp.status == 200
    assert elapsed < 0.5, f"/api/v1/info blocked for {elapsed:.3f}s"
    data = await resp.json()
    # Cold cache + slow probe → local reports unavailable on the fast path
    assert data["providers"]["local"]["available"] is False


async def test_api_v1_info_surfaces_local_when_cache_warm(aiohttp_client):
    """Warm cache → info endpoint surfaces the cached provider without probing."""
    from mullet.server.app import create_app
    from mullet.core.providers import local_detection as ld

    app = create_app()
    client = await aiohttp_client(app)

    ld._cached_local_provider = _StubProvider()  # type: ignore[assignment]
    ld._cache_timestamp = time.monotonic()

    resp = await client.get("/api/v1/info")
    assert resp.status == 200
    data = await resp.json()
    assert data["providers"]["local"]["available"] is True
    assert data["providers"]["local"]["model"] == "stub-model-1"


# ---------------------------------------------------------------------------
# Unit test for the new async helper
# ---------------------------------------------------------------------------


async def test_get_local_provider_nonblocking_returns_fast():
    """Direct unit test: the new async wrapper must not block the caller.

    FAIL MODE: the wrapper awaits the sync probe directly instead of using
    to_thread / background scheduling.
    """
    from mullet.core.providers import local_detection as ld

    assert hasattr(ld, "get_local_provider_nonblocking"), (
        "Cycle 26 adds `get_local_provider_nonblocking`; it is missing"
    )

    with patch.object(ld, "detect_local_provider", side_effect=_slow_probe(5.0)):
        start = asyncio.get_event_loop().time()
        result = await ld.get_local_provider_nonblocking()
        elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 0.2, f"wrapper blocked for {elapsed:.3f}s"
    # Cold cache + unresolved probe → None on the fast path
    assert result is None


async def test_get_local_provider_nonblocking_returns_cache_when_warm():
    """Warm cache → wrapper returns the cached provider immediately."""
    from mullet.core.providers import local_detection as ld

    ld._cached_local_provider = _StubProvider()  # type: ignore[assignment]
    ld._cache_timestamp = time.monotonic()

    start = asyncio.get_event_loop().time()
    result = await ld.get_local_provider_nonblocking()
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 0.05
    assert result is not None
    assert getattr(result, "model", None) == "stub-model-1"
