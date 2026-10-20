"""Canonical local-provider auto-detection with cache + health pings.

Extracted from ``mullet.cli.main`` so server/router code does not have to
import from the CLI layer (P2 — duplicate-logic cleanup, Cycle 7 C1).

``mullet.cli.main`` keeps a back-compat re-export of :func:`detect_local_provider`
so callers that patched ``mullet.cli.main.detect_local_provider`` continue to
work. New code should import from this module directly.

Async-safe entrypoint (Cycle 26): :func:`get_local_provider_nonblocking` is the
coroutine variant used by aiohttp handlers. It reads the module cache directly
and schedules a background refresh via ``asyncio.to_thread`` so the event loop
never blocks on the LMStudio subprocess probe.
"""

from __future__ import annotations

import asyncio
import logging
import time

from mullet.core.providers.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger("mullet.core.providers.local_detection")

# Ports probed in order. 1234 = LM Studio (handled via LMStudioManager),
# 11434 = Ollama, 8000 = generic OpenAI-compat.
LOCAL_PORTS = [1234, 11434, 8000]

_CACHE_TTL: float = 30.0  # Re-probe every 30 seconds
_cached_local_provider: OpenAICompatibleProvider | None = None
_cache_timestamp: float = 0.0

# In-flight background refresh coordination (Cycle 26).
# At most one thread-pool probe may be running at a time so we do not stack
# `lms load` subprocesses under request pressure.
_refresh_task: asyncio.Task | None = None


def clear_local_provider_cache() -> None:
    """Drop the cached provider so the next call re-probes from scratch.

    Useful for tests and for forcing a refresh after a model swap.
    """
    global _cached_local_provider, _cache_timestamp, _refresh_task
    _cached_local_provider = None
    _cache_timestamp = 0.0
    _refresh_task = None


def detect_local_provider() -> OpenAICompatibleProvider | None:
    """Auto-detect a running local LLM server, loading the best model if needed.

    Uses LMStudioManager for model selection and lifecycle. Falls back to
    plain port-scan for non-LM-Studio servers (Ollama, plain OpenAI-compat).
    Cached with 30s TTL + health ping.
    """
    global _cached_local_provider, _cache_timestamp

    now = time.monotonic()

    # Return cached if fresh AND healthy
    if _cached_local_provider and (now - _cache_timestamp) < _CACHE_TTL:
        if _cached_local_provider.health_check():
            return _cached_local_provider
        _cached_local_provider = None

    # Try LM Studio first (port 1234) using the full lifecycle manager
    try:
        from mullet.core.providers.lmstudio_manager import LMStudioManager
        mgr = LMStudioManager(port=1234)
        model_id, was_loaded = mgr.get_or_load_provider(ttl=60)
        if model_id:
            provider = OpenAICompatibleProvider(
                base_url="http://localhost:1234/v1",
                model=model_id,
            )
            _cached_local_provider = provider
            _cache_timestamp = now
            if not was_loaded:
                logger.info("Loaded local model: %s (TTL=60s)", model_id)
            return provider
    except Exception as e:
        logger.debug("LMStudioManager failed: %s", e)

    # Fallback: plain port scan for Ollama / other OpenAI-compat servers
    import urllib.request
    import json as _json
    for port in LOCAL_PORTS[1:]:  # skip 1234 — already tried above
        url = f"http://localhost:{port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status != 200:
                    continue
                body = _json.loads(resp.read())
                models = body.get("data", [])
                if not models:
                    continue
                embedding_markers = ("embed", "clip", "vision", "rerank")
                chat_model = next(
                    (m["id"] for m in models
                     if not any(e in m["id"].lower() for e in embedding_markers)),
                    "default",
                )
                _cached_local_provider = OpenAICompatibleProvider(
                    base_url=f"http://localhost:{port}/v1",
                    model=chat_model,
                )
                _cache_timestamp = now
                return _cached_local_provider
        except Exception as _exc:
            logger.debug(
                "local_detection: failed to detect local provider on port %s: %s",
                port,
                _exc,
            )
            continue

    _cached_local_provider = None
    _cache_timestamp = now
    return None


def _cache_is_fresh() -> bool:
    """True when we have a cached provider younger than the TTL.

    No side effects; no network; safe to call from the event loop.
    """
    if _cached_local_provider is None:
        return False
    return (time.monotonic() - _cache_timestamp) < _CACHE_TTL


async def get_local_provider_nonblocking() -> OpenAICompatibleProvider | None:
    """Async, non-blocking variant of :func:`detect_local_provider`.

    Stale-while-revalidate semantics:

    - **Warm cache (fresh)**: return the cached provider immediately, no probe.
    - **Warm cache (stale)**: return the cached provider immediately AND kick off
      a background refresh in a worker thread.
    - **Cold cache**: return ``None`` immediately AND kick off a background
      probe. The next caller after the probe finishes will see the populated
      cache.

    The handler path therefore never blocks on the LMStudio subprocess probe.
    At most one refresh is in flight at a time (deduplicated via
    :data:`_refresh_task`).
    """
    global _refresh_task

    # Fast path: fresh cache.
    if _cache_is_fresh():
        return _cached_local_provider

    # Stale or cold → schedule a background probe if one isn't already running.
    # We deliberately do NOT await it — the point is to return immediately.
    if _refresh_task is None or _refresh_task.done():
        try:
            loop = asyncio.get_running_loop()
            _refresh_task = loop.create_task(_refresh_cache_in_background())
        except RuntimeError:
            # No running loop (shouldn't happen from aiohttp handlers). Fall
            # back to synchronous probe so behaviour is at least defined.
            logger.debug("get_local_provider_nonblocking: no running loop; sync fallback")
            return detect_local_provider()

    # Return whatever we have — the stale value if any, else None.
    return _cached_local_provider


async def _refresh_cache_in_background() -> None:
    """Run :func:`detect_local_provider` in a worker thread and log the outcome.

    Errors are swallowed (logged at WARNING) so a failed refresh never raises
    into unrelated request handlers.
    """
    try:
        provider = await asyncio.to_thread(detect_local_provider)
        if provider is not None:
            logger.debug(
                "background refresh: local provider available at %s (model=%s)",
                getattr(provider, "base_url", "?"),
                getattr(provider, "model", "?"),
            )
        else:
            logger.debug("background refresh: no local provider available")
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("background refresh failed: %s", exc)
