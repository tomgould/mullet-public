"""Simple TTL cache for search results and LLM responses.

Usage:
    from mullet.knowledge.engine.cache import cached

    @cached(ttl=300)
    def run_search(query: str, top_k: int = 10):
        ...

Cache key is built from all positional and keyword arguments.
Thread-safe via a simple dict (GIL protects single dict ops in CPython).
"""

import functools
import hashlib
import json
import time
from typing import Any, Callable


_CACHE: dict[str, tuple[float, Any]] = {}  # key -> (expires_at, value)


def cached(ttl: float = 300) -> Callable:
    """Decorator: cache function return value for `ttl` seconds.

    Args:
        ttl: Time-to-live in seconds. 0 disables caching.
    """
    def decorator(func: Callable) -> Callable:
        """Wrap ``func`` with the configured TTL cache."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Cached wrapper that checks the TTL cache before invoking ``func``."""
            if ttl <= 0:
                return func(*args, **kwargs)

            cache_key = _make_key(func.__qualname__, args, kwargs)
            now = time.time()

            if cache_key in _CACHE:
                expires_at, value = _CACHE[cache_key]
                if now < expires_at:
                    return value
                del _CACHE[cache_key]

            result = func(*args, **kwargs)
            _CACHE[cache_key] = (now + ttl, result)
            return result

        wrapper.cache_clear = lambda: _clear_prefix(func.__qualname__)
        wrapper.cache_info = lambda: _info_prefix(func.__qualname__)
        return wrapper

    return decorator


def _make_key(prefix: str, args: tuple, kwargs: dict) -> str:
    try:
        payload = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = str(args) + str(sorted(kwargs.items()))
    return prefix + ":" + hashlib.md5(payload.encode()).hexdigest()


def _clear_prefix(prefix: str) -> int:
    keys = [k for k in _CACHE if k.startswith(prefix + ":")]
    for k in keys:
        del _CACHE[k]
    return len(keys)


def _info_prefix(prefix: str) -> dict:
    now = time.time()
    keys = [k for k in _CACHE if k.startswith(prefix + ":")]
    live = sum(1 for k in keys if _CACHE[k][0] > now)
    return {"cached": live, "expired": len(keys) - live, "total": len(_CACHE)}


def clear_all_caches() -> None:
    """Clear all cached entries (useful for testing)."""
    _CACHE.clear()
