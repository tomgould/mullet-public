"""Server middleware — rate limiting, error handling, request validation."""

import logging
import time
from collections import defaultdict

from aiohttp import web

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter per IP address.

    Uses a sliding window counter. Resets after window expires.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._counters: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.time()

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request is allowed."""
        now = time.time()
        cutoff = now - self.window

        # Periodic full cleanup to prevent memory leak from inactive clients
        if now - self._last_cleanup > self.window * 2:
            stale = [k for k, v in self._counters.items() if not v or v[-1] < cutoff]
            for k in stale:
                del self._counters[k]
            self._last_cleanup = now

        # Clean old entries for this client
        entries = self._counters[client_id]
        self._counters[client_id] = [t for t in entries if t > cutoff]

        if len(self._counters[client_id]) >= self.max_requests:
            return False

        self._counters[client_id].append(now)
        return True

    def remaining(self, client_id: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        cutoff = now - self.window
        recent = [t for t in self._counters[client_id] if t > cutoff]
        return max(0, self.max_requests - len(recent))


# Global rate limiter instance
_rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


@web.middleware
async def rate_limit_middleware(request, handler):
    """Rate limiting middleware — 60 requests per minute per IP."""
    client_ip = request.remote or "unknown"

    if not _rate_limiter.is_allowed(client_ip):
        return web.json_response(
            {"error": {"message": "Rate limit exceeded. Try again in a minute.", "type": "rate_limit_error"}},
            status=429,
            headers={"Retry-After": "60"},
        )

    response = await handler(request)
    remaining = _rate_limiter.remaining(client_ip)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = str(_rate_limiter.max_requests)
    return response


@web.middleware
async def error_handling_middleware(request, handler):
    """Global error handler — catches unhandled exceptions and returns proper JSON."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise  # Let aiohttp handle HTTP exceptions
    except Exception:
        logger.exception("Unhandled error in %s %s", request.method, request.path)
        return web.json_response(
            {"error": {"message": "Internal server error", "type": "server_error"}},
            status=500,
        )
