"""Provider health tracking with exponential backoff.

From Hermes: provider failures trigger backoff windows.
Different error types get different durations.
At the start of each turn, primary provider is restored.
"""

import logging
import time
from dataclasses import dataclass, field

from mullet.core.providers.backoff import compute_backoff

logger = logging.getLogger(__name__)


# Backoff durations by error type (seconds)
BACKOFF_DURATIONS = {
    "rate_limit": 30,       # 429 Too Many Requests
    "overloaded": 60,       # 503 Service Unavailable
    "server_error": 60,     # 500 Internal Server Error
    "connection": 120,      # Connection refused, timeout, DNS failure
    "auth": 3600,           # 401/403 — likely permanent until key is fixed
    "unknown": 60,          # Unrecognized errors
}

# Max consecutive failures before provider is considered dead
MAX_CONSECUTIVE_FAILURES = 5

# Exponential backoff multiplier
BACKOFF_MULTIPLIER = 1.5


_LATENCY_WINDOW = 20  # number of recent calls to keep for rolling average


@dataclass
class ProviderHealth:
    """Track a single provider's health with exponential backoff and latency."""

    provider_name: str
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float = 0
    last_failure_type: str = ""
    last_failure_message: str = ""
    backoff_until: float = 0
    last_success_time: float = 0
    _latency_samples: list = field(default_factory=list)

    def record_failure(self, error: Exception):
        """Record a failure and calculate backoff."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        self.last_failure_message = str(error)[:200]

        # Classify error
        error_type = classify_error(error)
        self.last_failure_type = error_type

        # Calculate backoff with exponential growth
        base_duration = BACKOFF_DURATIONS.get(error_type, BACKOFF_DURATIONS["unknown"])
        duration = compute_backoff(
            base=base_duration,
            factor=BACKOFF_MULTIPLIER,
            attempt=self.consecutive_failures - 1,
            max_attempt=5,
        )

        self.backoff_until = time.time() + duration
        logger.warning(
            "Provider %s: %s failure #%d (%s), backing off %.0fs",
            self.provider_name, error_type, self.consecutive_failures,
            self.last_failure_message[:80], duration,
        )

    def record_success(self, latency_ms: float | None = None):
        """Record a successful call — reset failure state and track latency."""
        self.consecutive_failures = 0
        self.total_successes += 1
        self.last_success_time = time.time()
        self.backoff_until = 0
        if latency_ms is not None:
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > _LATENCY_WINDOW:
                self._latency_samples = self._latency_samples[-_LATENCY_WINDOW:]

    @property
    def avg_latency_ms(self) -> float | None:
        """Rolling average latency over last N calls. None if no data."""
        if not self._latency_samples:
            return None
        return sum(self._latency_samples) / len(self._latency_samples)

    @property
    def p90_latency_ms(self) -> float | None:
        """P90 latency over recorded samples. None if <3 samples."""
        if len(self._latency_samples) < 3:
            return None
        s = sorted(self._latency_samples)
        idx = int(len(s) * 0.9)
        return s[min(idx, len(s) - 1)]

    @property
    def is_available(self) -> bool:
        """True if provider is past its backoff window."""
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return False
        if self.backoff_until and time.time() < self.backoff_until:
            return False
        return True

    @property
    def backoff_remaining(self) -> float:
        """Seconds until provider is available again."""
        if self.backoff_until <= 0:
            return 0
        remaining = self.backoff_until - time.time()
        return max(0, remaining)

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 1.0
        return self.total_successes / total

    def to_dict(self) -> dict:
        """Serialize for status endpoints."""
        d = {
            "provider": self.provider_name,
            "available": self.is_available,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": round(self.success_rate, 3),
            "backoff_remaining": round(self.backoff_remaining, 1),
            "last_failure_type": self.last_failure_type,
            "last_failure_message": self.last_failure_message[:100],
        }
        if self.avg_latency_ms is not None:
            d["avg_latency_ms"] = round(self.avg_latency_ms, 1)
        if self.p90_latency_ms is not None:
            d["p90_latency_ms"] = round(self.p90_latency_ms, 1)
        return d


class HealthTracker:
    """Track health across all providers."""

    def __init__(self):
        """Initialise an empty per-provider health registry."""
        self._providers: dict[str, ProviderHealth] = {}

    def get(self, provider_name: str) -> ProviderHealth:
        """Get or create health tracker for a provider."""
        if provider_name not in self._providers:
            self._providers[provider_name] = ProviderHealth(provider_name=provider_name)
        return self._providers[provider_name]

    def record_failure(self, provider_name: str, error: Exception):
        """Record a failure for *provider_name* and update its backoff window."""
        self.get(provider_name).record_failure(error)

    def record_success(self, provider_name: str):
        """Record a successful call for *provider_name* and reset failure state."""
        self.get(provider_name).record_success()

    def is_available(self, provider_name: str) -> bool:
        """Return True if *provider_name* is past its backoff window and under the failure cap."""
        return self.get(provider_name).is_available

    def get_available(self, provider_names: list[str]) -> list[str]:
        """Filter to only available providers."""
        return [n for n in provider_names if self.is_available(n)]

    def status(self) -> list[dict]:
        """Get health status for all tracked providers."""
        return [h.to_dict() for h in self._providers.values()]


def classify_error(error: Exception) -> str:
    """Classify an error into a backoff category.

    Single source of truth for provider error classification. `retry.is_retryable`
    delegates here — any new retryable pattern should be added to this function
    rather than duplicated in retry.py.
    """
    msg = str(error).lower()
    error_type = type(error).__name__.lower()

    # Rate limiting
    if (
        "429" in msg
        or ("rate" in msg and "limit" in msg)
        or "quota" in msg
        or "too many requests" in msg
        or "resource exhausted" in msg
    ):
        return "rate_limit"

    # Server overloaded / transient
    if (
        "503" in msg
        or "overloaded" in msg
        or "capacity" in msg
        or "service unavailable" in msg
        or "try again" in msg
    ):
        return "overloaded"

    # Server errors (500-class, excluding 503 which is overloaded above)
    if (
        "500" in msg
        or "502" in msg
        or "504" in msg
        or "internal server error" in msg
        or "internal error" in msg
        or "bad gateway" in msg
        or "gateway timeout" in msg
    ):
        return "server_error"

    # Auth errors
    if "401" in msg or "403" in msg or "unauthorized" in msg or "invalid.*key" in msg:
        return "auth"

    # Connection errors (includes timeouts and network failures)
    if any(t in error_type for t in ("connection", "timeout", "dns", "socket", "network")):
        return "connection"
    if any(
        t in msg
        for t in (
            "connection",
            "timeout",
            "refused",
            "unreachable",
            "dns",
            "connection reset",
            "network",
        )
    ):
        return "connection"

    return "unknown"
