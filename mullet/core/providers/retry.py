"""Universal retry with exponential backoff for LLM provider calls.

Handles transient errors (rate limits, server errors, timeouts) across
all providers with configurable retry behavior.

Error classification is delegated to `mullet.core.providers.health.classify_error`
(single source of truth). `is_retryable` converts a category into a boolean.
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from mullet.core.providers.backoff import compute_backoff
from mullet.core.providers.health import classify_error

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Error categories that warrant a retry. Everything else (e.g. "auth",
# "unknown") is surfaced immediately.
RETRYABLE_CATEGORIES = frozenset({
    "rate_limit",
    "overloaded",
    "server_error",
    "connection",
})


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.initial_delay < 0:
            raise ValueError(f"initial_delay must be >= 0, got {self.initial_delay}")
        if self.backoff_factor < 1.0:
            raise ValueError(f"backoff_factor must be >= 1.0, got {self.backoff_factor}")


@dataclass
class RetryResult:
    """Result of a retried operation."""
    value: object = None
    attempts: int = 1
    total_delay: float = 0.0
    last_error: str | None = None
    success: bool = True


def is_retryable(error: Exception) -> bool:
    """Determine if an error is transient and worth retrying.

    Delegates to `health.classify_error` — an error is retryable iff its
    category is in `RETRYABLE_CATEGORIES`. Extend `classify_error` (not this
    function) when a new pattern needs to become retryable.
    """
    return classify_error(error) in RETRYABLE_CATEGORIES


def _compute_delay(attempt: int, config: RetryConfig) -> float:
    """Compute delay for a given attempt number.

    Thin wrapper over `mullet.core.providers.backoff.compute_backoff` —
    extend that module rather than this function for new backoff behaviour.
    """
    return compute_backoff(
        base=config.initial_delay,
        factor=config.backoff_factor,
        attempt=attempt,
        max_delay=config.max_delay,
        jitter=config.jitter,
    )


def with_retry(
    func: Callable[[], T],
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> RetryResult:
    """Execute a function with exponential backoff retry.

    Args:
        func: The function to call (no arguments).
        config: Retry configuration. Uses defaults if None.
        on_retry: Optional callback(attempt, error, delay) called before each retry.

    Returns:
        RetryResult with the value or last error.
    """
    config = config or RetryConfig()
    total_delay = 0.0
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            value = func()
            return RetryResult(
                value=value,
                attempts=attempt + 1,
                total_delay=total_delay,
                success=True,
            )
        except Exception as e:
            last_error = e

            if attempt >= config.max_retries:
                break

            if not is_retryable(e):
                break

            delay = _compute_delay(attempt, config)
            total_delay += delay

            logger.warning(
                "Retry %d/%d after %.1fs: %s",
                attempt + 1, config.max_retries, delay, str(e)[:80],
            )

            if on_retry:
                on_retry(attempt + 1, e, delay)

            time.sleep(delay)

    return RetryResult(
        value=None,
        attempts=config.max_retries + 1,
        total_delay=total_delay,
        last_error=str(last_error) if last_error else None,
        success=False,
    )
