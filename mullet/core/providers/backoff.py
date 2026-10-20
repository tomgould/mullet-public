"""Pure backoff math — single source of truth for exponential backoff calculations.

`health.ProviderHealth.record_failure` and `retry._compute_delay` both delegate
to `compute_backoff` so the formula `base * factor**attempt` (with optional
clamping/cap/jitter) lives in exactly one place.
"""

import math


def compute_backoff(
    base: float,
    factor: float,
    attempt: int,
    *,
    max_delay: float = math.inf,
    max_attempt: int | None = None,
    jitter: bool = False,
) -> float:
    """Compute exponential backoff delay.

    Args:
        base: initial delay (seconds).
        factor: multiplier per attempt.
        attempt: 0-indexed attempt number.
        max_delay: cap on the returned delay.
        max_attempt: clamp ``attempt`` to this value before exponentiation
            (prevents runaway when callers pass unbounded counters).
        jitter: when True, multiply by a random uniform in [0.5, 1.0].

    Returns:
        Delay in seconds, in ``[0, max_delay]``.
    """
    n = attempt if max_attempt is None else min(attempt, max_attempt)
    try:
        delay = base * (factor ** n)
    except OverflowError:
        delay = max_delay
    delay = min(delay, max_delay)
    if jitter:
        import random
        delay *= 0.5 + random.random() * 0.5
    return max(0.0, delay)
