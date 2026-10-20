"""Thread-safe iteration budget for agent execution."""

import threading
import time


class IterationBudget:
    """Thread-safe iteration counter with optional wall-clock timeout.

    From Hermes: parent budget independent of child budgets.
    Each agent gets its own budget — parent default 90, subagent default 50.

    Args:
        max_total: Maximum number of LLM iterations.
        timeout_seconds: Optional wall-clock timeout. None = no timeout.
    """

    def __init__(self, max_total: int = 90, timeout_seconds: float | None = None):
        """Initialise the budget with a hard iteration cap and optional wall-clock deadline."""
        self.max_total = max_total
        self.timeout_seconds = timeout_seconds
        self._used = 0
        self._start_time = time.monotonic()
        self._lock = threading.Lock()

    @property
    def timed_out(self) -> bool:
        """True if wall-clock timeout has been exceeded."""
        if self.timeout_seconds is None:
            return False
        return (time.monotonic() - self._start_time) >= self.timeout_seconds

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since budget was created."""
        return time.monotonic() - self._start_time

    def consume(self) -> bool:
        """Consume one iteration. Returns False if budget or timeout exhausted."""
        if self.timed_out:
            return False
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self, count: int = 1):
        """Refund iterations (e.g., for zero-context-cost tool calls)."""
        with self._lock:
            self._used = max(0, self._used - count)

    @property
    def used(self) -> int:
        """Number of iterations consumed so far (thread-safe snapshot)."""
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        """Iterations still available before the budget is exhausted."""
        with self._lock:
            return max(0, self.max_total - self._used)

    @property
    def exhausted(self) -> bool:
        """True once `used` has reached `max_total`."""
        with self._lock:
            return self._used >= self.max_total

    @property
    def pressure(self) -> str:
        """Return budget pressure level: none, caution, urgent, critical.

        Thresholds (from Hermes):
        - caution at 70%: nudge LLM to start wrapping up
        - urgent at 90%: strong signal to respond now
        - critical at remaining <= 1: last chance
        """
        with self._lock:
            if self.max_total <= 0:
                return "critical"
            ratio = self._used / self.max_total
            remaining = self.max_total - self._used
            if remaining <= 1:
                return "critical"
            if ratio >= 0.9:
                return "urgent"
            if ratio >= 0.7:
                return "caution"
            return "none"

    def get_pressure_message(self) -> str | None:
        """Return a pressure message for injection into tool results, or None.

        Messages are injected into tool results (not separate messages)
        to preserve prompt caching.
        """
        level = self.pressure
        remaining = self.remaining
        if level == "critical":
            return f"[CRITICAL: {remaining} iteration(s) remaining. Provide your final answer NOW.]"
        if level == "urgent":
            return f"[URGENT: {remaining}/{self.max_total} iterations remaining. Wrap up your work immediately.]"
        if level == "caution":
            return f"[NOTE: {remaining}/{self.max_total} iterations remaining. Start wrapping up — focus on the most important remaining work.]"
        return None
