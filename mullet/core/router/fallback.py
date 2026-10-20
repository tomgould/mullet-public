"""FallbackManager — provider failover with health tracking.

From Hermes: when a provider fails, walk the fallback chain.
Restore primary at start of each turn (turn-scoped fallback).
Integrates with ProviderHealth for exponential backoff.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mullet.core.providers.base import Provider
    from mullet.core.providers.health import HealthTracker

logger = logging.getLogger(__name__)


class FallbackManager:
    """Manages provider failover with health-aware selection.

    Key behaviors:
    - On failure: try next healthy provider in the chain
    - Track which providers have been tried this turn
    - Restore primary at start of each turn
    - Integrate with HealthTracker for backoff timing
    """

    def __init__(
        self,
        primary: "Provider",
        fallbacks: list["Provider"],
        health: "HealthTracker | None" = None,
        allowed_providers: frozenset[str] | None = None,
    ):
        self._primary = primary
        self._fallbacks = fallbacks
        self._health = health
        self._allowed = allowed_providers
        self._current = primary
        self._tried: set[str] = set()
        self._activated = False

    @property
    def current(self) -> "Provider":
        """Return the currently selected provider."""
        return self._current

    @property
    def is_fallback_active(self) -> bool:
        """Return True if a fallback provider is currently active."""
        return self._activated

    def on_turn_start(self):
        """Restore primary provider at start of each turn.

        From Hermes: don't let one transient failure pin
        the session to the fallback permanently.
        """
        if self._activated:
            self._current = self._primary
            self._activated = False
            self._tried.clear()
            logger.debug("Restored primary: %s", self._primary.name)

    def on_failure(self, error: Exception) -> "Provider | None":
        """Handle provider failure. Returns next provider or None if exhausted."""
        self._tried.add(self._current.name)

        # Record failure in health tracker
        if self._health:
            self._health.record_failure(self._current.name, error)

        # Find next healthy, untried provider
        for fb in self._fallbacks:
            if self._allowed is not None and fb.name not in self._allowed:
                logger.debug("skipping fallback %s — not in allowlist %s", fb.name, self._allowed)
                continue
            if fb.name in self._tried:
                continue
            if self._health and not self._health.is_available(fb.name):
                continue

            logger.warning("Failing over from %s to %s: %s",
                           self._current.name, fb.name, str(error)[:80])
            self._current = fb
            self._activated = True
            return fb

        logger.error("Fallback chain exhausted. Tried: %s", self._tried)
        return None

    def on_success(self):
        """Record successful call."""
        if self._health:
            self._health.record_success(self._current.name)

    @property
    def chain_status(self) -> list[dict]:
        """Get status of the fallback chain."""
        chain = [{"name": self._primary.name, "role": "primary", "active": self._current == self._primary}]
        for fb in self._fallbacks:
            healthy = self._health.is_available(fb.name) if self._health else True
            chain.append({
                "name": fb.name,
                "role": "fallback",
                "active": self._current == fb,
                "healthy": healthy,
            })
        return chain
