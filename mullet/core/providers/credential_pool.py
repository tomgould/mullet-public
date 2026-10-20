"""Credential pool — multiple API keys per provider with rotation.

From Hermes: when one key hits rate limits, automatically rotate
to the next. Supports multiple selection strategies.
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Credential:
    """A single API key with usage tracking."""
    api_key: str
    name: str = ""
    total_requests: int = 0
    total_tokens: int = 0
    exhausted_until: float = 0  # Timestamp when cooldown expires
    last_used: float = 0
    consecutive_failures: int = 0

    @property
    def is_available(self) -> bool:
        """Return True if the credential is not on cooldown."""
        if self.exhausted_until and time.time() < self.exhausted_until:
            return False
        return True

    @property
    def cooldown_remaining(self) -> float:
        """Return seconds remaining until the credential is available."""
        if not self.exhausted_until:
            return 0
        return max(0, self.exhausted_until - time.time())


class CredentialPool:
    """Manage multiple API keys for a single provider.

    Selection strategies:
    - round_robin: cycle through keys in order
    - least_used: pick the key with fewest requests
    - random: pick randomly

    On rate limit (429): key enters 1-hour cooldown.
    On quota exceeded (402): key enters 24-hour cooldown.
    """

    def __init__(
        self,
        credentials: list[Credential] | None = None,
        strategy: str = "round_robin",
    ):
        self._credentials = credentials or []
        self._strategy = strategy
        self._round_robin_idx = 0

    def add(self, api_key: str, name: str = ""):
        """Add a credential to the pool."""
        self._credentials.append(Credential(api_key=api_key, name=name or f"key_{len(self._credentials)}"))

    def get_next(self) -> Credential | None:
        """Get the next available credential using the configured strategy."""
        available = [c for c in self._credentials if c.is_available]
        if not available:
            return None

        if self._strategy == "least_used":
            cred = min(available, key=lambda c: c.total_requests)
        elif self._strategy == "random":
            import random
            cred = random.choice(available)
        else:  # round_robin
            cred = None
            for _ in range(len(self._credentials)):
                idx = self._round_robin_idx % len(self._credentials)
                self._round_robin_idx += 1
                candidate = self._credentials[idx]
                if candidate.is_available:
                    cred = candidate
                    break
            if cred is None:  # pragma: no cover — available non-empty guarantees a match
                return None

        cred.last_used = time.time()
        cred.total_requests += 1
        return cred

    def mark_exhausted(self, api_key: str, error_code: int = 429):
        """Mark a credential as exhausted with cooldown."""
        for cred in self._credentials:
            if cred.api_key == api_key:
                cred.consecutive_failures += 1
                if error_code == 402:
                    # Quota exceeded: 24h cooldown
                    cred.exhausted_until = time.time() + 86400
                    logger.warning("Key %s quota exceeded, 24h cooldown", cred.name)
                else:
                    # Rate limit: 1h cooldown
                    cred.exhausted_until = time.time() + 3600
                    logger.warning("Key %s rate limited, 1h cooldown", cred.name)
                break

    def mark_success(self, api_key: str, tokens: int = 0):
        """Record successful use of a credential."""
        for cred in self._credentials:
            if cred.api_key == api_key:
                cred.consecutive_failures = 0
                cred.total_tokens += tokens
                break

    @property
    def available_count(self) -> int:
        """Return the number of credentials currently available."""
        return sum(1 for c in self._credentials if c.is_available)

    @property
    def total_count(self) -> int:
        """Return the total number of credentials in the pool."""
        return len(self._credentials)

    def status(self) -> list[dict]:
        """Get pool status for display."""
        return [
            {
                "name": c.name,
                "available": c.is_available,
                "requests": c.total_requests,
                "tokens": c.total_tokens,
                "cooldown": round(c.cooldown_remaining, 0),
                "failures": c.consecutive_failures,
            }
            for c in self._credentials
        ]
