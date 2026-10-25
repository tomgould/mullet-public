"""Check what the ledger says a job spent against what the providers say.

The ledger's cost comes from the responses the key proxy saw, but only the
provider knows what it billed. A snapshot of each provider's own figure is
taken when a job starts and when it stops; the difference is compared with the
job's recorded spend. The provider's figure also counts anything else using the
same key, so a difference is a prompt to look, not proof of a fault.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

PROVIDERS = ("openrouter", "kimi")  # the paid rungs whose balances can be probed
# A difference smaller than this share of the recorded spend, or this many dollars, is rounding.
TOLERANCE_SHARE = 0.03
TOLERANCE_USD = 0.25

Snapshot = dict[str, dict[str, Any]]  # provider -> {"used_usd": float | None, "remaining_usd": float | None}


def probe_providers() -> Snapshot:
    """Each paid provider's own usage or remaining credit now; providers that can't be probed are left out."""
    from mullet.core.providers.budget_probe import probe_all

    try:
        balances = asyncio.run(probe_all(include=list(PROVIDERS), force_refresh=True))
    except RuntimeError as exc:  # already inside an event loop: don't nest one
        logger.warning("reconcile: can't probe providers here: %s", exc)
        return {}
    found: Snapshot = {}
    for key, balance in balances.items():
        if key in PROVIDERS and not balance.error:
            found[key] = {"used_usd": balance.credits_used_usd, "remaining_usd": balance.credits_remaining_usd}
    return found


def provider_spent(before: dict[str, Any] | None, after: dict[str, Any] | None) -> float | None:
    """What the provider says was spent between two snapshots, or None when it can't be told.

    A provider that reports usage gives the rise in usage; one that reports only
    a balance gives the fall in the balance.
    """
    if not before or not after:
        return None
    if before.get("used_usd") is not None and after.get("used_usd") is not None:
        return max(0.0, float(after["used_usd"]) - float(before["used_usd"]))
    if before.get("remaining_usd") is not None and after.get("remaining_usd") is not None:
        return max(0.0, float(before["remaining_usd"]) - float(after["remaining_usd"]))
    return None


@dataclass
class Reconciliation:
    """One provider's recorded spend against its own figure."""

    provider: str
    recorded: float
    provider_says: float | None

    @property
    def difference(self) -> float | None:
        return None if self.provider_says is None else self.provider_says - self.recorded

    @property
    def off(self) -> bool:
        """True when the provider billed noticeably more or less than the ledger recorded."""
        diff = self.difference
        return diff is not None and abs(diff) > max(TOLERANCE_USD, TOLERANCE_SHARE * self.recorded)

    def line(self) -> str:
        if self.provider_says is None:
            return f"{self.provider}: recorded ${self.recorded:.2f}; the provider's own figure wasn't available"
        head = f"{self.provider}: recorded ${self.recorded:.2f}, the provider says ${self.provider_says:.2f}"
        if not self.off:
            return head + " (agrees)"
        word = "more" if (self.difference or 0) > 0 else "less"
        return (f"{head}: ${abs(self.difference or 0):.2f} {word} than recorded. Anything else using the same "
                "key counts here too; if nothing else did, the cost accounting needs a look")


def reconcile(before: Snapshot, after: Snapshot, recorded: dict[str, float]) -> list[Reconciliation]:
    """Compare the job's recorded spend per provider with the providers' own figures."""
    names = [p for p in PROVIDERS if p in recorded or p in before or p in after]
    return [Reconciliation(p, recorded.get(p, 0.0), provider_spent(before.get(p), after.get(p))) for p in names]


def take(probe: Callable[[], Snapshot] | None = None) -> Snapshot:
    """A snapshot from `probe` (the real providers by default); a failing probe gives an empty one."""
    try:
        return (probe or probe_providers)()
    except Exception as exc:
        logger.warning("reconcile: provider snapshot failed: %s", exc)
        return {}
