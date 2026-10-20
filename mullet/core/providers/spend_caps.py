"""User-configurable per-provider spend caps from .mullet/budget.yaml.

Example .mullet/budget.yaml:
    openrouter:
      cap_usd: 10.0
    kimi:
      cap_usd: 20.0
    monthly_total_cap_usd: 50.0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BUDGET_FILE_NAME = "budget.yaml"
BUDGET_JSON_NAME = "budget.json"


@dataclass
class SpendCap:
    """Spend cap for a single provider."""
    provider: str
    cap_usd: float          # USD cap for this provider
    current_spend_usd: float = 0.0  # tracked spend (for session)

    @property
    def remaining_cap_usd(self) -> float:
        """Return the remaining budget in USD."""
        return max(0.0, self.cap_usd - self.current_spend_usd)

    @property
    def is_within_cap(self) -> bool:
        """Return whether current spend is below the cap."""
        return self.current_spend_usd < self.cap_usd


@dataclass
class BudgetConfig:
    """Collection of per-provider spend caps and optional monthly total."""
    caps: dict[str, SpendCap]         # provider_key -> SpendCap
    monthly_total_cap_usd: float | None = None


def _find_budget_file(cwd: Path | None = None) -> Path | None:
    """Walk up from cwd looking for .mullet/budget.yaml (or budget.json)."""
    start = cwd or Path.cwd()
    # Ensure we start from a directory
    if not start.is_dir():
        start = start.parent
    for parent in [start, *start.parents]:
        mullet_dir = parent / ".mullet"
        if mullet_dir.is_dir():
            yaml_path = mullet_dir / BUDGET_FILE_NAME
            if yaml_path.exists():
                return yaml_path
            json_path = mullet_dir / BUDGET_JSON_NAME
            if json_path.exists():
                return json_path
    return None


def _parse_budget_data(raw: dict[str, Any]) -> BudgetConfig:
    """Parse raw dict into BudgetConfig."""
    caps: dict[str, SpendCap] = {}
    monthly_total_cap_usd: float | None = None

    for key, value in raw.items():
        if key == "monthly_total_cap_usd":
            try:
                monthly_total_cap_usd = float(value)
            except (TypeError, ValueError):
                logger.warning("Invalid monthly_total_cap_usd value: %r", value)
            continue

        if isinstance(value, dict):
            cap_val = value.get("cap_usd")
            if cap_val is not None:
                try:
                    cap_f = float(cap_val)
                    caps[key] = SpendCap(provider=key, cap_usd=cap_f)
                except (TypeError, ValueError):
                    logger.warning("Invalid cap_usd for provider %s: %r", key, cap_val)

    return BudgetConfig(caps=caps, monthly_total_cap_usd=monthly_total_cap_usd)


def load_budget_config(cwd: Path | None = None) -> BudgetConfig:
    """Load .mullet/budget.yaml from cwd or parent dirs. Returns empty config if not found."""
    try:
        file_path = _find_budget_file(cwd)
        if file_path is None:
            return BudgetConfig(caps={})

        text = file_path.read_text(encoding="utf-8")

        if file_path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
                data: Any = yaml.safe_load(text) or {}
            except Exception:
                logger.debug("YAML parsing failed for %s", file_path)
                return BudgetConfig(caps={})
        else:
            try:
                import json
                data = json.loads(text)
            except Exception:
                logger.debug("JSON parsing failed for %s", file_path)
                return BudgetConfig(caps={})

        if not isinstance(data, dict):
            return BudgetConfig(caps={})

        return _parse_budget_data(data)
    except Exception:
        logger.debug("Failed to load budget config", exc_info=True)
        return BudgetConfig(caps={})


def check_spend_cap(
    provider_key: str,
    balance_usd: float | None,
    config: BudgetConfig | None = None,
) -> tuple[bool, str]:
    """Check if provider is within spend cap.

    Returns (is_allowed: bool, reason: str).
    Uses live balance from budget_probe to estimate remaining vs cap.
    ``balance_usd`` should be the *spent* amount (e.g. ``credits_used_usd``
    from ``ProviderBalance``). Pass ``None`` when the spend is unknown.
    """
    try:
        cfg = config or load_budget_config()
    except Exception:
        return True, "config load failed — allowing"

    cap = cfg.caps.get(provider_key)
    if cap is None:
        return True, "no cap set"

    if balance_usd is None:
        return True, "balance unknown — allowing"

    spent = float(balance_usd)
    if spent >= cap.cap_usd:
        return False, f"cap exceeded: spent ${spent:.2f} of ${cap.cap_usd:.2f} cap"

    return True, f"within cap: spent ${spent:.2f} of ${cap.cap_usd:.2f} cap"
