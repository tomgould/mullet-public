"""Compact HUD statusline for shell prompts or tmux status bars."""
from pathlib import Path
from typing import Optional

from mullet.core.paths import state_dir
from mullet.hud.guard import BudgetStatus, check_budget


def _active_provider_from_log(log_path: Path) -> Optional[str]:
    """Read the last line of *log_path* and return the provider name.

    Expected line format:
        2026-04-28 22:03:15  provider/model  TIER  reasons  |  task
    Returns ``None`` when the file is missing, empty, or unparseable.
    """
    try:
        with open(log_path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except (OSError, ValueError):
        return None

    if not lines:
        return None

    last = lines[-1].strip()
    if not last:
        return None

    # Split on double-space first, then fall back to whitespace
    parts = [p for p in last.split("  ") if p]
    if len(parts) < 2:
        parts = last.split()
        if len(parts) < 2:
            return None

    provider_model = parts[1]
    if "/" not in provider_model:
        return None

    return provider_model.split("/", 1)[0]


def render_statusline() -> str:
    """Return a compact statusline string.

    Format: ``●GO provider-name`` (or ``⚠THROTTLE …``, ``✗PAUSE …``, ``●?``).
    """
    try:
        budget = check_budget()
    except Exception:
        return "●?"

    if budget.status == BudgetStatus.GO:
        badge = "●GO"
    elif budget.status == BudgetStatus.THROTTLE:
        badge = "⚠THROTTLE"
    elif budget.status == BudgetStatus.PAUSE:
        badge = "✗PAUSE"
    else:
        badge = "●?"

    provider: Optional[str] = None
    try:
        log_path = state_dir() / "routing.log"
        provider = _active_provider_from_log(log_path)
    except Exception:
        pass

    if provider:
        return f"{badge} {provider}"
    return badge


def main() -> None:
    """Print the statusline to stdout."""
    print(render_statusline())


if __name__ == "__main__":
    main()
