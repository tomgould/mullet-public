"""Budget tracking — token spend per session, per day, with dollar limits.

When a limit is hit, the router downgrades to a cheaper provider/model
rather than failing. This is how Mullet saves money automatically.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default cost rates per 1K tokens (conservative estimates)
DEFAULT_COSTS = {
    "local": {"input": 0.0, "output": 0.0},
    "haiku": {"input": 0.0008, "output": 0.004},
    "flash": {"input": 0.00015, "output": 0.0006},
    "sonnet": {"input": 0.003, "output": 0.015},
    "opus": {"input": 0.015, "output": 0.075},
}


@dataclass
class UsageRecord:
    """A single LLM usage record."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class BudgetLimits:
    """Configurable budget limits."""
    per_task_max_tokens: int = 100_000
    session_max_tokens: int = 1_000_000
    daily_max_dollars: float = 10.0


class BudgetTracker:
    """Track token spend across a session with budget enforcement.

    When limits are approached, suggests cheaper alternatives.
    When limits are exceeded, blocks further spending on expensive tiers.
    """

    def __init__(self, limits: BudgetLimits | None = None, cost_rates: dict | None = None):
        self.limits = limits or BudgetLimits()
        self._cost_rates = cost_rates or DEFAULT_COSTS
        self._records: list[UsageRecord] = []
        self._session_start = time.time()
        self._day_start = _start_of_day()

    def estimate_cost(self, provider: str, model: str, est_input_tokens: int, est_output_tokens: int = 0) -> float:
        """Estimate cost for a task before running it."""
        rates = self._get_rates(model)
        input_cost = (est_input_tokens / 1000) * rates["input"]
        output_cost = ((est_output_tokens or est_input_tokens) / 1000) * rates["output"]
        return input_cost + output_cost

    def record_usage(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        """Record actual token usage after a task."""
        rates = self._get_rates(model)
        cost = (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]

        self._records.append(UsageRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
        ))

        logger.debug("Usage: %s/%s %d+%d tokens, $%.4f (session total: $%.4f)",
                      provider, model, input_tokens, output_tokens, cost, self.session_cost)

    def can_afford(self, provider: str, model: str, est_tokens: int) -> bool:
        """Check if budget allows this provider/model for the estimated tokens."""
        # Check per-task token limit
        if est_tokens > self.limits.per_task_max_tokens:
            return False

        # Check session token limit
        if self.session_tokens + est_tokens > self.limits.session_max_tokens:
            return False

        # Check daily dollar limit
        est_cost = self.estimate_cost(provider, model, est_tokens)
        if self.daily_cost + est_cost > self.limits.daily_max_dollars:
            return False

        return True

    def suggest_downgrade(self, current_model: str) -> str | None:
        """If over budget for current model, suggest a cheaper alternative."""
        if self.daily_cost >= self.limits.daily_max_dollars * 0.9:
            # Near daily limit — suggest cheapest option
            return "local"

        if self.session_tokens >= self.limits.session_max_tokens * 0.8:
            # Near session token limit — suggest cheaper model
            model_lower = current_model.lower()
            if "opus" in model_lower:
                return "sonnet"
            if "sonnet" in model_lower:
                return "haiku"
            if "haiku" in model_lower:
                return "local"

        return None

    @property
    def session_tokens(self) -> int:
        """Total tokens used in this session."""
        return sum(r.input_tokens + r.output_tokens for r in self._records)

    @property
    def session_cost(self) -> float:
        """Total cost in this session."""
        return sum(r.estimated_cost for r in self._records)

    @property
    def daily_cost(self) -> float:
        """Total cost today."""
        day_start = _start_of_day()
        return sum(r.estimated_cost for r in self._records if r.timestamp >= day_start)

    @property
    def daily_tokens(self) -> int:
        """Total tokens today."""
        day_start = _start_of_day()
        return sum(r.input_tokens + r.output_tokens for r in self._records if r.timestamp >= day_start)

    def status(self) -> dict:
        """Get budget status for display/API."""
        return {
            "session_tokens": self.session_tokens,
            "session_cost": round(self.session_cost, 4),
            "daily_tokens": self.daily_tokens,
            "daily_cost": round(self.daily_cost, 4),
            "limits": {
                "per_task_max_tokens": self.limits.per_task_max_tokens,
                "session_max_tokens": self.limits.session_max_tokens,
                "daily_max_dollars": self.limits.daily_max_dollars,
            },
            "remaining": {
                "session_tokens": max(0, self.limits.session_max_tokens - self.session_tokens),
                "daily_dollars": round(max(0, self.limits.daily_max_dollars - self.daily_cost), 4),
            },
        }

    def _get_rates(self, model: str) -> dict:
        """Get cost rates for a model.

        Checks custom rates first, then falls back to model_info database.
        """
        model_lower = model.lower()
        for name, rates in self._cost_rates.items():
            if name in model_lower:
                return rates

        # Fall back to comprehensive model database
        try:
            from mullet.core.providers.model_info import get_pricing
            input_cost, output_cost = get_pricing(model)
            if input_cost > 0 or output_cost > 0:
                return {"input": input_cost, "output": output_cost}
        except ImportError as _exc:
            logger.warning("budget: Model info library not found, using default costs: %s", _exc)
            pass

        # Default: assume moderate cost
        return {"input": 0.001, "output": 0.005}


def _start_of_day() -> float:
    """Get timestamp for start of current UTC day."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp()
