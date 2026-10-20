"""Output formatting for agent results.

Formats agent responses for CLI, API, and WebSocket display.
Tracks tool calls, timing, and cost for verbose output.
"""

import time
from dataclasses import dataclass, field
from mullet.core.agent.types import AgentResult, TokenUsage


@dataclass
class ExecutionSummary:
    """Summary of an agent execution for display."""
    task: str
    provider: str = ""
    model: str = ""
    tier: str = ""
    tool_calls: list[str] = field(default_factory=list)
    duration_seconds: float = 0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    iterations: int = 0
    completed: bool = False
    cost_estimate: float = 0

    def format_verbose(self) -> str:
        """Format as verbose CLI output."""
        lines = [
            f"Provider: {self.provider}/{self.model}",
            f"Tier: {self.tier}",
            f"Iterations: {self.iterations}",
            f"Tool calls: {len(self.tool_calls)}",
        ]
        if self.tool_calls:
            lines.append(f"  Tools used: {', '.join(set(self.tool_calls))}")
        lines.extend([
            f"Tokens: {self.tokens.total_tokens:,}",
            f"Duration: {self.duration_seconds:.1f}s",
        ])
        if self.cost_estimate > 0:
            lines.append(f"Est. cost: ${self.cost_estimate:.4f}")
        return "\n".join(lines)

    def format_json(self) -> dict:
        """Format as JSON for API responses."""
        return {
            "task": self.task,
            "completed": self.completed,
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier,
            "tool_calls": self.tool_calls,
            "unique_tools": list(set(self.tool_calls)),
            "duration_seconds": round(self.duration_seconds, 2),
            "tokens": {
                "input": self.tokens.input_tokens,
                "output": self.tokens.output_tokens,
                "total": self.tokens.total_tokens,
            },
            "iterations": self.iterations,
            "cost_estimate": round(self.cost_estimate, 6),
        }


class ExecutionTracker:
    """Track tool calls and timing during agent execution."""

    def __init__(self, task: str):
        self.task = task
        self._tool_calls: list[str] = []
        self._start_time = time.time()

    def on_tool_call(self, name: str, preview: str):
        """Called when a tool is invoked."""
        self._tool_calls.append(name)

    def build_summary(self, result: AgentResult) -> ExecutionSummary:
        """Build summary from agent result."""
        duration = time.time() - self._start_time
        return ExecutionSummary(
            task=self.task,
            provider=result.provider_used,
            tier=result.tier_used,
            tool_calls=self._tool_calls,
            duration_seconds=duration,
            tokens=result.tokens_used,
            iterations=result.iterations_used,
            completed=result.completed,
        )
