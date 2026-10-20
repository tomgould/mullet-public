"""Core types for the agent execution engine."""

from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    """Token usage from an LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class AgentResult:
    """Result of an agent execution."""
    completed: bool
    final_response: str
    messages: list[dict] = field(default_factory=list)
    tool_calls_made: int = 0
    iterations_used: int = 0
    tokens_used: TokenUsage = field(default_factory=TokenUsage)
    provider_used: str = ""
    tier_used: str = ""


@dataclass
class DelegationResult:
    """Result of a delegated subtask."""
    goal: str
    completed: bool
    summary: str
    agent_result: AgentResult | None = None
    error: str | None = None
