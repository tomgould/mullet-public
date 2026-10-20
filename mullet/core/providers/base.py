"""Base provider interface for all LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from mullet.core.agent.types import TokenUsage


@dataclass
class LLMResponse:
    """Standardized response from any provider."""
    content: str = ""
    tool_calls: list[dict] | None = None
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    provider: str = ""
    reasoning_content: str | None = None  # Kimi K2 thinking output — must be replayed in multi-turn


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    content: str = ""
    tool_call_delta: dict | None = None
    finish_reason: str | None = None
    reasoning_content: str | None = None
    usage: dict | None = None


class Provider(ABC):
    """Base interface for all LLM providers.

    All providers speak OpenAI message format internally.
    Provider adapters handle format conversion (e.g., Anthropic).
    """

    name: str = ""

    @abstractmethod
    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        """Synchronous completion."""
        ...

    def create_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Iterator[StreamChunk]:
        """Streaming completion. Default: fall back to non-streaming."""
        response = self.create(messages, tools, max_tokens, temperature, stream=False)
        yield StreamChunk(content=response.content, finish_reason=response.finish_reason)

    def health_check(self) -> bool:
        """Check if provider is reachable. Default: assume healthy."""
        return True

    @property
    def supports_tool_calling(self) -> bool:
        """Whether this provider/model supports function calling."""
        return True

    def describe(self) -> str:
        """Return a canonical 'provider/model' label from the live provider instance.

        Always reads self.model — the attribute set at instantiation time, which
        is the actual model that will be sent on the wire. Use this for HUD labels
        and logs to avoid the resolver-time vs wire-time desync bug.
        """
        model = getattr(self, "model", "")
        return f"{self.name}/{model}" if model else self.name

    def parse_error(self, exception: Exception) -> tuple[str, str] | None:
        """Return (param_to_omit, reason) if *exception* is a known constraint
        violation for this provider, else None.

        Default: defer to the module-level parse_constraint_error, dispatched by
        self.name. Subclasses can override for provider-specific logic without
        editing the shared parser registry.
        """
        from mullet.core.providers.constraints import parse_constraint_error
        return parse_constraint_error(self.name, exception)
