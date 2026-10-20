"""Claude CLI adapter — use the claude CLI as a Mullet provider.

Wraps the `claude` CLI (which uses Max subscription OAuth)
so Mullet can delegate to Sonnet/Opus/Haiku without an API key.

Requires: claude CLI installed and authenticated (Max subscription)
"""

import logging
import subprocess
from typing import Iterator

from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider, StreamChunk

logger = logging.getLogger(__name__)


def is_claude_cli_available() -> bool:
    """Check if claude CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class ClaudeCLIProvider(Provider):
    """Provider that wraps the claude CLI command.

    Uses Max subscription OAuth — no API key needed.
    Calls `claude -p --model <model>` via subprocess.
    """

    def __init__(self, model: str = "sonnet"):
        self.model = model
        self.name = "claude-cli"

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> LLMResponse:
        """Run a prompt through the claude CLI and return the response."""
        prompt = self._messages_to_prompt(messages)

        import time as _time
        _t0 = _time.monotonic()
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", self.model],
                input=prompt,
                capture_output=True, text=True,
                timeout=300,  # 5 min timeout for long tasks
            )
            _ms = int((_time.monotonic() - _t0) * 1000)

            if result.returncode != 0:
                error_msg = result.stderr.strip() or f"claude CLI exited with code {result.returncode}"
                logger.debug("Claude CLI error: %s", error_msg)
                try:
                    from mullet.core.logging.interaction_log import log_interaction
                    log_interaction(
                        provider=self.name, model=self.model, messages=messages,
                        response_content=None, prompt_tokens=0, completion_tokens=0,
                        latency_ms=_ms, success=False, error=error_msg[:200],
                    )
                except Exception:
                    pass
                return LLMResponse(
                    content=f"Error: {error_msg}",
                    usage=TokenUsage(total_tokens=0),
                )

            content = result.stdout.strip()
            token_estimate = len(content.split()) * 2
            try:
                from mullet.core.logging.interaction_log import log_interaction
                log_interaction(
                    provider=self.name, model=self.model, messages=messages,
                    response_content=content, prompt_tokens=0,
                    completion_tokens=token_estimate,
                    latency_ms=_ms, success=True,
                )
            except Exception:
                pass
            return LLMResponse(
                content=content,
                usage=TokenUsage(total_tokens=token_estimate),
            )

        except subprocess.TimeoutExpired:
            _ms = int((_time.monotonic() - _t0) * 1000)
            logger.warning("Claude CLI timed out after 300s")
            try:
                from mullet.core.logging.interaction_log import log_interaction
                log_interaction(
                    provider=self.name, model=self.model, messages=messages,
                    response_content=None, prompt_tokens=0, completion_tokens=0,
                    latency_ms=_ms, success=False, error="timeout after 300s",
                )
            except Exception as e:
                logger.warning("claude_cli: Failed to log interaction for timeout: %s", e)
            return LLMResponse(
                content="Error: claude CLI timed out",
                usage=TokenUsage(total_tokens=0),
            )

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Iterator[StreamChunk]:
        """Yield the claude CLI response as a single stream chunk."""
        # No streaming support — fall back to blocking call
        response = self.create(messages, tools, max_tokens, temperature)
        yield StreamChunk(content=response.content, finish_reason="stop")

    @staticmethod
    def _messages_to_prompt(messages: list[dict]) -> str:
        """Convert message list to a single prompt string for the CLI."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"Previous response: {content}")
        return "\n\n".join(parts)
