"""Gemini CLI adapter — use the gemini CLI as a Mullet provider.

Wraps the `gemini` CLI command (which uses OAuth authentication)
so Mullet can route tasks to Gemini without needing a GOOGLE_API_KEY.

Requires: npm install -g @google/gemini-cli
Auth: gemini (first run does OAuth flow)
"""

import logging
import subprocess

from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider

logger = logging.getLogger(__name__)


def is_gemini_cli_available() -> bool:
    """Check if gemini CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gemini", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class GeminiCLIProvider(Provider):
    """Provider that wraps the gemini CLI command.

    Uses the CLI's OAuth authentication — no API key needed.
    Calls gemini via subprocess with the prompt on stdin.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.name = "gemini-cli"

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request via the gemini CLI."""
        # Extract the conversation into a single prompt
        prompt = self._messages_to_prompt(messages)

        try:
            # Ensure GOOGLE_CLOUD_PROJECT is passed through (needed for OAuth auth)
            import os
            env = os.environ.copy()

            # Try --prompt first (headless mode)
            result = subprocess.run(
                ["gemini", "--prompt", prompt],
                capture_output=True, text=True,
                timeout=120,
                env=env,
            )

            content = result.stdout.strip()

            # If --prompt fails with permission error, fall back to piped interactive
            if result.returncode != 0 and "Permission denied" in (result.stderr or ""):
                logger.info("Gemini --prompt failed, trying piped interactive mode")
                result = subprocess.run(
                    ["gemini"],
                    input=prompt + "\n/quit\n",
                    capture_output=True, text=True,
                    timeout=120,
                )
                # Extract response from interactive output (skip header lines)
                lines = result.stdout.strip().split("\n")
                # Filter out Gemini CLI chrome (prompts, headers, etc.)
                content_lines = [
                    ln for ln in lines
                    if ln.strip() and not ln.startswith(">") and not ln.startswith("Gemini CLI")
                    and "Signed in" not in ln and "Plan:" not in ln and "/auth" not in ln
                    and "workspace" not in ln.lower() and "branch" not in ln.lower()
                    and "Shift+Tab" not in ln
                ]
                content = "\n".join(content_lines).strip()

            elif result.returncode != 0:
                error = result.stderr.strip()[:200]
                logger.debug("Gemini CLI error: %s", error)
                content = f"Error from Gemini CLI: {error}"

            # Rough token estimation
            prompt_tokens = len(prompt) // 4
            output_tokens = len(content) // 4

            return LLMResponse(
                content=content,
                tool_calls=None,  # CLI doesn't support tool calling
                finish_reason="stop",
                usage=TokenUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=prompt_tokens + output_tokens,
                ),
                model=self.model,
                provider=self.name,
            )

        except subprocess.TimeoutExpired:
            return LLMResponse(
                content="Gemini CLI timed out (120s)",
                tool_calls=None,
                finish_reason="error",
                usage=TokenUsage(),
                model=self.model,
                provider=self.name,
            )
        except FileNotFoundError:
            return LLMResponse(
                content="Gemini CLI not found. Install: npm install -g @google/gemini-cli",
                tool_calls=None,
                finish_reason="error",
                usage=TokenUsage(),
                model=self.model,
                provider=self.name,
            )

    def health_check(self) -> bool:
        """Check if the gemini CLI is available and working."""
        return is_gemini_cli_available()

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool calling."""
        return False  # CLI doesn't support structured tool calls

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert message list to a single prompt string for the CLI."""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                )
            if role == "system":
                parts.append(f"Instructions: {content}")
            elif role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"Previous response: {content}")
        return "\n\n".join(parts)
