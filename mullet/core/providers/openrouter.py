"""OpenRouter provider."""

import logging
import os
import requests
from typing import Iterator

from openai.types.chat import ChatCompletion

from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, StreamChunk
from mullet.core.providers.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(OpenAICompatibleProvider):
    """Provider for OpenRouter.ai."""

    def __init__(self, model: str = "auto", api_key: str | None = None):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or pass it to the constructor.")

        self._extra_headers = {
            "HTTP-Referer": "https://github.com/tomgould/mullet-public",  # Optional: for OpenRouter attribution
            "X-Title": "Mullet",  # Optional: for OpenRouter attribution
        }

        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model,
            default_headers=self._extra_headers,
        )
        self.name = "openrouter"

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to OpenRouter."""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "extra_headers": self._extra_headers,
        }
        if tools and self.supports_tool_calling:
            kwargs["tools"] = tools
        if reasoning_config:
            kwargs["reasoning"] = reasoning_config

        # Apply model constraints (strip/rename/clamp) before dispatch
        from mullet.core.providers.constraints import shape_request
        kwargs = shape_request(self.model, kwargs)

        if stream:
            return self._handle_stream(kwargs)

        return self._call_with_retry(kwargs)

    def create_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion from OpenRouter."""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "extra_headers": self._extra_headers,
        }
        if tools and self.supports_tool_calling:
            kwargs["tools"] = tools

        # Apply model constraints before dispatch
        from mullet.core.providers.constraints import shape_request, get_constraint_registry
        kwargs = shape_request(self.model, kwargs)

        # Constraint self-healing on stream open (parent class pattern)
        try:
            stream = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            fix = self.parse_error(e)
            if fix is None:
                raise
            omit_param, reason = fix
            get_constraint_registry().learn(self.model, omit_param, reason)
            kwargs = shape_request(self.model, kwargs)
            stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            tool_call_delta = None
            if delta.tool_calls:
                tc = delta.tool_calls[0]
                tool_call_delta = {
                    "index": tc.index,
                    "id": getattr(tc, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(tc.function, "name", None),
                        "arguments": getattr(tc.function, "arguments", ""),
                    },
                }
            yield StreamChunk(
                content=delta.content or "",
                tool_call_delta=tool_call_delta,
                finish_reason=chunk.choices[0].finish_reason,
            )

    def list_models(self) -> list[str]:
        """List models available from OpenRouter."""
        try:
            headers = {
                "Authorization": f"Bearer {self.client.api_key}",
                "HTTP-Referer": "https://github.com/tomgould/mullet-public",
                "X-Title": "Mullet",
            }
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
            response.raise_for_status()
            models_data = response.json()
            return [m["id"] for m in models_data["data"]]
        except Exception as e:
            logger.error("Failed to list OpenRouter models: %s", e)
            return []

    def _to_response(self, response: ChatCompletion) -> LLMResponse:
        """Convert OpenAI ChatCompletion to LLMResponse."""
        if not response.choices:
            logger.warning("Provider %s returned empty choices", self.name)
            return LLMResponse()

        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            ) if response.usage else TokenUsage(),
            model=self.model,
            provider=self.name,
        )
