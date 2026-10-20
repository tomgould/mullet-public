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

def _attribution_headers() -> dict[str, str]:
    """OpenRouter's optional app-attribution headers.

    X-Title names the app. HTTP-Referer (the app's URL) is sent only when
    MULLET_OPENROUTER_REFERER is set.
    """
    from mullet.core.env import openrouter_referer

    headers = {}
    referer = openrouter_referer()
    if referer:
        headers["HTTP-Referer"] = referer
    headers["X-Title"] = "Mullet"
    return headers


_VALUE_TIERS = ("LOW", "STANDARD", "ADVANCED", "CRITICAL")
_TIER_ALIASES = {"PREMIUM": "CRITICAL", "MAXIMUM": "CRITICAL"}


def next_openrouter_model(
    failed: str,
    exclude: set[str] | frozenset[str] = frozenset(),
    tier: str | None = None,
) -> str | None:
    """The model to try after OpenRouter refused ``failed`` as unavailable.

    A withdrawn `:free` model falls back to the other free candidates first,
    then to the paid value-picker ladder; a paid model goes straight to the
    paid ladder. Within a list the value picker chooses (cheapest model that
    clears the tier's quality floor), skipping ``exclude`` and every model
    already recorded as unavailable. ``tier`` defaults to LOW for a free
    model (free models are only routed for LOW work) and STANDARD otherwise.
    Returns None when nothing is left.
    """
    from mullet.core.providers.model_availability import unavailable_models
    from mullet.core.providers.value_picker import pick_by_value
    from mullet.core.providers.value_picker_constants import (
        OPENROUTER_FREE_TIER_CANDIDATES,
        OPENROUTER_VALUE_PICK_CANDIDATES,
    )

    skip = set(exclude) | {failed} | unavailable_models("openrouter")
    was_free = failed.endswith(":free")
    tier = _TIER_ALIASES.get((tier or "").upper(), (tier or "").upper())
    if tier not in _VALUE_TIERS:
        tier = "LOW" if was_free else "STANDARD"
    pools = ([OPENROUTER_FREE_TIER_CANDIDATES] if was_free else []) + [OPENROUTER_VALUE_PICK_CANDIDATES]
    for pool in pools:
        remaining = [(m, ctx) for m, ctx in pool if m not in skip]
        if not remaining:
            continue
        try:
            pick = pick_by_value(remaining, tier, 0, "coding", prefer_reasoning=False)
        except Exception as exc:
            logger.warning("next_openrouter_model: value picker failed (%s)", exc)
            pick = None
        if pick:
            return pick
    return None


class OpenRouterProvider(OpenAICompatibleProvider):
    """Provider for OpenRouter.ai.

    When OpenRouter answers that the model can't be served (withdrawn, or no
    longer offered free), the model is recorded as unavailable and the same
    request goes to the next candidate (see next_openrouter_model) instead of
    failing the task. The provider keeps the new model for later calls.
    """

    # The routing tier this provider was picked for, when the caller knows it
    # (the MCP router sets it), so a fallback model suits the same tier.
    routing_tier: str | None = None
    # At most this many replacement models per request.
    MAX_MODEL_FALLBACKS = 3

    def __init__(self, model: str = "auto", api_key: str | None = None):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or pass it to the constructor.")

        self._extra_headers = _attribution_headers()

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
        """Send a chat completion request to OpenRouter.

        Falls back to the next candidate model if OpenRouter says this one is
        unavailable (see the class docstring).
        """
        tried: list[str] = []
        while True:
            try:
                return self._create_once(
                    messages, tools, max_tokens, temperature, stream, reasoning_config,
                )
            except Exception as exc:
                if not self._switch_from_unavailable_model(exc, tried):
                    raise

    def _switch_from_unavailable_model(self, exc: Exception, tried: list[str]) -> bool:
        """Record an unavailable model and move to the next one.

        Returns True when ``self.model`` now names a replacement to try, False
        when ``exc`` is some other error or no candidate is left.
        """
        from mullet.core.providers.model_availability import (
            is_model_unavailable_error,
            mark_unavailable,
            suggested_replacement,
        )

        if not is_model_unavailable_error(exc):
            return False
        failed = self.model
        mark_unavailable(self.name, failed, str(exc))
        tried.append(failed)
        if len(tried) > self.MAX_MODEL_FALLBACKS:
            return False
        replacement = next_openrouter_model(failed, exclude=set(tried), tier=self.routing_tier)
        if replacement is None:
            logger.warning("OpenRouter can't serve %s and no other candidate is left: %s", failed, exc)
            return False
        hint = suggested_replacement(exc)
        logger.warning(
            "OpenRouter can't serve %s (%s); trying %s instead%s",
            failed, str(exc)[:160], replacement,
            f" (OpenRouter suggested {hint})" if hint and hint != replacement else "",
        )
        self.model = replacement
        return True

    def _create_once(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        stream: bool,
        reasoning_config: dict | None,
    ) -> LLMResponse:
        """One chat completion request for ``self.model``."""
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
        """Stream a chat completion from OpenRouter.

        An unavailable model at stream-open time falls back like create().
        """
        tried: list[str] = []
        while True:
            try:
                stream = self._open_stream(messages, tools, max_tokens, temperature)
                break
            except Exception as exc:
                if not self._switch_from_unavailable_model(exc, tried):
                    raise
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

    def _open_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
    ):
        """Open a streaming request for ``self.model``."""
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
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            fix = self.parse_error(e)
            if fix is None:
                raise
            omit_param, reason = fix
            get_constraint_registry().learn(self.model, omit_param, reason)
            kwargs = shape_request(self.model, kwargs)
            return self.client.chat.completions.create(**kwargs)

    def list_models(self) -> list[str]:
        """List models available from OpenRouter."""
        try:
            headers = {
                "Authorization": f"Bearer {self.client.api_key}",
                **_attribution_headers(),
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
