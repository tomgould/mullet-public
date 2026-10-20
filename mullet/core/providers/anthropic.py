"""Native Anthropic provider with prompt caching support.

Benefits over OpenAI-compatible wrapper:
- Prompt caching (significant cost savings on long conversations)
- Extended thinking support
- Native tool_use format
"""

import logging
import time as _time
from typing import Iterator

from mullet.core.agent.types import TokenUsage
from mullet.core.providers.constraints import shape_request, get_constraint_registry, shape_request as _reshape
from mullet.core.providers.openai_compat import _hud_agent_track

logger = logging.getLogger(__name__)
from mullet.core.providers.base import LLMResponse, Provider, StreamChunk


class AnthropicProvider(Provider):
    """Native Anthropic provider using the anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
    ):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key, timeout=120.0)
        self.model = model
        self.name = "anthropic"

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the Anthropic API."""
        # Apply prompt caching before conversion (reduces costs ~75%)
        from mullet.core.providers.prompt_cache import apply_anthropic_cache_control
        cached_messages = apply_anthropic_cache_control(messages, native_anthropic=True)
        system, anthropic_messages = self._convert_messages(cached_messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        kwargs: dict = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if temperature != 1.0:
            kwargs["temperature"] = temperature
        if reasoning_config:
            kwargs["thinking"] = reasoning_config

        # Apply model constraints (strip/rename/clamp) before dispatch
        kwargs = shape_request(self.model, kwargs)

        with _hud_agent_track(self.name, self.model, messages):
            return self._call_with_retry(kwargs)

    def _call_with_retry(self, kwargs: dict, max_retries: int = 3) -> "LLMResponse":
        """Call with exponential backoff on 429/5xx errors + constraint self-healing."""
        _logger = logger
        # D4 dispatch-time safety net: abort before HTTP if composed payload
        # would overflow the model's context. Soft-fails on estimator /
        # registry errors — see mullet.core.providers.context_gate.
        from mullet.core.providers.context_gate import assert_context_fits
        assert_context_fits(self.name, self.model, kwargs.get("messages", []))
        # Ground-truth wire log — records the ACTUAL model being sent on the wire.
        _logger.info("wire: provider=%s model=%s msg_count=%d",
                     self.name, self.model, len(kwargs.get("messages", [])))
        last_error = None
        for attempt in range(max_retries):
            try:
                _t0 = _time.monotonic()
                response = self.client.messages.create(**kwargs)
                llm_response = self._to_response(response)
                _ms = int((_time.monotonic() - _t0) * 1000)
                try:
                    from mullet.core.logging.interaction_log import log_interaction
                    log_interaction(
                        provider=self.name,
                        model=self.model,
                        messages=kwargs.get("messages", []),
                        response_content=llm_response.content,
                        prompt_tokens=llm_response.usage.input_tokens if llm_response.usage else 0,
                        completion_tokens=llm_response.usage.output_tokens if llm_response.usage else 0,
                        latency_ms=_ms,
                        success=True,
                    )
                except Exception:
                    pass
                return llm_response
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                if "429" in err_msg or "rate" in err_msg or "500" in err_msg or "overloaded" in err_msg:
                    wait = (2 ** attempt) + (attempt * 0.5)
                    _logger.warning("Anthropic: %s (retry %d/%d in %.1fs)",
                                    str(e)[:80], attempt + 1, max_retries, wait)
                    _time.sleep(wait)
                else:
                    # Constraint self-healing: detect novel parameter rejections,
                    # learn them, reshape the request, retry exactly once.
                    fix = self.parse_error(e)
                    if fix is not None:
                        omit_param, reason = fix
                        get_constraint_registry().learn(self.model, omit_param, reason)
                        kwargs = _reshape(self.model, kwargs)
                        _logger.warning(
                            "Anthropic: constraint error — learned omit=%r, reshaping and retrying once",
                            omit_param,
                        )
                        try:
                            response = self.client.messages.create(**kwargs)
                            return self._to_response(response)
                        except Exception as e2:
                            e = e2  # fall through to raise
                    try:
                        from mullet.core.logging.interaction_log import log_interaction
                        log_interaction(
                            provider=self.name,
                            model=self.model,
                            messages=kwargs.get("messages", []),
                            response_content=None,
                            prompt_tokens=0,
                            completion_tokens=0,
                            latency_ms=0,
                            success=False,
                            error=str(e)[:200],
                        )
                    except Exception:
                        pass
                    raise
        try:
            from mullet.core.logging.interaction_log import log_interaction
            log_interaction(
                provider=self.name, model=self.model,
                messages=kwargs.get("messages", []),
                response_content=None, prompt_tokens=0, completion_tokens=0,
                latency_ms=0, success=False, error=str(last_error)[:200],
            )
        except Exception:
            pass
        raise last_error

    def create_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion from the Anthropic API."""
        system, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        kwargs: dict = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if temperature != 1.0:
            kwargs["temperature"] = temperature

        # Apply model constraints before dispatch
        kwargs = shape_request(self.model, kwargs)
        logger.info("wire: provider=%s model=%s msg_count=%d stream=True",
                    self.name, self.model, len(kwargs.get("messages", [])))

        # Constraint self-healing on stream open (pre-first-chunk errors only)
        def _open_stream(k):
            return self.client.messages.stream(**k)

        try:
            _stream_cm = _open_stream(kwargs)
        except Exception as e:
            fix = self.parse_error(e)
            if fix is None:
                raise
            omit_param, reason = fix
            get_constraint_registry().learn(self.model, omit_param, reason)
            kwargs = _reshape(self.model, kwargs)
            logger.warning(
                "Anthropic: stream constraint error — learned omit=%r, reshaping and retrying once",
                omit_param,
            )
            _stream_cm = _open_stream(kwargs)

        # Use the full event stream to capture both text and tool_use blocks
        with _hud_agent_track(self.name, self.model, messages), _stream_cm as stream:
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_delta':
                        delta = event.delta
                        if hasattr(delta, 'text'):
                            yield StreamChunk(content=delta.text)
                        elif hasattr(delta, 'partial_json'):
                            # tool_use input delta — accumulate
                            yield StreamChunk(tool_call_delta={
                                "partial_json": delta.partial_json,
                            })
                    elif event.type == 'content_block_start':
                        block = event.content_block
                        if hasattr(block, 'type') and block.type == 'tool_use':
                            yield StreamChunk(tool_call_delta={
                                "id": block.id,
                                "type": "function",
                                "function": {"name": block.name, "arguments": ""},
                            })
                    elif event.type == 'message_delta':
                        if hasattr(event, 'delta') and hasattr(event.delta, 'stop_reason'):
                            yield StreamChunk(finish_reason=event.delta.stop_reason or "stop")

    def health_check(self) -> bool:
        """Verify connectivity to the Anthropic API."""
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            logger.debug("Anthropic health check failed: %s", e)
            return False

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool calling."""
        return True

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert OpenAI format messages to Anthropic format.

        Handles the critical difference: Anthropic requires strict user/assistant
        alternation. Multiple consecutive tool results must be batched into a
        single user message with a list of tool_result content blocks.
        """
        import json as _json

        system = ""
        anthropic_messages: list[dict] = []

        # Pending tool_result blocks to batch into one user message
        pending_tool_results: list[dict] = []

        def _flush_tool_results():
            if pending_tool_results:
                anthropic_messages.append({
                    "role": "user",
                    "content": list(pending_tool_results),
                })
                pending_tool_results.clear()

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                system = msg.get("content", "")

            elif role == "user":
                _flush_tool_results()
                anthropic_messages.append({"role": "user", "content": msg["content"]})

            elif role == "assistant":
                _flush_tool_results()
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        args = tc["function"]["arguments"]
                        if isinstance(args, str):
                            try:
                                args = _json.loads(args)
                            except _json.JSONDecodeError:
                                logger.debug("Malformed tool args JSON, using empty: %s", args[:80])
                                args = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": args,
                        })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content_blocks if content_blocks else (msg.get("content") or ""),
                })

            elif role == "tool":
                # Batch consecutive tool results into one user message
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                })

        _flush_tool_results()
        return system, anthropic_messages

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-format tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    def _to_response(self, response) -> LLMResponse:
        """Convert Anthropic response to standard LLMResponse."""
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                import json
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        finish_reason = "stop"
        if response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        if hasattr(response.usage, "cache_read_input_tokens"):
            usage.cache_read_tokens = response.usage.cache_read_input_tokens or 0
        if hasattr(response.usage, "cache_creation_input_tokens"):
            usage.cache_write_tokens = response.usage.cache_creation_input_tokens or 0

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
            model=self.model,
            provider=self.name,
        )
