"""ClaudeMaxProvider — DISABLED. Extracting and reusing Claude Max/Pro/Free
subscription OAuth tokens in a third-party tool violates Anthropic's Consumer
Terms of Service: "Using OAuth tokens obtained through Claude Free, Pro, or
Max accounts in any other product, tool, or service -- including the Agent
SDK -- is not permitted." Anthropic began actively enforcing this in January
2026 (account bans for "token arbitrage"; see
https://www.theregister.com/2026/02/20/anthropic_clarifies_ban_third_party_claude_access/).

This module previously read the OAuth token straight out of
~/.claude/.credentials.json (the same file used by the Claude Code CLI) and
fed it directly to the raw `anthropic` SDK client from inside Mullet -- a
third-party tool, not Claude Code itself. That is exactly the prohibited
pattern. It is now permanently disabled: `_read_oauth_token()` always
returns None, so `ClaudeMaxProvider.__init__` always raises RuntimeError,
which every existing call site already treats as "provider not configured"
and falls through to the next candidate. Use a real ANTHROPIC_API_KEY
(governed by separate, permitted terms) if you want Mullet to route to
Anthropic models directly.
"""

import json
import logging
import threading
import time as _time
from pathlib import Path
from typing import Iterator, Optional

from mullet.core.providers.base import LLMResponse, Provider, StreamChunk
from mullet.core.providers.constraints import shape_request, parse_constraint_error, get_constraint_registry
from mullet.core.providers.openai_compat import _hud_agent_track
from mullet.core.agent.types import TokenUsage

logger = logging.getLogger(__name__)

# OAuth beta header required for bearer-token auth on the Anthropic API
_OAUTH_BETA = "oauth-2025-04-20"

# Claude Max's OAuth endpoint caps concurrent requests per-token at roughly
# 1 in-flight per model — parallel calls on the same token consistently
# return 429 rate_limit_error even when the plan's quota is barely used.
# Serialise all ClaudeMax invocations inside this process so we never
# collide with ourselves. (Won't protect against an outside Claude Code
# session competing for the same token — that's inherent to OAuth.)
_CLAUDE_MAX_LOCK = threading.Lock()


def _read_oauth_token() -> Optional[str]:
    """Permanently disabled -- see module docstring. Always returns None so
    ClaudeMaxProvider construction always fails closed as "not configured".
    """
    return None


class ClaudeMaxProvider(Provider):
    """Anthropic /v1/messages via Claude Max subscription OAuth bearer token.

    Uses the stock ``anthropic`` SDK with ``auth_token`` so the request
    path is identical to ``AnthropicProvider`` — same retry logic, same
    constraint self-healing, same HUD tracking. The only difference is
    the authentication credential: OAuth bearer instead of API key.

    Args:
        model: Anthropic model ID to use.
        access_token: Bearer token override. When ``None`` the token is
            read from ``~/.claude/.credentials.json``.

    Raises:
        RuntimeError: When no token is available from either source.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        access_token: Optional[str] = None,
    ):
        token = access_token or _read_oauth_token()
        if not token:
            raise RuntimeError(
                "ClaudeMaxProvider: no OAuth token available. "
                "Expected either an explicit access_token= or a valid "
                "~/.claude/.credentials.json with claudeAiOauth.accessToken. "
                "Sign in via the Claude Code CLI to populate credentials."
            )
        from anthropic import Anthropic
        # The anthropic SDK accepts auth_token= for bearer-token auth.
        # The oauth beta header is added via default_headers.
        self.client = Anthropic(
            auth_token=token,
            default_headers={"anthropic-beta": _OAUTH_BETA},
            timeout=120.0,
        )
        self.model = model
        self.name = "claude_max"
        self._token = token  # kept for health_check only

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        """Send a chat completion request via Claude Max OAuth."""
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
        if reasoning_config:
            kwargs["thinking"] = reasoning_config

        kwargs = shape_request(self.model, kwargs)

        with _hud_agent_track(self.name, self.model, messages), _CLAUDE_MAX_LOCK:
            return self._call_with_retry(kwargs)

    def create_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion via Claude Max OAuth."""
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

        kwargs = shape_request(self.model, kwargs)
        logger.info("wire: provider=%s model=%s msg_count=%d stream=True",
                    self.name, self.model, len(kwargs.get("messages", [])))

        with _hud_agent_track(self.name, self.model, messages), _CLAUDE_MAX_LOCK, \
                self.client.messages.stream(**kwargs) as stream_cm:
            for event in stream_cm:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text"):
                            yield StreamChunk(content=delta.text)
                        elif hasattr(delta, "partial_json"):
                            yield StreamChunk(tool_call_delta={"partial_json": delta.partial_json})
                    elif event.type == "content_block_start":
                        block = event.content_block
                        if hasattr(block, "type") and block.type == "tool_use":
                            yield StreamChunk(tool_call_delta={
                                "id": block.id,
                                "type": "function",
                                "function": {"name": block.name, "arguments": ""},
                            })
                    elif event.type == "message_delta":
                        if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                            yield StreamChunk(finish_reason=event.delta.stop_reason or "stop")

    def health_check(self) -> bool:
        """Verify connectivity to the Claude Max API."""
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            logger.debug("ClaudeMax health check failed: %s", e)
            return False

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool calling."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, kwargs: dict, max_retries: int = 3) -> LLMResponse:
        """Wire call with exponential backoff on 429/5xx + constraint self-healing."""
        from mullet.core.providers.context_gate import assert_context_fits
        assert_context_fits(self.name, self.model, kwargs.get("messages", []))
        logger.info("wire: provider=%s model=%s msg_count=%d",
                    self.name, self.model, len(kwargs.get("messages", [])))

        last_error: Exception = RuntimeError("all retries exhausted")
        for attempt in range(max_retries):
            try:
                t0 = _time.monotonic()
                response = self.client.messages.create(**kwargs)
                llm_response = self._to_response(response)
                ms = int((_time.monotonic() - t0) * 1000)
                self._log(kwargs, llm_response, ms, success=True)
                return llm_response
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                # Check for auth errors — don't retry, raise immediately with clear message
                if "401" in err_msg or "403" in err_msg or "insufficient_permissions" in err_msg or "authentication" in err_msg:
                    raise RuntimeError(
                        f"ClaudeMaxProvider: authentication failed ({e}). "
                        "The OAuth token may be expired or the Max plan may not "
                        "grant access to model '{self.model}'. "
                        "Re-authenticate via the Claude Code CLI."
                    ) from e
                if "429" in err_msg or "rate" in err_msg or "500" in err_msg or "overloaded" in err_msg:
                    wait = (2 ** attempt) + (attempt * 0.5)
                    logger.warning("ClaudeMax: %s (retry %d/%d in %.1fs)",
                                   str(e)[:80], attempt + 1, max_retries, wait)
                    _time.sleep(wait)
                else:
                    # Constraint self-healing
                    fix = parse_constraint_error(self.name, e)
                    if fix is not None:
                        omit_param, reason = fix
                        get_constraint_registry().learn(self.model, omit_param, reason)
                        kwargs = shape_request(self.model, kwargs)
                        logger.warning(
                            "ClaudeMax: constraint error — learned omit=%r, reshaping and retrying once",
                            omit_param,
                        )
                        try:
                            response = self.client.messages.create(**kwargs)
                            return self._to_response(response)
                        except Exception as e2:
                            e = e2
                    self._log(kwargs, None, 0, success=False, error=str(e)[:200])
                    raise e

        self._log(kwargs, None, 0, success=False, error=str(last_error)[:200])
        raise last_error

    def _log(
        self,
        kwargs: dict,
        response: Optional[LLMResponse],
        latency_ms: int,
        *,
        success: bool,
        error: str = "",
    ) -> None:
        try:
            from mullet.core.logging.interaction_log import log_interaction
            log_interaction(
                provider="claude_max",
                model=self.model,
                messages=kwargs.get("messages", []),
                response_content=response.content if response else None,
                prompt_tokens=response.usage.input_tokens if response and response.usage else 0,
                completion_tokens=response.usage.output_tokens if response and response.usage else 0,
                latency_ms=latency_ms,
                success=success,
                error=error,
            )
        except Exception:
            pass

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert OpenAI-format messages to Anthropic format (copied from AnthropicProvider)."""
        import json as _json

        system = ""
        anthropic_messages: list[dict] = []
        pending_tool_results: list[dict] = []

        def _flush():
            if pending_tool_results:
                anthropic_messages.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system = msg.get("content", "")
            elif role == "user":
                _flush()
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                _flush()
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
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                })

        _flush()
        return system, anthropic_messages

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-format tools to Anthropic format (copied from AnthropicProvider)."""
        result = []
        for tool in tools:
            func = tool.get("function", {})
            result.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    def _to_response(self, response) -> LLMResponse:
        """Convert Anthropic SDK response to standard LLMResponse."""
        import json
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
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
