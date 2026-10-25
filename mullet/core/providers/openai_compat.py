"""OpenAI-compatible provider covering LM Studio, Ollama, vLLM, OpenRouter, OpenAI."""

import json
import logging
import threading
import urllib.request
import uuid
from contextlib import contextmanager
from openai import OpenAI

logger = logging.getLogger(__name__)


def _brief_task_summary(messages: list[dict]) -> str:
    """Return the last user message's first 80 chars for HUD display."""
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part content — join text parts
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if isinstance(content, str) and content.strip():
                return content.strip()[:80]
    return "LLM call"


@contextmanager
def _hud_agent_track(provider_name: str, model: str, messages: list[dict]):
    """Write the active call to HUD state on enter; remove it on exit.

    Uses :func:`mullet.hud.agent_state.add_agent` / ``remove_agent`` so parallel
    calls to multiple providers (Kimi + local + Moonshot swarms) all appear on
    the HUD concurrently. Any failure to write is swallowed — tracking must
    never break an LLM call.
    """
    agent_id = uuid.uuid4().hex[:12]
    description = _brief_task_summary(messages)
    _tracked = False
    try:
        from mullet.hud.agent_state import add_agent
        add_agent(agent_id, description, provider_name, model)
        _tracked = True
    except Exception as exc:
        logger.debug("hud agent tracker add failed: %s", exc)
    try:
        yield
    except Exception as exc:
        if _tracked:
            try:
                from mullet.hud.agent_state import update_agent_status
                _is_timeout = "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower()
                update_agent_status(agent_id, "timed_out" if _is_timeout else "failed")
            except Exception:
                pass
        raise
    finally:
        try:
            from mullet.hud.agent_state import remove_agent
            remove_agent(agent_id)
        except Exception as exc:
            logger.debug("hud agent tracker remove failed: %s", exc)

from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider, StreamChunk
from typing import Iterator

# Consecutive failure tracking per provider name — used for auto-escalation.
# When a local provider fails _ESCALATION_THRESHOLD times in a row, the router
# should skip it and use a cloud provider instead.
_consecutive_failures: dict[str, int] = {}
_consecutive_failures_lock = threading.Lock()
_ESCALATION_THRESHOLD = 2

# Overall wall-clock budget for the entire retry sequence in _call_with_retry.
# Prevents compounding timeouts from hanging a caller for many minutes.
MAX_RETRY_WALL_SECONDS = 180


def _approx_chars(kwargs: dict) -> int:
    """Rough char count of messages + tools in a kwargs dict."""
    import json
    msgs = json.dumps(kwargs.get("messages", []))
    tools = json.dumps(kwargs.get("tools", []))
    return len(msgs) + len(tools)


def _strip_for_small_context(kwargs: dict) -> dict:
    """Return a copy of kwargs with tools removed and system prompt truncated to ~800 tokens."""
    import copy
    k = copy.deepcopy(kwargs)
    # Step 1: drop tools entirely (~4K tokens saved for typical mullet toolset)
    k.pop("tools", None)
    k.pop("tool_choice", None)
    # Step 2: truncate system prompt to ~3200 chars (~800 tokens) if still too large
    messages = k.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system" and len(msg.get("content", "")) > 3200:
            msg["content"] = msg["content"][:3200] + "\n[truncated for context]"
    return k


def get_consecutive_failures(provider_name: str) -> int:
    """Return consecutive failure count for a provider (0 = healthy)."""
    with _consecutive_failures_lock:
        return _consecutive_failures.get(provider_name, 0)


def reset_consecutive_failures(provider_name: str) -> None:
    """Reset failure count on successful call (called internally)."""
    with _consecutive_failures_lock:
        _consecutive_failures.pop(provider_name, None)


_CLOUD_ESCALATION_TARGETS: frozenset[str] = frozenset({
    "kimi", "gemini", "haiku", "sonnet", "opus", "opus-4-8", "opus-4-7",
    "cheap_cloud", "premium_cloud",
})


def is_escalation_recommended(
    provider_name: str,
    allowed: set[str] | frozenset[str] | None = None,
) -> bool:
    """Return True iff provider is over failure threshold AND escalation is permitted by allowlist."""
    with _consecutive_failures_lock:
        over_threshold = _consecutive_failures.get(provider_name, 0) >= _ESCALATION_THRESHOLD
    if not over_threshold:
        return False
    if allowed is None:
        return True
    return any(name in _CLOUD_ESCALATION_TARGETS for name in allowed)


class OpenAICompatibleProvider(Provider):
    """Provider for any OpenAI-compatible endpoint.

    Covers: LM Studio, Ollama, vLLM, SGLang, OpenRouter, OpenAI, Nous Portal.
    Uses the OpenAI Python SDK client. When api_key is not needed (local),
    use "no-key-required".
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "no-key-required",
        model: str = "default",
        default_headers: dict | None = None,
        openrouter_config: dict | None = None,
    ):
        """Configure an OpenAI-compatible client for *base_url* and *model*."""
        headers = dict(default_headers or {})

        # OpenRouter-specific headers (from Hermes)
        if "openrouter" in base_url.lower() and openrouter_config:
            if openrouter_config.get("app_name"):
                headers["X-Title"] = openrouter_config["app_name"]
            if openrouter_config.get("providers_order"):
                headers["X-OpenRouter-Provider-Order"] = ",".join(openrouter_config["providers_order"])

        self.name = self._infer_name(base_url)
        # Local GPU generation is much slower per-token than cloud APIs — a full
        # file-write tool call from a 14B model can legitimately take >60s. Cloud
        # providers keep the tight 60s timeout since a genuinely stuck connection
        # there should fail fast into fallback.
        _timeout = 300.0 if self.name == "local" else 60.0
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers if headers else None,
            timeout=_timeout,
        )
        self.model = model
        self._openrouter_config = openrouter_config or {}

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return a normalised LLMResponse.

        ``reasoning_effort`` is sent as the OpenAI-style top-level field;
        LM Studio honours ``"none"`` to switch thinking off on local
        reasoning models (Ornith, Qwen3.x).
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        kwargs["temperature"] = temperature
        if tools and self.supports_tool_calling:
            kwargs["tools"] = tools
        # Pass reasoning config for O3/extended thinking models (OpenRouter)
        if reasoning_config:
            kwargs["reasoning"] = reasoning_config
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        # OpenRouter provider preferences
        if self._openrouter_config:
            transforms = {}
            if self._openrouter_config.get("providers_allowed"):
                transforms["providers"] = {"allow": self._openrouter_config["providers_allowed"]}
            if self._openrouter_config.get("providers_ignored"):
                transforms.setdefault("providers", {})["deny"] = self._openrouter_config["providers_ignored"]
            if transforms:
                kwargs["transforms"] = transforms

        # Apply model constraints (strips/renames/clamps params) before dispatch
        from mullet.core.providers.constraints import shape_request
        kwargs = shape_request(self.model, kwargs)

        with _hud_agent_track(self.name, self.model, messages):
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
        """Yield StreamChunks from a streaming chat completion request."""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        kwargs["temperature"] = temperature
        if tools and self.supports_tool_calling:
            kwargs["tools"] = tools

        # Apply model constraints before dispatch
        from mullet.core.providers.constraints import shape_request, get_constraint_registry
        kwargs = shape_request(self.model, kwargs)
        logger.info("wire: provider=%s model=%s msg_count=%d stream=True",
                    self.name, self.model, len(kwargs.get("messages", [])))

        # Constraint self-healing: errors raised at stream-open time (before
        # the first chunk) are recoverable — learn, reshape, retry once.
        # Errors raised MID-stream are not recoverable here.
        with _hud_agent_track(self.name, self.model, messages):
            try:
                stream = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                fix = self.parse_error(e)
                if fix is None:
                    raise
                omit_param, reason = fix
                get_constraint_registry().learn(self.model, omit_param, reason)
                kwargs = shape_request(self.model, kwargs)
                logger.debug(
                    "Provider %s: stream constraint error — learned omit=%r, reshaping and retrying once",
                    self.name, omit_param,
                )
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

    @staticmethod
    def _is_garbage_content(content: str) -> bool:
        """Detect garbled output (e.g. all-'?' from LM Studio Channel Error)."""
        if not content or len(content) < 40:
            return False
        non_ws = [c for c in content if not c.isspace()]
        if not non_ws:
            return False
        question_ratio = non_ws.count("?") / len(non_ws)
        return question_ratio > 0.8

    def _call_with_retry(self, kwargs: dict, max_retries: int = 3) -> LLMResponse:
        """Call with exponential backoff retry on 429/5xx errors."""
        import time as _time
        # D4 dispatch-time safety net: abort before HTTP if composed payload
        # would overflow the model's context. Soft-fails on estimator /
        # registry errors — see mullet.core.providers.context_gate.
        from mullet.core.providers.context_gate import assert_context_fits
        assert_context_fits(self.name, self.model, kwargs.get("messages", []))
        # Ground-truth wire log — records the ACTUAL model being sent on the wire.
        # Use this to detect any future HUD/wire desync: if HUD says kimi/kimi-k2.5
        # but this line logs moonshot-v1-128k, the registry is picking the wrong model.
        logger.info("wire: provider=%s model=%s msg_count=%d shaped_keys=%s",
                    self.name, self.model, len(kwargs.get("messages", [])), list(kwargs.keys()))
        last_error = None
        _loop_start = _time.monotonic()
        for attempt in range(max_retries):
            if _time.monotonic() - _loop_start > MAX_RETRY_WALL_SECONDS:
                logger.warning(
                    "Provider %s: retry wall-clock budget (%ds) exceeded after %d attempt(s), giving up",
                    self.name, MAX_RETRY_WALL_SECONDS, attempt,
                )
                if last_error is not None:
                    raise last_error
                raise TimeoutError(
                    f"Provider {self.name}: exceeded {MAX_RETRY_WALL_SECONDS}s retry budget"
                )
            try:
                _t0 = _time.monotonic()
                response = self.client.chat.completions.create(**kwargs)
                llm_response = self._to_response(response)
                # Guard: detect garbled response (LM Studio Channel Error → all '?')
                if self._is_garbage_content(llm_response.content):
                    raise RuntimeError(
                        f"Provider {self.name}: garbled response detected (all-'?' content) — "
                        "likely LM Studio Channel Error. Retrying."
                    )
                # Success — reset consecutive failure counter
                reset_consecutive_failures(self.name)
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
                        tier=None,
                    )
                except Exception:
                    pass
                return llm_response
            except Exception as e:
                last_error = e
                # Retry on rate limit and server errors using SDK types when available
                retryable = False
                _is_free_tier = ":free" in self.model
                try:
                    from openai import RateLimitError, APIStatusError
                except ImportError:
                    RateLimitError = type(None)
                    APIStatusError = type(None)
                if isinstance(e, RateLimitError):
                    if _is_free_tier:
                        logger.warning(
                            "Provider %s model %s: 429 on free-tier — not retrying, escalating",
                            self.name, self.model,
                        )
                    else:
                        retryable = True
                elif isinstance(e, APIStatusError) and e.status_code in (500, 502, 503, 504, 529):
                    retryable = True
                if not retryable:
                    # Fallback: check HTTP status codes in error message
                    err_msg = str(e)
                    for code in ("429", "500", "502", "503", "504", "overloaded"):
                        if code in err_msg:
                            if code == "429":
                                if _is_free_tier:
                                    logger.warning(
                                        "Provider %s model %s: 429 on free-tier — not retrying, escalating",
                                        self.name, self.model,
                                    )
                                else:
                                    retryable = True
                            else:
                                retryable = True
                            if retryable:
                                break
                # Garbled response (LM Studio Channel Error → all '?') — retryable
                if not retryable and "garbled response detected" in str(e).lower():
                    retryable = True
                # Context size exceeded — try stripped payload before giving up
                _ctx_exceeded = (
                    "context size has been exceeded" in str(e).lower()
                    or "n_keep" in str(e) and "n_ctx" in str(e)
                    or "context length" in str(e).lower() and "tokens" in str(e).lower()
                )
                if not retryable and _ctx_exceeded and attempt == 0:
                    try:
                        stripped = _strip_for_small_context(kwargs)
                        logger.warning(
                            "Provider %s: context exceeded — retrying with stripped tools/prompt (%d→%d chars)",
                            self.name, _approx_chars(kwargs), _approx_chars(stripped),
                        )
                        kwargs = stripped
                        retryable = True
                    except Exception as strip_err:
                        logger.warning("context strip failed: %s", strip_err)
                if retryable:
                    wait = (2 ** attempt) + (attempt * 0.5)  # 1s, 2.5s, 4.5s
                    logger.warning("Provider %s: %s (retry %d/%d in %.1fs)",
                                   self.name, str(e)[:80], attempt + 1, max_retries, wait)
                    if _time.monotonic() - _loop_start > MAX_RETRY_WALL_SECONDS:
                        logger.warning(
                            "Provider %s: retry wall-clock budget (%ds) exceeded after %d attempt(s), giving up",
                            self.name, MAX_RETRY_WALL_SECONDS, attempt + 1,
                        )
                        if last_error is not None:
                            raise last_error
                        raise TimeoutError(
                            f"Provider {self.name}: exceeded {MAX_RETRY_WALL_SECONDS}s retry budget"
                        )
                    _time.sleep(wait)
                else:
                    # Constraint self-healing: detect novel parameter rejections,
                    # learn them, reshape the request, and retry exactly once.
                    from mullet.core.providers.constraints import (
                        get_constraint_registry, shape_request,
                    )
                    constraint_fix = self.parse_error(e)
                    if constraint_fix is not None:
                        omit_param, reason = constraint_fix
                        get_constraint_registry().learn(self.model, omit_param, reason)
                        kwargs = shape_request(self.model, kwargs)
                        logger.debug(
                            "Provider %s: constraint error — learned omit=%r, reshaping and retrying once",
                            self.name, omit_param,
                        )
                        try:
                            response = self.client.chat.completions.create(**kwargs)
                            llm_response = self._to_response(response)
                            reset_consecutive_failures(self.name)
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
                                    tier=None,
                                )
                            except Exception:
                                pass
                            return llm_response
                        except Exception as e2:
                            e = e2  # constraint fix didn't help — fall through

                    _ms = int((_time.monotonic() - _t0) * 1000)
                    try:
                        from mullet.core.logging.interaction_log import log_interaction
                        log_interaction(
                            provider=self.name,
                            model=self.model,
                            messages=kwargs.get("messages", []),
                            response_content=None,
                            prompt_tokens=0,
                            completion_tokens=0,
                            latency_ms=_ms,
                            success=False,
                            error=str(e)[:200],
                        )
                    except Exception:
                        pass
                    # Track consecutive failures for auto-escalation
                    with _consecutive_failures_lock:
                        _consecutive_failures[self.name] = _consecutive_failures.get(self.name, 0) + 1
                        if _consecutive_failures[self.name] >= _ESCALATION_THRESHOLD:
                            logger.warning(
                                "Provider %s: %d consecutive failures — escalation recommended",
                                self.name, _consecutive_failures[self.name],
                            )
                    raise  # Non-retryable error
        # Track failure when all retries exhausted
        with _consecutive_failures_lock:
            _consecutive_failures[self.name] = _consecutive_failures.get(self.name, 0) + 1
            if _consecutive_failures[self.name] >= _ESCALATION_THRESHOLD:
                logger.warning(
                    "Provider %s: %d consecutive failures — escalation recommended",
                    self.name, _consecutive_failures[self.name],
                )
        _ms = int((_time.monotonic() - _t0) * 1000)
        try:
            from mullet.core.logging.interaction_log import log_interaction
            log_interaction(
                provider=self.name,
                model=self.model,
                messages=kwargs.get("messages", []),
                response_content=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=_ms,
                success=False,
                error=str(last_error)[:200],
            )
        except Exception:
            pass
        raise last_error  # All retries exhausted

    def _handle_stream(self, kwargs: dict) -> LLMResponse:
        """Collect a streaming response into a single LLMResponse."""
        kwargs["stream"] = True
        # Request usage data in the final streaming chunk if supported
        kwargs["stream_options"] = {"include_usage": True}
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        finish_reason = "stop"
        usage = TokenUsage()

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            # Capture usage from the final chunk (has usage but may have no choices)
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    input_tokens=getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(chunk.usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(chunk.usage, "total_tokens", 0) or 0,
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["function"]["arguments"] += tc.function.arguments
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=list(tool_calls.values()) if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
            model=self.model,
            provider=self.name,
        )

    def _to_response(self, response) -> LLMResponse:
        """Convert a raw OpenAI SDK response into the shared LLMResponse shape."""
        if not response.choices:
            logger.warning("Provider %s returned empty choices", self.name)
            return LLMResponse(
                content="", tool_calls=None, finish_reason="error",
                usage=TokenUsage(), model=self.model, provider=self.name,
            )
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

        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        reasoning_content = getattr(choice.message, "reasoning_content", None)

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            model=response.model or self.model,
            provider=self.name,
            reasoning_content=reasoning_content or None,
        )

    def health_check(self) -> bool:
        """Whether a model is ready to serve right now, without loading one.

        A completion probe made LM Studio load a model onto the GPU just to
        answer "is it up?", so this only reads model lists. LM Studio's own
        /api/v0/models reports which models are loaded; any other server is
        healthy when its model list answers.
        """
        base = str(self.client.base_url).rstrip("/")
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        try:
            req = urllib.request.Request(f"{root}/api/v0/models", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read())
        except Exception:
            data = None
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return any(isinstance(m, dict) and m.get("state") == "loaded" for m in data["data"])
        try:
            return bool(getattr(self.client.models.list(timeout=3.0), "data", None))
        except Exception as e:
            logger.debug("Health check failed for %s: %s", self.name, e)
            return False

    def _infer_name(self, base_url: str) -> str:
        """Derive a short provider name from *base_url* (local/openrouter/openai/custom)."""
        if "localhost" in base_url or "127.0.0.1" in base_url:
            return "local"
        if "openrouter" in base_url:
            return "openrouter"
        if "openai.com" in base_url:
            return "openai"
        return "custom"
