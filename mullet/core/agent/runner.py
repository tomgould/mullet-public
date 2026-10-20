"""Agent execution engine — the core conversation loop.

Simplified from Hermes AIAgent. Keeps the essentials: LLM calls,
tool dispatch, iteration budget, interrupt handling, streaming.
"""

import json
import logging
import threading
from typing import Callable

from mullet.core.agent.compression import ContextCompressor
from mullet.core.agent.iteration_budget import IterationBudget
from mullet.core.agent.reasoning import strip_think_blocks
from mullet.core.agent.redact import redact_sensitive_text
from mullet.core.agent.types import AgentResult, TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentRunner:
    """Execute a task using an LLM with tool calling.

    The main loop: send messages -> get response -> if tool_calls,
    dispatch -> append results -> repeat until done or budget exhausted.

    allowed_providers: Optional allowlist envelope inherited by delegated children.
    """

    def __init__(
        self,
        provider: Provider,
        tool_registry: ToolRegistry,
        toolsets: list[str] | None = None,
        system_prompt: str = "",
        max_iterations: int = 90,
        on_token: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, str], None] | None = None,
        on_tool_complete: Callable[[str, str], None] | None = None,
        on_usage: Callable[[str, str, int, int], None] | None = None,
        fallback_providers: list[Provider] | None = None,
        prefill_messages: list[dict] | None = None,
        allowed_providers: frozenset[str] | None = None,
    ):
        """Wire up the runner with provider, tools, callbacks, and budget configuration."""
        self.provider = provider
        self._primary_provider = provider  # Snapshot for turn-scoped restoration
        self._prefill_messages = prefill_messages or []  # Ephemeral, never persisted
        self._fallback_providers = fallback_providers or []
        self._fallback_activated = False
        self.allowed_providers = allowed_providers
        self.tool_registry = tool_registry
        self.toolsets = toolsets
        self.system_prompt = system_prompt
        self.budget = IterationBudget(max_iterations)
        self.on_token = on_token
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_thinking = on_thinking
        self.on_tool_start = on_tool_start
        self.on_tool_complete = on_tool_complete
        # Tool usage analytics
        from mullet.core.agent.analytics import ToolAnalytics
        self._analytics = ToolAnalytics()

        self._interrupt_flag = False
        self._interrupt_message: str | None = None
        self._interrupt_lock = threading.Lock()
        self._active_children: list[AgentRunner] = []
        self.on_usage = on_usage
        self.tier_used = ""  # Set externally by router or CLI
        from mullet.core.agent.compression import get_context_length
        model_name = getattr(provider, "model", "")
        self._compressor = ContextCompressor(context_length=get_context_length(model_name))

    def run(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> AgentResult:
        """Main execution loop."""
        # Restore primary provider at start of each turn (Hermes pattern)
        self._restore_primary_provider()

        messages = self._build_messages(user_message, conversation_history)
        tools = self.tool_registry.get_schemas(self.toolsets) if self.toolsets else self.tool_registry.get_schemas()
        total_usage = TokenUsage()
        tool_calls_made = 0

        while True:
            if self._is_interrupted():
                return AgentResult(
                    completed=False,
                    final_response=self._interrupt_message or "Interrupted.",
                    messages=messages,
                    tool_calls_made=tool_calls_made,
                    iterations_used=self.budget.used,
                    tokens_used=total_usage,
                    provider_used=self.provider.name,
                    tier_used=self.tier_used,
                )

            if not self.budget.consume():
                logger.warning("Iteration budget exhausted (%d/%d)", self.budget.used, self.budget.max_total)
                return AgentResult(
                    completed=False,
                    final_response="Iteration budget exhausted.",
                    messages=messages,
                    tool_calls_made=tool_calls_made,
                    iterations_used=self.budget.used,
                    tokens_used=total_usage,
                    provider_used=self.provider.name,
                    tier_used=self.tier_used,
                )

            # Compress context if approaching limits
            if self._compressor.should_compress(messages):
                logger.info("Compressing context (%d messages)", len(messages))
                messages = self._compressor.compress(messages, provider=self.provider)

            # Hard safety cap — if compression failed or was insufficient
            MAX_MESSAGES = 200
            if len(messages) > MAX_MESSAGES:
                system = [m for m in messages[:2] if m.get("role") == "system"]
                messages = system + messages[-(MAX_MESSAGES - len(system)):]
                logger.warning("Hard message cap hit, truncated to %d messages", len(messages))

            response = self._call_llm(messages, tools)
            if response.usage is not None:
                total_usage = total_usage + response.usage

            # Record metrics
            try:
                from mullet.server.metrics import get_metrics
                m = get_metrics()
                m.inc("mullet_llm_calls_total", labels={"provider": self.provider.name})
                m.inc("mullet_tokens_total", value=response.usage.total_tokens, labels={"type": "total"})
                m.inc("mullet_tokens_total", value=response.usage.input_tokens, labels={"type": "input"})
                m.inc("mullet_tokens_total", value=response.usage.output_tokens, labels={"type": "output"})
            except Exception as _me:
                logger.debug("Metrics update failed (non-fatal): %s", _me)

            # Fire usage callback for budget/tracking integration
            if self.on_usage and response.usage is not None and response.usage.total_tokens > 0:
                self.on_usage(
                    self.provider.name,
                    getattr(self.provider, "model", "unknown"),
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )

            if response.tool_calls:
                # Stream any content that came alongside tool calls
                if self.on_token and response.content:
                    self.on_token(response.content)

                # Append assistant message with tool calls.
                # Include reasoning_content when present so Kimi K2 multi-turn
                # doesn't error with "thinking enabled but reasoning_content missing".
                assistant_msg: dict = {"role": "assistant", "content": response.content or None, "tool_calls": response.tool_calls}
                if response.reasoning_content:
                    assistant_msg["reasoning_content"] = response.reasoning_content
                messages.append(assistant_msg)

                # Try parallel execution for safe batches
                from mullet.core.agent.parallel import should_parallelize, execute_parallel
                if len(response.tool_calls) > 1 and should_parallelize(response.tool_calls):
                    parallel_results = execute_parallel(
                        tool_calls=response.tool_calls,
                        dispatch_fn=self.tool_registry.dispatch,
                        on_tool_call=self.on_tool_call,
                        on_tool_result=self.on_tool_result,
                        on_tool_start=self.on_tool_start,
                        on_tool_complete=self.on_tool_complete,
                    )
                    for pr in parallel_results:
                        result = pr["result"]
                        # Truncate large outputs
                        MAX_TOOL_RESULT_CHARS = 100_000
                        if len(result) > MAX_TOOL_RESULT_CHARS:
                            original_len = len(result)
                            head = result[:MAX_TOOL_RESULT_CHARS * 9 // 10]
                            tail = result[-MAX_TOOL_RESULT_CHARS // 10:]
                            result = f"{head}\n\n[... {original_len - len(head) - len(tail)} chars truncated ...]\n\n{tail}"
                        tool_calls_made += 1
                        self._analytics.record(pr["name"], success=pr["success"],
                                               duration_ms=pr["duration_ms"])
                        # Budget warnings
                        if self.budget.remaining <= 1:
                            result += "\n\n[CRITICAL: Last iteration. Provide your final answer now.]"
                        elif self.budget.remaining <= 3:
                            result += f"\n\n[WARNING: {self.budget.remaining} iterations remaining. Wrap up your work.]"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": pr["tool_call_id"],
                            "content": result,
                        })
                    continue  # Skip sequential loop below

                # Sequential dispatch for single tools or unsafe batches
                for tc in response.tool_calls:
                    try:
                        func = tc.get("function", tc) if isinstance(tc, dict) else {"name": "unknown", "arguments": "{}"}
                        tool_name = func.get("name", "unknown")
                        tool_args = func.get("arguments", "{}")
                        tool_id = tc.get("id", f"call_{tool_calls_made:06d}")
                    except (AttributeError, TypeError):
                        logger.warning("Malformed tool call, skipping: %s", str(tc)[:100])
                        # Must still append a tool result to satisfy API message contract
                        malformed_id = tc.get("id", f"call_malformed_{tool_calls_made:06d}") if isinstance(tc, dict) else f"call_malformed_{tool_calls_made:06d}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": malformed_id,
                            "content": json.dumps({"error": "Malformed tool call — could not parse"}),
                        })
                        continue

                    if self.on_tool_call:
                        self.on_tool_call(tool_name, _preview_args(tool_args, tool_name=tool_name))

                    if self.on_tool_start:
                        self.on_tool_start(tool_name, _preview_args(tool_args, tool_name=tool_name))

                    logger.debug("Tool call: %s(%s)", tool_name, _preview_args(tool_args, 80, tool_name=tool_name))
                    import time as _tc_time
                    _tc_start = _tc_time.time()
                    try:
                        result = self.tool_registry.dispatch(tool_name, tool_args)
                        result = str(result) if result is not None else ""
                        _tc_dur = (_tc_time.time() - _tc_start) * 1000
                        if self.on_tool_complete:
                            self.on_tool_complete(tool_name, str(result)[:150].replace("\n", " "))
                        self._analytics.record(tool_name, success=True, duration_ms=_tc_dur)
                        try:
                            from mullet.server.metrics import get_metrics
                            get_metrics().inc("mullet_tool_calls_total", labels={"tool": tool_name, "status": "success"})
                            get_metrics().observe("mullet_tool_duration_ms", _tc_dur, labels={"tool": tool_name})
                        except Exception as _me:
                            logger.debug("Metrics update failed (non-fatal): %s", _me)
                    except Exception as e:
                        logger.warning("Tool %s raised exception: %s", tool_name, e)
                        result = json.dumps({"error": str(e), "tool": tool_name, "hint": "Check arguments and try again"})
                        _tc_dur = (_tc_time.time() - _tc_start) * 1000
                        self._analytics.record(tool_name, success=False, duration_ms=_tc_dur,
                                               error=str(e))
                        try:
                            from mullet.server.metrics import get_metrics
                            get_metrics().inc("mullet_tool_calls_total", labels={"tool": tool_name, "status": "error"})
                        except Exception as _me:
                            logger.debug("Metrics update failed (non-fatal): %s", _me)

                    # Truncate large tool outputs to prevent context blowup (from Hermes)
                    MAX_TOOL_RESULT_CHARS = 100_000
                    if len(result) > MAX_TOOL_RESULT_CHARS:
                        original_len = len(result)
                        head = result[:MAX_TOOL_RESULT_CHARS * 9 // 10]
                        tail = result[-MAX_TOOL_RESULT_CHARS // 10:]
                        truncated = original_len - len(head) - len(tail)
                        result = f"{head}\n\n[... {truncated} chars truncated ...]\n\n{tail}"

                    tool_calls_made += 1

                    # Inject budget pressure warning (from Hermes thresholds)
                    pressure_msg = self.budget.get_pressure_message()
                    if pressure_msg:
                        result += f"\n\n{pressure_msg}"

                    # Fire tool result callback (redact secrets from preview)
                    if self.on_tool_result:
                        preview = redact_sensitive_text(result[:150].replace("\n", " "))
                        self.on_tool_result(tool_name, preview)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result,
                    })

                    # Handle image analysis results — inject image as multimodal content
                    try:
                        result_data = json.loads(result)
                        if isinstance(result_data, dict) and result_data.get("type") == "image_analysis" and result_data.get("image_data_url"):
                            messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": result_data["image_data_url"]}},
                                    {"type": "text", "text": result_data.get("question", "Describe this image.")},
                                ],
                            })
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass  # Normal — most tool results aren't image analysis
            else:
                # No tool calls — check if this is an intermediate ack ("I'll do X")
                # that should continue rather than stop (from Hermes pattern)
                # Stream thinking content if present
                if self.on_thinking and response.thinking:
                    self.on_thinking(response.thinking)

                clean_content = strip_think_blocks(response.content) or response.content

                if _is_intermediate_ack(clean_content) and tool_calls_made == 0 and self.budget.remaining > 1:
                    # LLM said "I'll do X" but didn't use tools — nudge it to act
                    messages.append({"role": "assistant", "content": clean_content})
                    messages.append({"role": "user", "content": "Please proceed with the action you described. Use the appropriate tools."})
                    logger.debug("Intermediate ack detected, nudging agent to act")
                    continue

                messages.append({"role": "assistant", "content": clean_content})

                if self.on_token and clean_content:
                    self.on_token(clean_content)

                # Save tool analytics
                try:
                    self._analytics.save()
                except Exception as e:
                    logger.debug("Analytics save failed: %s", e)

                return AgentResult(
                    completed=True,
                    final_response=clean_content,
                    messages=messages,
                    tool_calls_made=tool_calls_made,
                    iterations_used=self.budget.used,
                    tokens_used=total_usage,
                    provider_used=self.provider.name,
                    tier_used=self.tier_used,
                )

    def interrupt(self, message: str | None = None):
        """Signal the agent to stop at next safe point and propagate to children."""
        with self._interrupt_lock:
            logger.info("Agent interrupt signal received.")
            self._interrupt_flag = True
            self._interrupt_message = message
            for child in self._active_children:
                child.interrupt(message)

    def clear_interrupt(self):
        """Clear any pending interrupt."""
        with self._interrupt_lock:
            self._interrupt_flag = False
            self._interrupt_message = None

    def _is_interrupted(self) -> bool:
        with self._interrupt_lock:
            return self._interrupt_flag or any(child._is_interrupted() for child in self._active_children)

    def _restore_primary_provider(self):
        """Restore primary provider at start of each turn.

        From Hermes: fallback is turn-scoped. Don't let one transient
        failure pin the session to the fallback permanently.
        """
        if self._fallback_activated:
            self.provider = self._primary_provider
            self._fallback_activated = False
            self._tried_providers = set()  # Reset for new turn
            logger.debug("Restored primary provider: %s", self.provider.name)

    def _try_fallback(self, error: Exception) -> bool:
        """Try activating the next fallback provider.

        Returns True if a fallback was activated, False if chain exhausted.
        Tracks tried providers to avoid cycling back to failed ones.
        """
        if not hasattr(self, "_tried_providers"):
            self._tried_providers: set[str] = set()
        self._tried_providers.add(self.provider.name)

        for fb in self._fallback_providers:
            if self.allowed_providers is not None and fb.name not in self.allowed_providers:
                logger.debug("Skipping fallback %s: not in allowed_providers %s",
                             fb.name, self.allowed_providers)
                continue
            if fb.name not in self._tried_providers:
                logger.warning("Provider %s failed (%s), falling back to %s",
                               self.provider.name, error, fb.name)
                self.provider = fb
                self._fallback_activated = True
                return True
        return False

    def _build_messages(self, user_message: str, history: list[dict] | None) -> list[dict]:
        """Compose the message list (system prompt + prefill + history + user turn) for an LLM call."""
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        # Inject prefill messages (ephemeral few-shot priming, never persisted)
        if self._prefill_messages:
            messages.extend(self._prefill_messages)
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _call_llm(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Call the LLM with automatic fallback on provider failure.

        Delegation safety D1: measure the composed message list before
        dispatch so downstream fallback logic can filter out models whose
        ``max_context`` can't hold the prompt. The estimate is stored on
        ``self._last_prompt_tokens`` for observability and future wiring
        into per-call model selection.
        """
        import time as _time
        # D1: Pre-dispatch prompt-token measurement.
        try:
            from mullet.core.providers.tokenize import estimate_tokens as _estimate_tokens
            self._last_prompt_tokens = _estimate_tokens(messages)
        except Exception as _est_err:
            # Estimator failure must never block dispatch. Fall back to a
            # rough char-based estimate (6 chars/token matches tokenize.py).
            logger.warning("estimate_tokens failed (%s); using rough fallback", _est_err)
            try:
                import json as _json
                self._last_prompt_tokens = max(1, len(_json.dumps(messages, default=str)) // 6)
            except Exception:
                self._last_prompt_tokens = 0
        start = _time.time()
        try:
            result = self.provider.create(
                messages=messages,
                tools=tools if tools else None,
                stream=False,
            )
            elapsed = _time.time() - start
            if elapsed > 30:
                logger.info("LLM call took %.1fs (slow)", elapsed)
            return result
        except Exception as e:
            logger.warning("LLM call failed after %.1fs: %s", _time.time() - start, e)
            # Try all fallback providers until one succeeds
            last_error = e
            while self._try_fallback(last_error):
                self._tried_providers.add(self.provider.name)
                try:
                    return self.provider.create(
                        messages=messages,
                        tools=tools if tools else None,
                        stream=False,
                    )
                except Exception as fb_err:
                    logger.warning("Fallback %s also failed: %s", self.provider.name, fb_err)
                    last_error = fb_err
            raise last_error from e


def _is_intermediate_ack(content: str) -> bool:
    """Detect planning/ack messages that should continue instead of stopping.

    When the LLM says "I'll check the file" or "Let me run the tests" but
    doesn't actually call tools, we should nudge it to act rather than
    accepting the planning statement as the final response.

    From Hermes _looks_like_codex_intermediate_ack pattern.
    """
    if not content or len(content) > 800:
        return False

    lower = content.lower().strip()

    # Must have a future action indicator
    future_phrases = (
        "i'll", "i will", "let me", "i can do that", "i'm going to",
        "let's", "i need to", "i should", "allow me to",
    )
    if not any(phrase in lower for phrase in future_phrases):
        return False

    # Must reference an action
    action_words = (
        "check", "look", "read", "open", "run", "test", "fix",
        "search", "find", "analyze", "review", "inspect", "scan",
        "debug", "examine", "explore", "investigate",
    )
    if not any(word in lower for word in action_words):
        return False

    # Must NOT already contain a result/answer
    result_phrases = ("here is", "the answer", "the result", "i found", "it shows")
    if any(phrase in lower for phrase in result_phrases):
        return False

    return True


def _preview_args(args: str | dict, max_len: int = 120, tool_name: str = "") -> str:
    """Create a short, readable preview of tool arguments.

    Uses tool-specific formatting for common tools (from Hermes display.py).
    """
    if isinstance(args, str):
        try:
            args_dict = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return args[:max_len]
    else:
        args_dict = args

    if not isinstance(args_dict, dict):
        return str(args_dict)[:max_len]

    if not args_dict:
        return ""

    # Tool-specific primary argument extraction
    _PRIMARY = {
        "terminal": "command", "read_file": "path", "write_file": "path",
        "patch": "path", "search_files": "pattern", "list_directory": "path",
        "web_search": "query", "fetch_page": "url", "browse": "url",
        "code_exec": "code", "delegate": "goal",
        "git": "action", "run_tests": "path",
        "session_search": "query", "trove_search": "query",
    }
    primary_key = _PRIMARY.get(tool_name, "")
    if primary_key and primary_key in args_dict:
        val = str(args_dict[primary_key])
        if len(val) > max_len:
            val = val[:max_len - 3] + "..."
        return val

    # Fallback: show first key=value
    s = json.dumps(args_dict)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s
