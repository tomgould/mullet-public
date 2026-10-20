"""Native Google Gemini provider using the google-genai SDK.

Supports Gemini 2.5 Flash/Pro models with tool calling.
Requires GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
"""

import json
import logging

from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.providers.constraints import (
    shape_request,
    get_constraint_registry,
)

logger = logging.getLogger(__name__)


class GeminiProvider(Provider):
    """Native Google Gemini provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
    ):
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except ImportError as exc:
            raise ImportError("google-genai required: pip install google-genai") from exc

        self.model = model
        self.name = "gemini"

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request to the Gemini API."""
        from google.genai import types
        import time as _time

        system_instruction, contents = self._convert_messages(messages)
        gemini_tools = self._convert_tools(tools) if tools else None

        # Apply model constraints via a pseudo-kwargs dict. Gemini uses a
        # GenerateContentConfig object rather than an OpenAI-style kwargs map,
        # so we shape kwargs first, then pull values out when building the
        # config. Any constraint-stripped params simply won't be re-applied.
        shaped_kwargs = shape_request(self.model, {
            "model": self.model,
            "messages": messages,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        })

        _cfg_kwargs: dict = {}
        if "max_output_tokens" in shaped_kwargs:
            _cfg_kwargs["max_output_tokens"] = shaped_kwargs["max_output_tokens"]
        if "temperature" in shaped_kwargs:
            _cfg_kwargs["temperature"] = shaped_kwargs["temperature"]
        config = types.GenerateContentConfig(**_cfg_kwargs)
        if system_instruction:
            config.system_instruction = system_instruction
        if gemini_tools:
            config.tools = gemini_tools

        _t0 = _time.monotonic()
        try:
            response = self._call_with_retry(contents, config)
            _ms = int((_time.monotonic() - _t0) * 1000)
            try:
                from mullet.core.logging.interaction_log import log_interaction
                log_interaction(
                    provider=self.name,
                    model=self.model,
                    messages=messages,
                    response_content=response.content,
                    prompt_tokens=response.usage.input_tokens if response.usage else 0,
                    completion_tokens=response.usage.output_tokens if response.usage else 0,
                    latency_ms=_ms,
                    success=True,
                    tier=None,
                )
            except Exception:
                pass
            return response
        except Exception as e:
            _ms = int((_time.monotonic() - _t0) * 1000)
            try:
                from mullet.core.logging.interaction_log import log_interaction
                log_interaction(
                    provider=self.name,
                    model=self.model,
                    messages=messages,
                    response_content=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=_ms,
                    success=False,
                    error=str(e)[:200],
                )
            except Exception:
                pass
            raise

    def _call_with_retry(self, contents, config, max_retries: int = 3) -> LLMResponse:
        """Call with exponential backoff on retryable errors + constraint self-healing."""
        import time as _time
        # D4 dispatch-time safety net: abort before HTTP if composed payload
        # would overflow the model's context. Soft-fails on estimator /
        # registry errors — see mullet.core.providers.context_gate.
        from mullet.core.providers.context_gate import assert_context_fits
        assert_context_fits(self.name, self.model, contents)
        # Ground-truth wire log
        logger.info("wire: provider=%s model=%s contents_count=%d",
                    self.name, self.model, len(contents) if contents else 0)
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return self._to_response(response)
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                if "429" in err_msg or "resource" in err_msg or "quota" in err_msg or "500" in err_msg:
                    wait = (2 ** attempt) + (attempt * 0.5)
                    logger.warning("Gemini: %s (retry %d/%d in %.1fs)",
                                   str(e)[:80], attempt + 1, max_retries, wait)
                    _time.sleep(wait)
                else:
                    # Constraint self-healing: learn the rejection and rebuild config
                    fix = self.parse_error(e)
                    if fix is not None:
                        omit_param, reason = fix
                        get_constraint_registry().learn(self.model, omit_param, reason)
                        logger.warning(
                            "Gemini: constraint error — learned omit=%r, rebuilding config and retrying once",
                            omit_param,
                        )
                        # Rebuild config with the offending param removed
                        try:
                            cfg_dict = config.to_dict() if hasattr(config, "to_dict") else {}
                        except Exception:
                            cfg_dict = {}
                        cfg_dict.pop(omit_param, None)
                        try:
                            from google.genai import types as _types
                            retry_config = _types.GenerateContentConfig(**cfg_dict)
                            response = self.client.models.generate_content(
                                model=self.model, contents=contents, config=retry_config,
                            )
                            return self._to_response(response)
                        except Exception as e2:
                            last_error = e2
                    raise
        raise last_error

    def health_check(self) -> bool:
        """Check if the Gemini API is reachable."""
        try:
            from google.genai import types
            self.client.models.generate_content(
                model=self.model,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=1),
            )
            return True
        except Exception as e:
            logger.debug("Gemini health check failed: %s", e)
            return False

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool calling."""
        return True

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list]:
        """Convert OpenAI format messages to Gemini format."""
        from google.genai import types

        system = ""
        contents = []
        # Build mapping from tool_call_id -> function name for tool result messages
        tool_name_map: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("function", {}).get("name", "")
                    if tc_id and tc_name:
                        tool_name_map[tc_id] = tc_name

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system = content
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=content)],
                ))
            elif role == "assistant":
                parts = []
                if content:
                    parts.append(types.Part(text=content))
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        args = tc["function"]["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        parts.append(types.Part(function_call=types.FunctionCall(
                            name=tc["function"]["name"],
                            args=args,
                        )))
                contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                # Look up the function name from the preceding assistant tool_calls
                tool_call_id = msg.get("tool_call_id", "")
                func_name = tool_name_map.get(tool_call_id, msg.get("name", "tool"))
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(
                        name=func_name,
                        response={"result": content},
                    ))],
                ))

        return system, contents

    def _convert_tools(self, tools: list[dict]) -> list:
        """Convert OpenAI-format tools to Gemini format."""
        from google.genai import types

        declarations = []
        for tool in tools:
            func = tool.get("function", {})
            params = func.get("parameters", {})
            # Clean up the schema for Gemini (remove unsupported keys)
            clean_params = {
                "type": params.get("type", "object"),
                "properties": params.get("properties", {}),
            }
            if "required" in params:
                clean_params["required"] = params["required"]

            declarations.append(types.FunctionDeclaration(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=clean_params,
            ))

        return [types.Tool(function_declarations=declarations)]

    def _to_response(self, response) -> LLMResponse:
        """Convert Gemini response to standard LLMResponse."""
        content = ""
        tool_calls = []

        if response.candidates:
            candidate = response.candidates[0]
            parts = getattr(candidate.content, "parts", None) if candidate.content else None
            for part in (parts or []):
                if hasattr(part, "text") and part.text:
                    content += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "id": f"call_{hash(fc.name) & 0xFFFFFF:06x}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(dict(fc.args) if fc.args else {}),
                        },
                    })

        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif response.candidates and getattr(response.candidates[0], "finish_reason", None):
            fr = str(response.candidates[0].finish_reason)
            if "MAX_TOKENS" in fr:
                finish_reason = "length"

        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = TokenUsage(
                input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                total_tokens=getattr(um, "total_token_count", 0) or 0,
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
            model=self.model,
            provider=self.name,
        )
