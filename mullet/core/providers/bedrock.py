"""AWS Bedrock provider for enterprise deployments.

Uses boto3 bedrock-runtime client for Claude models via AWS.
Requires AWS credentials configured (env vars, ~/.aws/credentials, or IAM role).
"""

import json
import time
import logging

from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider

logger = logging.getLogger(__name__)


class BedrockProvider(Provider):
    """AWS Bedrock provider using the Converse API."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-sonnet-4-6-20250514-v1:0",
    ):
        try:
            import boto3
            self.client = boto3.client("bedrock-runtime", region_name=region)
        except ImportError as exc:
            raise ImportError("boto3 required for Bedrock provider: pip install boto3") from exc

        self.model_id = model_id
        self.model = model_id
        self.name = "bedrock"

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> LLMResponse:
        """Send a conversation request to AWS Bedrock."""
        # Convert to Bedrock Converse format
        system_parts, bedrock_messages = self._convert_messages(messages)

        kwargs = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system_parts:
            kwargs["system"] = system_parts
        if tools:
            kwargs["toolConfig"] = {"tools": self._convert_tools(tools)}
        return self._call_with_retry(kwargs, messages)

    def _call_with_retry(self, kwargs, messages, max_retries=3):
        for attempt in range(max_retries + 1):
            _t0 = time.monotonic()
            try:
                response = self.client.converse(**kwargs)
                llm_response = self._to_response(response)
                _ms = int((time.monotonic() - _t0) * 1000)
                try:
                    from mullet.core.logging.interaction_log import log_interaction
                    log_interaction(
                        provider=self.name,
                        model=self.model,
                        messages=messages,  # original OpenAI-format messages
                        response_content=llm_response.content,
                        prompt_tokens=llm_response.usage.input_tokens if llm_response.usage else 0,
                        completion_tokens=llm_response.usage.output_tokens if llm_response.usage else 0,
                        latency_ms=_ms,
                        success=True,
                    )
                except Exception as _exc:
                    logger.warning("bedrock: failed to log successful interaction: %s", _exc)
                return llm_response
            except Exception as e:
                error_str = str(e).lower()
                retryable = any(x in error_str for x in ["throttlingexception", "serviceunavailableexception", "429", "500"])
                if retryable and attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(f"Bedrock retryable error (attempt {attempt + 1}): {str(e)[:200]}")
                    time.sleep(wait)
                    continue
                _ms = int((time.monotonic() - _t0) * 1000)
                try:
                    from mullet.core.logging.interaction_log import log_interaction
                    log_interaction(
                        provider=self.name,
                        model=self.model,
                        messages=messages,
                        response_content="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=_ms,
                        success=False,
                        error=str(e)[:200],
                    )
                except Exception as _exc:
                    logger.warning("bedrock: failed to log failed interaction: %s", _exc)
                raise

    def health_check(self) -> bool:
        """Check if the Bedrock provider is reachable."""
        try:
            self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": "ping"}]}],
                inferenceConfig={"maxTokens": 1},
            )
            return True
        except Exception:
            return False

    def _convert_messages(self, messages: list[dict]) -> tuple[list, list]:
        """Convert OpenAI format to Bedrock Converse format."""
        system_parts = []
        bedrock_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append({"text": content})
            elif role == "user":
                bedrock_messages.append({"role": "user", "content": [{"text": content}]})
            elif role == "assistant":
                parts = []
                if content:
                    parts.append({"text": content})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        args = tc["function"]["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                logger.warning("Malformed tool args JSON, using empty dict: %s", args[:80] if args else "")
                                args = {}
                        parts.append({
                            "toolUse": {
                                "toolUseId": tc["id"],
                                "name": tc["function"]["name"],
                                "input": args,
                            }
                        })
                bedrock_messages.append({"role": "assistant", "content": parts})
            elif role == "tool":
                bedrock_messages.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": msg["tool_call_id"],
                            "content": [{"text": msg["content"]}],
                        }
                    }],
                })

        return system_parts, bedrock_messages

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-format tools to Bedrock format."""
        bedrock_tools = []
        for tool in tools:
            func = tool.get("function", {})
            bedrock_tools.append({
                "toolSpec": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "inputSchema": {"json": func.get("parameters", {})},
                }
            })
        return bedrock_tools

    def _to_response(self, response: dict) -> LLMResponse:
        """Convert Bedrock response to standard LLMResponse."""
        content = ""
        tool_calls = []

        output = response.get("output", {})
        message = output.get("message", {})

        for block in message.get("content", []):
            if "text" in block:
                content += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append({
                    "id": tu["toolUseId"],
                    "type": "function",
                    "function": {
                        "name": tu["name"],
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                })

        usage_data = response.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("inputTokens", 0),
            output_tokens=usage_data.get("outputTokens", 0),
            total_tokens=usage_data.get("totalTokens", 0),
        )

        stop_reason = response.get("stopReason", "end_turn")
        finish_reason = "tool_calls" if stop_reason == "tool_use" else "stop"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
            model=self.model_id,
            provider=self.name,
        )