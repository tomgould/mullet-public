"""Stream accumulator — handle partial JSON in streaming LLM responses.

When streaming tool_call responses, the JSON arguments may arrive in
fragments. This module accumulates partial chunks and emits complete
JSON objects as they become parseable.
"""

import json
import logging

logger = logging.getLogger(__name__)


class StreamAccumulator:
    """Accumulate streaming text and extract complete JSON objects.

    Handles the common case where tool_call arguments arrive as partial
    JSON fragments during streaming. Buffers until braces balance, then
    attempts to parse.
    """

    def __init__(self):
        self._buffer: str = ""
        self._text_buffer: str = ""
        self._brace_depth: int = 0
        self._in_string: bool = False
        self._escape_next: bool = False

    def accumulate(self, chunk: str) -> list[dict]:
        """Feed a chunk of streamed text. Returns list of complete JSON objects found."""
        results: list[dict] = []

        for char in chunk:
            self._buffer += char

            if self._escape_next:
                self._escape_next = False
                continue

            if char == '\\' and self._in_string:
                self._escape_next = True
                continue

            if char == '"' and not self._escape_next:
                self._in_string = not self._in_string
                continue

            if self._in_string:
                continue

            if char == '{':
                if self._brace_depth == 0:
                    # Save any text before this JSON object
                    prefix = self._buffer[:-1]
                    if prefix.strip():
                        self._text_buffer += prefix
                    self._buffer = "{"
                self._brace_depth += 1
            elif char == '}':
                self._brace_depth -= 1
                if self._brace_depth == 0:
                    # Try to parse complete JSON object
                    try:
                        obj = json.loads(self._buffer)
                        results.append(obj)
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse JSON: %s", self._buffer[:100])
                    self._buffer = ""
                elif self._brace_depth < 0:
                    # Unbalanced — reset
                    self._brace_depth = 0
                    self._buffer = ""

        return results

    def flush(self) -> str:
        """Flush remaining buffer as plain text. Call when stream ends."""
        remaining = self._text_buffer + self._buffer
        self._buffer = ""
        self._text_buffer = ""
        self._brace_depth = 0
        self._in_string = False
        self._escape_next = False
        return remaining.strip()

    @property
    def pending(self) -> str:
        """Current buffer contents (for debugging)."""
        return self._buffer

    @property
    def has_pending_json(self) -> bool:
        """Whether there's an incomplete JSON object being accumulated."""
        return self._brace_depth > 0


class ToolCallAccumulator:
    """Higher-level accumulator specifically for OpenAI-style tool call streaming.

    Handles the pattern where tool_calls arrive as:
    - index, id, function.name (in one chunk)
    - function.arguments (in fragments across many chunks)
    """

    def __init__(self):
        self._calls: dict[int, dict] = {}  # index -> partial tool_call
        self._completed: list[dict] = []   # tool calls with valid JSON arguments

    def accumulate_delta(self, delta: dict) -> list[dict]:
        """Process a streaming delta and return any newly completed tool calls.

        A tool call is considered complete when its accumulated arguments
        form valid JSON. Completed calls are also available via flush().

        Delta format (OpenAI-style):
            {"tool_calls": [{"index": 0, "id": "call_x", "function": {"name": "f", "arguments": "..."}}]}
        """
        complete: list[dict] = []

        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)

            if idx not in self._calls:
                self._calls[idx] = {
                    "id": tc_delta.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc_delta.get("function", {}).get("name", ""),
                        "arguments": "",
                    },
                }

            call = self._calls[idx]

            # Accumulate function name
            if tc_delta.get("function", {}).get("name"):
                call["function"]["name"] = tc_delta["function"]["name"]

            # Accumulate ID
            if tc_delta.get("id"):
                call["id"] = tc_delta["id"]

            # Accumulate arguments
            args_fragment = tc_delta.get("function", {}).get("arguments", "")
            if args_fragment:
                call["function"]["arguments"] += args_fragment

                # Check if accumulated arguments form valid JSON — if so, the call is complete
                try:
                    json.loads(call["function"]["arguments"])
                    finished = self._calls.pop(idx)
                    self._completed.append(finished)
                    complete.append(finished)
                except json.JSONDecodeError:
                    pass  # Still accumulating

        return complete

    def flush(self) -> list[dict]:
        """Return all accumulated tool calls (complete and incomplete) and reset."""
        calls = list(self._completed)
        for idx in sorted(self._calls.keys()):
            call = self._calls[idx]
            # Validate arguments JSON
            try:
                json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in tool call arguments: %s",
                             call["function"]["arguments"][:100])
            calls.append(call)
        self._calls.clear()
        self._completed.clear()
        return calls

    @property
    def pending_count(self) -> int:
        """Return the number of tool calls still pending."""
        return len(self._calls)
