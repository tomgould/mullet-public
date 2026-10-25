"""Reads the usage a provider reports in a response, so cost comes from the provider.

A response is either one JSON object or a stream of server-sent events whose
last data event carries ``usage``. The meter keeps a bounded copy of the
body, so a long stream costs no more memory than a short one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

MAX_JSON_BYTES = 4 * 1024 * 1024
TAIL_BYTES = 64 * 1024


@dataclass
class ReportedUsage:
    """One response's usage. Token counts do not overlap: ``input`` is the uncached prompt."""

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float | None = None  # dollars, only when the provider states it

    def tokens(self) -> dict[str, int]:
        return {"input": self.input, "output": self.output, "reasoning": self.reasoning,
                "cache_read": self.cache_read, "cache_write": self.cache_write}


def _count(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else 0


def parse_usage(usage: dict[str, Any]) -> ReportedUsage:
    """Turn an OpenAI-style ``usage`` object into non-overlapping counts, plus the cost when stated."""
    prompt = _count(usage.get("prompt_tokens"))
    completion = _count(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    out_details = (usage.get("completion_tokens_details")
                   if isinstance(usage.get("completion_tokens_details"), dict) else {})
    cached = _count(details.get("cached_tokens")) or _count(usage.get("cached_tokens"))
    written = _count(details.get("cache_write_tokens"))
    reasoning = _count(out_details.get("reasoning_tokens"))
    cost = usage.get("cost")
    return ReportedUsage(
        input=max(0, prompt - cached - written),
        output=max(0, completion - reasoning),
        reasoning=reasoning,
        cache_read=cached,
        cache_write=written,
        cost=float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0 else None,
    )


def _usage_of(payload: bytes) -> dict[str, Any] | None:
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    usage = data.get("usage") if isinstance(data, dict) else None
    return usage if isinstance(usage, dict) else None


class ResponseMeter:
    """Collects one response body and finds the usage in it."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        if len(self._buf) > MAX_JSON_BYTES:
            del self._buf[:-TAIL_BYTES]

    def usage(self) -> ReportedUsage | None:
        """The response's usage, or None when it has none (an error body, say)."""
        body = bytes(self._buf).strip()
        if not body:
            return None
        found = _usage_of(body) if body.startswith(b"{") else None
        if found is None:
            for line in reversed(body.split(b"\n")):
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                found = _usage_of(payload)
                if found is not None:
                    break
        return parse_usage(found) if found is not None else None
