"""Anthropic prompt caching (system_and_3 strategy).

Reduces input token costs by ~75% on multi-turn conversations by caching
the conversation prefix. Uses 4 cache_control breakpoints (Anthropic max):
  1. System prompt (stable across all turns)
  2-4. Last 3 non-system messages (rolling window)

Ported from Hermes agent engine. Pure functions, no class state.
"""

import copy
from typing import Any


def _apply_cache_marker(
    msg: dict[str, Any],
    cache_marker: dict[str, str],
    native_anthropic: bool = False,
) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def apply_anthropic_cache_control(
    api_messages: list[dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> list[dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to 4 cache_control breakpoints:
    - System prompt (stable, high cache hit rate)
    - Last 3 non-system messages (rolling window)

    Args:
        api_messages: Conversation messages in OpenAI or Anthropic format.
        cache_ttl: Cache duration ("5m" default, "1h" for long sessions).
        native_anthropic: True if using Anthropic native API format.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker: dict[str, str] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    breakpoints_used = 0

    # Cache system prompt (most stable, highest hit rate)
    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    # Cache last N non-system messages (rolling window)
    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
