"""Reasoning and thinking block extraction for LLM responses.

Handles the variety of reasoning model output formats:
- DeepSeek-R1: `reasoning` / `reasoning_content` structured fields
- Claude: `<thinking>` blocks in content
- Qwen-QwQ: inline `<think>` tags
- OpenRouter: `reasoning_details` unified array
- Moonshot/Novita: structured `reasoning` fields

Ported from Hermes agent engine.
"""

import re


# Tags that wrap reasoning/thinking content in model outputs
_THINK_TAGS = r"<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>"
_THINK_CLOSE = r"</(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>"
_THINK_PATTERN = re.compile(
    _THINK_TAGS + r".*?" + _THINK_CLOSE,
    flags=re.DOTALL | re.IGNORECASE,
)
# Orphaned open/close tags
_ORPHAN_TAG = re.compile(
    r"</?(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>\s*",
    flags=re.IGNORECASE,
)


def strip_think_blocks(content: str | None) -> str:
    """Remove reasoning/thinking blocks from content, returning only visible text.

    Handles:
    - Paired tags: <think>...</think>, <thinking>...</thinking>, etc.
    - Orphaned tags: content starting mid-think (e.g., "reasoning</think>answer")
    - Case-insensitive matching
    """
    if not content:
        return ""
    cleaned = _THINK_PATTERN.sub("", content)
    cleaned = _ORPHAN_TAG.sub("", cleaned)
    return cleaned.strip()


def extract_reasoning(content: str | None) -> str:
    """Extract just the reasoning/thinking content from model output.

    Returns the reasoning text if found, empty string otherwise.
    """
    if not content:
        return ""
    blocks = []
    for match in re.finditer(
        _THINK_TAGS + r"(.*?)" + _THINK_CLOSE,
        content, flags=re.DOTALL | re.IGNORECASE,
    ):
        blocks.append(match.group(1).strip())
    return "\n\n".join(blocks)


def extract_content_or_reasoning(response_message: dict) -> str:
    """Extract usable content from an LLM response, falling back to reasoning.

    Resolution order:
      1. message.content — strip inline think blocks, check for remaining text
      2. message.reasoning / message.reasoning_content — structured fields
      3. message.reasoning_details — OpenRouter unified array format

    Args:
        response_message: The message dict from an LLM response (OpenAI format).

    Returns:
        Best available text, or "" if nothing found.
    """
    content = (response_message.get("content") or "").strip()

    if content:
        cleaned = strip_think_blocks(content)
        if cleaned:
            return cleaned

    # Content is empty or reasoning-only — try structured reasoning fields
    reasoning_parts: list[str] = []
    for field in ("reasoning", "reasoning_content"):
        val = response_message.get(field)
        if val and isinstance(val, str) and val.strip() and val not in reasoning_parts:
            reasoning_parts.append(val.strip())

    details = response_message.get("reasoning_details")
    if details and isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                summary = (
                    detail.get("summary")
                    or detail.get("content")
                    or detail.get("text")
                )
                if summary and summary not in reasoning_parts:
                    text = summary.strip() if isinstance(summary, str) else str(summary)
                    reasoning_parts.append(text)

    if reasoning_parts:
        return "\n\n".join(reasoning_parts)

    return ""
