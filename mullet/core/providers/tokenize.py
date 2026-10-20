"""Token estimation — tiktoken when available, conservative fallback otherwise.

Canonical helper for estimating prompt size before provider dispatch. Use this
instead of ad-hoc `len(json.dumps(x)) // 4` heuristics — the char/4 rule
overestimates JSON payloads by 20-40% because structural characters (`{`, `"`,
`,`, newlines) count as chars but tokenise to very few tokens.

tiktoken is an optional dependency. When it's not installed we fall back to a
conservative 6-chars-per-token heuristic: plain text is roughly 4 chars/token,
but JSON-wrapped dispatch payloads contain enough structural bloat that 6 is
closer to reality without over-counting.
"""
from __future__ import annotations

import json
from typing import Any

_ENCODER: Any = None


def _lazy_encoder() -> Any:
    """Load and cache the tiktoken encoder. Returns False sentinel when unavailable."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken

            # cl100k_base is the shared GPT-3.5/4 BPE. It's not a perfect match
            # for Claude/Kimi/Gemini but it's within ~10% for routing
            # decisions, and we don't need exactness — just "is this request
            # plausibly going to fit a given model's context window?".
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _ENCODER = False  # sentinel: no tiktoken available
    return _ENCODER


def estimate_tokens(obj: Any) -> int:
    """Best-effort token count for a string or JSON-serialisable object.

    Uses tiktoken (cl100k_base) when available; otherwise falls back to a
    conservative 6 chars/token heuristic for JSON payloads.

    Non-string inputs are JSON-serialised first (messages lists, dicts, etc.).
    Always returns at least 1 for a non-empty input.
    """
    enc = _lazy_encoder()
    if isinstance(obj, str):
        text = obj
    else:
        try:
            text = json.dumps(obj, default=str)
        except (TypeError, ValueError):
            text = str(obj)

    if enc:
        return len(enc.encode(text))

    # Fallback: 6 chars/token. Chosen to avoid overestimating JSON payloads
    # while still leaving some headroom for models with strict context limits.
    return max(1, len(text) // 6)


__all__ = ["estimate_tokens"]
