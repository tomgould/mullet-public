"""Query contextualization — expand follow-up queries using conversation history."""

import re

# Patterns that indicate a follow-up question
_FOLLOWUP_PATTERNS = [
    r"^(what about|how about|and the|and its|what's the|what are the)\b",
    r"^(also|additionally|furthermore|besides)\b",
    r"^(same|that|this|those|these|it|its|they|them)\b",
    r"^(more|another|other|else)\b",
    r"^\w{1,5}\?$",  # Very short questions like "tests?" reference prior context
]

_FOLLOWUP_RE = re.compile("|".join(_FOLLOWUP_PATTERNS), re.IGNORECASE)


def query_contextualize(query: str, conversation_history: list[str]) -> str:
    """Expand a follow-up query using prior conversation context.

    If the query looks like a follow-up (e.g. "what about the tests?"),
    prepend the most recent substantive query for context.

    Returns the original query if it's self-contained.
    """
    if not conversation_history or not query:
        return query

    # Check if this looks like a follow-up
    if not _FOLLOWUP_RE.search(query.strip()):
        return query

    # Find the most recent substantive query (>10 chars, not a follow-up itself)
    for prev in reversed(conversation_history):
        if len(prev.strip()) > 10 and not _FOLLOWUP_RE.search(prev.strip()):
            return f"{prev.strip()} — {query.strip()}"

    return query
