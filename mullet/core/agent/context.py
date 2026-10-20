"""Multi-source context injection system.

From OMC: multiple subsystems (knowledge, skills, memory, orchestration)
register context entries with priorities. These get merged and injected
into the system prompt at the start of each turn.

Priority levels: critical > high > normal > low
Sources: knowledge, skills, memory, orchestration, session, custom
"""

import logging
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Priority levels for context entries."""
    CRITICAL = 0  # Always first (e.g., safety rules)
    HIGH = 1      # Important context (e.g., project knowledge)
    NORMAL = 2    # Standard context (e.g., skills, memory)
    LOW = 3       # Background context (e.g., session hints)


@dataclass
class ContextEntry:
    """A single context entry from a source."""
    source: str       # "knowledge" | "skill" | "memory" | "orchestration" | "session" | "custom"
    content: str
    priority: Priority = Priority.NORMAL
    one_shot: bool = False  # If True, consumed after injection (not repeated)
    token_estimate: int = 0

    def __post_init__(self):
        """Estimate token count from content length if not provided."""
        if not self.token_estimate:
            self.token_estimate = len(self.content) // 4


class ContextCollector:
    """Collect and merge context from multiple sources.

    Usage:
        collector = ContextCollector()
        collector.add("knowledge", "Project uses FastAPI...", Priority.HIGH)
        collector.add("skill", "Code review instructions...", Priority.NORMAL)
        collector.add("memory", "User prefers concise output", Priority.LOW)
        injected = collector.build(max_tokens=8000)
    """

    def __init__(self):
        """Create an empty context collector."""
        self._entries: list[ContextEntry] = []

    def add(self, source: str, content: str, priority: Priority = Priority.NORMAL,
            one_shot: bool = False):
        """Register a context entry."""
        if not content or not content.strip():
            return
        self._entries.append(ContextEntry(
            source=source,
            content=content.strip(),
            priority=priority,
            one_shot=one_shot,
        ))

    def build(self, max_tokens: int = 8000) -> str:
        """Build the merged context string, respecting priority and budget."""
        # Sort by priority (lower number = higher priority)
        sorted_entries = sorted(self._entries, key=lambda e: e.priority)

        parts = []
        tokens_used = 0

        for entry in sorted_entries:
            if tokens_used + entry.token_estimate > max_tokens:
                logger.debug("Context budget exceeded at %d tokens, skipping %s",
                             tokens_used, entry.source)
                continue
            parts.append(entry.content)
            tokens_used += entry.token_estimate

        # Remove one-shot entries after consumption
        self._entries = [e for e in self._entries if not e.one_shot]

        return "\n\n".join(parts)

    def clear(self):
        """Clear all entries."""
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        """Return the number of stored entries."""
        return len(self._entries)

    @property
    def total_tokens(self) -> int:
        """Return the total estimated token count."""
        return sum(e.token_estimate for e in self._entries)

    def summary(self) -> dict:
        """Summary of collected context by source."""
        by_source: dict[str, int] = {}
        for e in self._entries:
            by_source[e.source] = by_source.get(e.source, 0) + 1
        return {
            "entries": self.entry_count,
            "tokens": self.total_tokens,
            "sources": by_source,
        }
