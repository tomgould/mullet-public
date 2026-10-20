"""Conversation memory — save and recall past conversation summaries.

Stores conversation summaries with metadata for retrieval in future sessions.
Used by the agent to maintain context across conversations without keeping
full conversation history in memory.

Storage: JSON file at .mullet/conversations.json
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationSummary:
    """Summary of a past conversation."""
    id: str
    summary: str
    topics: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    token_count: int = 0

    def to_dict(self) -> dict:
        """Return a dictionary representation of the summary."""
        return {
            "id": self.id,
            "summary": self.summary,
            "topics": self.topics,
            "files_touched": self.files_touched,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationSummary":
        """Create a ConversationSummary from a dictionary."""
        return cls(
            id=data.get("id", ""),
            summary=data.get("summary", ""),
            topics=data.get("topics", []),
            files_touched=data.get("files_touched", []),
            timestamp=data.get("timestamp", 0.0),
            token_count=data.get("token_count", 0),
        )


class ConversationMemory:
    """Persistent conversation memory store.

    Saves summaries of past conversations for recall in future sessions.
    Supports search by topic, file, or keyword.
    """

    def __init__(self, storage_dir: str = ".mullet"):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._storage_dir / "conversations.json"
        self._conversations: list[ConversationSummary] = []
        self._load()

    def save_conversation(self, summary: ConversationSummary):
        """Save a new conversation summary."""
        if not summary.timestamp:
            summary.timestamp = time.time()
        # Replace existing with same ID
        self._conversations = [c for c in self._conversations if c.id != summary.id]
        self._conversations.append(summary)
        self._save()
        logger.debug("Saved conversation %s", summary.id)

    def get_conversation(self, conv_id: str) -> ConversationSummary | None:
        """Get a specific conversation by ID."""
        for c in self._conversations:
            if c.id == conv_id:
                return c
        return None

    def search(self, query: str, max_results: int = 5) -> list[ConversationSummary]:
        """Search conversations by keyword in summary, topics, or files."""
        query_lower = query.lower()
        scored: list[tuple[float, ConversationSummary]] = []

        for conv in self._conversations:
            score = 0.0
            if query_lower in conv.summary.lower():
                score += 2.0
            for topic in conv.topics:
                if query_lower in topic.lower():
                    score += 1.5
            for f in conv.files_touched:
                if query_lower in f.lower():
                    score += 1.0
            if score > 0:
                scored.append((score, conv))

        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [c for _, c in scored[:max_results]]

    def recent(self, count: int = 5) -> list[ConversationSummary]:
        """Get the most recent conversations."""
        sorted_convs = sorted(self._conversations, key=lambda c: -c.timestamp)
        return sorted_convs[:count]

    def list_all(self) -> list[ConversationSummary]:
        """List all stored conversations."""
        return list(self._conversations)

    def delete(self, conv_id: str) -> bool:
        """Delete a conversation by ID."""
        before = len(self._conversations)
        self._conversations = [c for c in self._conversations if c.id != conv_id]
        if len(self._conversations) < before:
            self._save()
            return True
        return False

    def clear(self) -> int:
        """Clear all conversations. Returns count deleted."""
        count = len(self._conversations)
        self._conversations.clear()
        self._save()
        return count

    @property
    def count(self) -> int:
        """Return the number of stored conversations."""
        return len(self._conversations)

    def _load(self):
        """Load conversations from disk."""
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text())
                self._conversations = [
                    ConversationSummary.from_dict(c) for c in data
                ]
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load conversations: %s", e)
                self._conversations = []

    def _save(self):
        """Save conversations to disk."""
        try:
            data = [c.to_dict() for c in self._conversations]
            self._file.write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.error("Failed to save conversations: %s", e)


def summarize_messages(messages: list[dict], max_length: int = 500) -> str:
    """Create a summary from a list of conversation messages.

    Simple extractive summary — takes key user messages and assistant responses.
    For LLM-based summarization, use the agent with a summarize prompt.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        if role in ("user", "assistant") and content:
            prefix = "User: " if role == "user" else "Assistant: "
            parts.append(f"{prefix}{content[:150]}")

    summary = "\n".join(parts[:10])
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    return summary


def extract_topics(messages: list[dict]) -> list[str]:
    """Extract topic keywords from conversation messages."""
    import re
    text = " ".join(
        msg.get("content", "") for msg in messages
        if isinstance(msg.get("content"), str) and msg.get("role") == "user"
    )
    # Extract potential topics (capitalized words, file paths, etc.)
    words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text)
    files = re.findall(r'[\w/]+\.\w+', text)
    topics = list(set(words[:10] + files[:5]))
    return topics[:10]
