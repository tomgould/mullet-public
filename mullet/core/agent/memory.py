"""Agent memory — persistent context across sessions.

From Hermes: the agent stores durable facts (user preferences,
environment details, tool quirks, conventions) that get injected
into every turn. This reduces future user steering.

Key principle: save what prevents the user from correcting you again.
Don't save task progress or temporary state — save stable facts.
"""

import json
import logging
import time
from typing import TYPE_CHECKING

from mullet.core.paths import data_dir

if TYPE_CHECKING:
    from mullet.core.knowledge.manager import KnowledgeManager

logger = logging.getLogger(__name__)

MEMORY_NAMESPACE = "agent_memory"
MAX_MEMORY_ENTRIES = 20  # Keep context lean
MAX_MEMORY_CHARS = 4000  # Cap injected memory size


class AgentMemory:
    """Persistent memory for the agent across sessions.

    Stores: user preferences, environment facts, recurring corrections,
    project conventions, tool-specific quirks.

    Does NOT store: task progress, session logs, temporary state.
    """

    def __init__(self, knowledge: "KnowledgeManager | None" = None):
        self._knowledge = knowledge
        self._local_path = data_dir() / "memory.json"
        self._entries: dict[str, dict] = {}  # key -> {"content": str, "expires_at": float|None}
        self._load()

    def _load(self):
        """Load memories from Trove or local file."""
        if self._knowledge:
            try:
                results = self._knowledge.memory.list(namespace=MEMORY_NAMESPACE)
                for entry in results[:MAX_MEMORY_ENTRIES]:
                    key = entry.get("key", "")
                    content = entry.get("content", "")
                    if key and content:
                        # Knowledge manager doesn't support TTL, store as permanent
                        self._entries[key] = {"content": content, "expires_at": None}
                return
            except Exception as e:
                logger.warning("agent_memory: Failed to load from KnowledgeManager: %s", e)
                pass

        # Fallback: local file
        if self._local_path.exists():
            try:
                data = json.loads(self._local_path.read_text())
                for k, v in data.items():
                    if isinstance(v, str):
                        # Backward compatibility: plain string -> permanent entry
                        self._entries[k] = {"content": v, "expires_at": None}
                    elif isinstance(v, dict) and "content" in v:
                        # New format: dict with content and expires_at
                        self._entries[k] = {
                            "content": v["content"],
                            "expires_at": v.get("expires_at")
                        }
            except Exception as e:
                logger.warning("agent_memory: Failed to load from local file: %s", e)
                pass

    def save(self, key: str, content: str, ttl_seconds: float | None = None) -> None:
        """Save a memory entry with optional TTL."""
        self.cleanup_expired_memories()
        
        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds
            
        self._entries[key] = {"content": content, "expires_at": expires_at}

        if self._knowledge:
            try:
                # Knowledge manager doesn't support TTL yet, save as permanent
                self._knowledge.remember(key, content, namespace=MEMORY_NAMESPACE)
            except Exception as e:
                logger.warning("agent_memory: Failed to save to KnowledgeManager: %s", e)
                pass

        # Always save locally as backup
        self._save_local()

    def forget(self, key: str):
        """Remove a memory entry."""
        self._entries.pop(key, None)

        if self._knowledge:
            try:
                self._knowledge.memory.delete(namespace=MEMORY_NAMESPACE, key=key)
            except Exception as e:
                logger.warning("agent_memory: Failed to delete from KnowledgeManager: %s", e)
                pass

        self._save_local()

    def recall_all(self) -> dict[str, str]:
        """Get all non-expired memory entries."""
        self.cleanup_expired_memories()
        return {
            key: entry["content"] 
            for key, entry in self._entries.items()
            if entry.get("expires_at") is None or entry["expires_at"] > time.time()
        }

    def build_memory_prompt(self) -> str:
        """Build the memory section for system prompt injection."""
        if not self._entries:
            return ""

        lines = ["# Agent Memory (persistent across sessions)"]
        total_chars = 0

        # Filter expired entries
        active_entries = {
            key: entry["content"]
            for key, entry in self._entries.items()
            if entry.get("expires_at") is None or entry["expires_at"] > time.time()
        }

        if not active_entries:
            return ""

        for key, content in sorted(active_entries.items()):
            entry_text = f"- **{key}**: {content}"
            if total_chars + len(entry_text) > MAX_MEMORY_CHARS:
                lines.append(f"- ... ({len(active_entries) - len(lines) + 1} more entries)")
                break
            lines.append(entry_text)
            total_chars += len(entry_text)

        return "\n".join(lines)

    def cleanup_expired_memories(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._entries.items()
            if entry.get("expires_at") is not None and entry["expires_at"] <= now
        ]
        for key in expired_keys:
            self._entries.pop(key, None)
        
        if expired_keys:
            self._save_local()
        
        return len(expired_keys)

    def _save_local(self):
        """Persist to local JSON file."""
        try:
            self._local_path.parent.mkdir(parents=True, exist_ok=True)
            self._local_path.write_text(json.dumps(self._entries, indent=2))
        except Exception as e:
            logger.warning("agent_memory: Failed to save locally: %s", e)
            pass

    @property
    def count(self) -> int:
        """Return the number of active memory entries."""
        return len(self.recall_all())
