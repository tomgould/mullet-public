"""Session management — track and search conversation history.

From Hermes: session_search lets the agent recall context from
past conversations without the user repeating themselves.

Sessions are stored as JSON files in ``sessions/`` under the state directory
(see ``mullet.core.paths``). An FTS5 SQLite index in ``sessions.db`` next to
it provides fast full-text search.
"""

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mullet.core.paths import state_dir

logger = logging.getLogger(__name__)

# Tests point these at temporary paths; None means the state directory.
SESSION_DIR: Path | None = None
SESSION_DB: Path | None = None
MAX_SESSIONS = 50  # Keep last N sessions


def _session_dir() -> Path:
    return SESSION_DIR or state_dir() / "sessions"


def _session_db() -> Path:
    return SESSION_DB or state_dir() / "sessions.db"


def _get_db() -> sqlite3.Connection:
    """Open (and initialize if needed) the session FTS5 database."""
    _session_db().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_session_db()))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT DEFAULT '',
            created_at REAL DEFAULT 0,
            provider TEXT DEFAULT '',
            project TEXT DEFAULT '',
            message_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
            session_id UNINDEXED,
            role UNINDEXED,
            content,
            timestamp UNINDEXED,
            tokenize='porter unicode61'
        )
    """)
    conn.commit()
    return conn


def _index_session(conn: sqlite3.Connection, session_id: str, metadata: dict, messages: list[dict]):
    """Index a session's messages into the FTS5 table."""
    # Upsert session metadata
    conn.execute("""
        INSERT OR REPLACE INTO sessions (session_id, title, created_at, provider, project, message_count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        metadata.get("title", ""),
        metadata.get("created_at", 0),
        metadata.get("provider", ""),
        metadata.get("project", ""),
        len(messages),
    ))
    # Remove old FTS entries for this session
    conn.execute("DELETE FROM session_fts WHERE session_id = ?", (session_id,))
    # Insert messages
    for msg in messages:
        content = msg.get("content", "")
        if not content or len(content) < 5:
            continue
        conn.execute(
            "INSERT INTO session_fts (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, msg.get("role", ""), content, msg.get("timestamp", 0)),
        )
    conn.commit()


class Session:
    """A single conversation session."""

    def __init__(self, session_id: str | None = None):
        self.id = session_id or f"session_{int(time.time())}_{os.getpid()}"
        self.messages: list[dict] = []
        self.metadata: dict[str, Any] = {
            "created_at": time.time(),
            "provider": "",
            "project": "",
            "tool_calls": 0,
        }

    def add_message(self, role: str, content: str):
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

    def save(self):
        """Persist session to disk and update FTS5 index."""
        # Auto-title from first user message if untitled
        if not self.metadata.get("title") and self.messages:
            self.metadata["title"] = generate_title(self.messages)

        _session_dir().mkdir(parents=True, exist_ok=True)
        path = _session_dir() / f"{self.id}.json"
        data = {
            "id": self.id,
            "metadata": self.metadata,
            "messages": self.messages,
        }
        path.write_text(json.dumps(data, indent=2))

        # Update FTS5 index
        try:
            conn = _get_db()
            _index_session(conn, self.id, self.metadata, self.messages)
            conn.close()
        except Exception as e:
            logger.debug("FTS5 index update failed: %s", e)

        _cleanup_old_sessions()

    @classmethod
    def load(cls, session_id: str) -> "Session | None":
        """Load a session from disk."""
        path = _session_dir() / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            session = cls(session_id=data["id"])
            session.messages = data.get("messages", [])
            session.metadata = data.get("metadata", {})
            return session
        except Exception:
            return None


def generate_title(messages: list[dict], max_words: int = 7) -> str:
    """Generate a short title from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if not content:
                continue
            content = re.sub(r"@\S+", "", content).strip()
            words = content.split()[:max_words]
            title = " ".join(words)
            if len(content.split()) > max_words:
                title += "..."
            return title
    return "Untitled session"


def search_sessions(query: str, max_results: int = 5) -> list[dict]:
    """Search past sessions using FTS5 full-text search.

    Falls back to naive word overlap if FTS5 is unavailable.
    """
    # Try FTS5 first
    try:
        return _search_fts5(query, max_results)
    except Exception as e:
        logger.debug("FTS5 search failed, falling back to naive: %s", e)
        return _search_naive(query, max_results)


def _search_fts5(query: str, max_results: int) -> list[dict]:
    """Full-text search via SQLite FTS5."""
    conn = _get_db()
    try:
        # FTS5 match query — quote terms for safety
        terms = query.strip().split()
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)

        rows = conn.execute("""
            SELECT session_id, role, snippet(session_fts, 2, '>>>', '<<<', '...', 40) as snippet,
                   timestamp, rank
            FROM session_fts
            WHERE session_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, max_results)).fetchall()

        return [
            {
                "session_id": row[0],
                "role": row[1],
                "content": row[2],
                "timestamp": row[3],
                "score": -row[4],  # FTS5 rank is negative (lower = better)
            }
            for row in rows
        ]
    finally:
        conn.close()


def _search_naive(query: str, max_results: int) -> list[dict]:
    """Fallback word-overlap search on JSON files."""
    if not _session_dir().exists():
        return []

    query_lower = query.lower()
    query_words = set(query_lower.split())
    results = []

    for path in sorted(_session_dir().glob("session_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            messages = data.get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                if not content:
                    continue
                content_words = set(content.lower().split())
                overlap = len(query_words & content_words)
                if overlap >= max(1, len(query_words) // 2):
                    results.append({
                        "session_id": data.get("id", ""),
                        "role": msg.get("role", ""),
                        "content": content[:300],
                        "timestamp": msg.get("timestamp", 0),
                        "score": overlap / max(len(query_words), 1),
                    })
            if len(results) >= max_results * 3:
                break
        except Exception as _exc:
            logger.warning("agent.session: failed to load session from JSON %s for naive search: %s", path, _exc)
            continue

    results.sort(key=lambda r: -r["score"])
    return results[:max_results]


def list_recent_sessions(limit: int = 10) -> list[dict]:
    """List recent sessions from the database, falling back to JSON files."""
    # Try database first
    try:
        conn = _get_db()
        rows = conn.execute("""
            SELECT session_id, title, created_at, provider, project, message_count
            FROM sessions ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        if rows:
            return [
                {
                    "id": r[0], "title": r[1], "created_at": r[2],
                    "provider": r[3], "project": r[4], "messages": r[5],
                }
                for r in rows
            ]
    except Exception:
        pass

    # Fallback to JSON files
    if not _session_dir().exists():
        return []
    sessions = []
    for path in sorted(_session_dir().glob("session_*.json"), reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text())
            meta = data.get("metadata", {})
            sessions.append({
                "id": data.get("id", ""),
                "title": meta.get("title", ""),
                "created_at": meta.get("created_at", 0),
                "provider": meta.get("provider", ""),
                "project": meta.get("project", ""),
                "messages": len(data.get("messages", [])),
            })
        except Exception:
            continue
    return sessions


def rebuild_fts_index():
    """Rebuild the FTS5 index from all JSON session files."""
    if not _session_dir().exists():
        return 0
    conn = _get_db()
    try:
        conn.execute("DELETE FROM session_fts")
        conn.execute("DELETE FROM sessions")
        count = 0
        for path in _session_dir().glob("session_*.json"):
            try:
                data = json.loads(path.read_text())
                _index_session(conn, data["id"], data.get("metadata", {}), data.get("messages", []))
                count += 1
            except Exception:
                continue
        return count
    finally:
        conn.close()


@dataclass
class SessionAggregates:
    """Aggregated statistics across a set of sessions."""

    total_sessions: int = 0
    total_messages: int = 0
    total_tool_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    by_provider: dict[str, dict[str, int | float]] = field(default_factory=dict)
    by_tool: dict[str, int] = field(default_factory=dict)
    window_start: str | None = None  # ISO timestamp of oldest session included
    window_end: str | None = None    # ISO timestamp of newest session included


def get_session_aggregates(
    days: int | None = None,
    limit: int | None = None,
    *,
    sessions_dir: Path | None = None,
) -> SessionAggregates:
    """Aggregate tool_calls / provider / token / cost stats across sessions.

    If ``days`` is set, only sessions whose modification time falls within
    the last *days* days are included.  If ``limit`` is set and no ``days``
    filter applies, only the N most-recent sessions (by mtime) are read.

    Fail-open: malformed session files are skipped with a DEBUG log; a
    missing directory returns all-zero aggregates.
    """
    base_dir = sessions_dir if sessions_dir is not None else _session_dir()
    agg = SessionAggregates()

    if not base_dir.exists():
        return agg

    paths = sorted(base_dir.glob("session_*.json"), key=os.path.getmtime, reverse=True)

    # Apply days filter
    if days is not None:
        cutoff = time.time() - days * 86400
        paths = [p for p in paths if os.path.getmtime(p) >= cutoff]

    # Apply limit (after days filter, or standalone)
    if limit is not None:
        paths = paths[:limit]

    # Collect timestamps for window_start / window_end
    earliest_ts: float | None = None
    latest_ts: float | None = None

    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            logger.debug("session: skipping malformed file %s: %s", path.name, exc)
            continue

        meta = data.get("metadata", {})
        messages = data.get("messages", [])

        agg.total_sessions += 1
        agg.total_messages += len(messages)
        agg.total_tool_calls += int(meta.get("tool_calls", 0))
        agg.total_tokens_in += int(meta.get("tokens_in", 0))
        agg.total_tokens_out += int(meta.get("tokens_out", 0))
        agg.total_cost_usd += float(meta.get("cost_usd", 0.0))

        provider = meta.get("provider", "unknown") or "unknown"
        pb = agg.by_provider.setdefault(provider, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0})
        pb["calls"] += 1
        pb["tokens_in"] += int(meta.get("tokens_in", 0))
        pb["tokens_out"] += int(meta.get("tokens_out", 0))
        pb["cost"] += float(meta.get("cost_usd", 0.0))

        for tool_name, count in meta.get("tool_call_counts", {}).items():
            agg.by_tool[tool_name] = agg.by_tool.get(tool_name, 0) + int(count)

        created = meta.get("created_at")
        if created:
            ts = float(created)
            if earliest_ts is None or ts < earliest_ts:
                earliest_ts = ts
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts

    if earliest_ts is not None and latest_ts is not None:
        import datetime as _dt
        agg.window_start = _dt.datetime.fromtimestamp(earliest_ts).isoformat(timespec="seconds")
        agg.window_end = _dt.datetime.fromtimestamp(latest_ts).isoformat(timespec="seconds")

    return agg


def _cleanup_old_sessions():
    """Remove oldest sessions beyond MAX_SESSIONS."""
    if not _session_dir().exists():
        return
    sessions = sorted(_session_dir().glob("session_*.json"), key=os.path.getmtime)
    while len(sessions) > MAX_SESSIONS:
        oldest = sessions.pop(0)
        oldest.unlink(missing_ok=True)
