"""
Lightweight LLM interaction logger for storing prompts, responses, and metadata to SQLite.

This module provides a best-effort logging mechanism that never raises exceptions,
ensuring that logging failures don't crash the caller.
"""

import atexit
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

# Constants
DEFAULT_DB_PATH = Path.home() / ".mullet" / "interactions.db"
MULLET_LOG_FULL_PROMPTS = os.getenv("MULLET_LOG_FULL_PROMPTS", "0") == "1"
MULLET_LOG_FULL_RESPONSES = os.getenv("MULLET_LOG_FULL_RESPONSES", "0") == "1"

# Global connection cache and initialization flag
_CONNECTION: Optional[sqlite3.Connection] = None
_DB_INITIALIZED: bool = False

logger = logging.getLogger(__name__)


def _get_db_path() -> Path:
    """Get the database path, creating directory if needed."""
    db_path = Path(os.getenv("MULLET_INTERACTIONS_DB", str(DEFAULT_DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _get_connection() -> Optional[sqlite3.Connection]:
    """Get a cached SQLite connection, initializing schema on first use."""
    global _CONNECTION, _DB_INITIALIZED

    if _CONNECTION is not None:
        try:
            _CONNECTION.execute("SELECT 1")
            return _CONNECTION
        except (sqlite3.Error, OSError):
            try:
                _CONNECTION.close()
            except Exception:
                pass
            _CONNECTION = None
            _DB_INITIALIZED = False

    try:
        db_path = _get_db_path()
        # isolation_level=None = autocommit; we use explicit BEGIN/COMMIT in writes
        # to avoid "cannot commit — no transaction is active" from implicit tx mgmt
        # clashing with PRAGMA statements.
        _CONNECTION = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        _CONNECTION.execute("PRAGMA journal_mode=WAL")
        _CONNECTION.execute("PRAGMA synchronous=NORMAL")
        if not _DB_INITIALIZED:
            _init_db(_CONNECTION)
            _DB_INITIALIZED = True
        return _CONNECTION
    except (sqlite3.Error, OSError) as e:
        logger.warning("interaction_log: Failed to get DB connection: %s", e)
        return None


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema. Best-effort."""
    try:
        conn.execute("BEGIN")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tier TEXT,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                success INTEGER NOT NULL,
                error TEXT,
                prompt_preview TEXT NOT NULL,
                response_preview TEXT,
                full_prompt TEXT,
                full_response TEXT
            )
        """)
        conn.execute("COMMIT")
    except sqlite3.Error as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        logger.warning("interaction_log: Failed to initialize DB schema: %s", e)


def _extract_preview(text: str, max_length: int = 500) -> str:
    """Extract preview text, handling None and truncation."""
    if text is None:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _extract_last_user_message(messages: List[Dict[str, Any]]) -> str:
    """Extract the last user message from a list of messages."""
    if not messages:
        return ""
    
    # Look for messages in reverse order
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle content as list of text/image blocks
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "\n".join(text_parts)
    return ""


def log_interaction(
    provider: str,
    model: str,
    messages: List[Dict[str, Any]],
    response_content: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    success: bool,
    error: Optional[str] = None,
    tier: Optional[str] = None,
) -> None:
    """
    Log one LLM interaction. Best-effort — never raises.
    
    Args:
        provider: The LLM provider name (e.g., "kimi", "local", "gemini", "anthropic")
        model: The model name (e.g., "kimi-k2-0711-preview")
        messages: List of message dictionaries with "role" and "content" keys
        response_content: The response text from the LLM (None if failed)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the response
        latency_ms: Request latency in milliseconds
        success: Whether the request succeeded
        error: Error message if the request failed
        tier: Service tier (e.g., "STANDARD", "LOW")
    """
    try:
        conn = _get_connection()
        if conn is None:
            return  # Database unavailable, silently skip
        
        # Prepare data for logging
        timestamp = datetime.now(timezone.utc).isoformat()
        prompt_preview = _extract_preview(_extract_last_user_message(messages))
        response_preview = _extract_preview(response_content)
        
        # Serialize full data only if environment variables allow
        full_prompt = json.dumps(messages) if MULLET_LOG_FULL_PROMPTS else None
        full_response = response_content if MULLET_LOG_FULL_RESPONSES else None
        
        # Insert the interaction inside an explicit transaction (isolation_level=None)
        conn.execute("BEGIN")
        conn.execute("""
            INSERT INTO interactions (
                ts, provider, model, tier,
                prompt_tokens, completion_tokens, latency_ms,
                success, error,
                prompt_preview, response_preview,
                full_prompt, full_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            provider,
            model,
            tier,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            1 if success else 0,
            error,
            prompt_preview,
            response_preview,
            full_prompt,
            full_response
        ))
        conn.execute("COMMIT")
    except Exception as e:
        try:
            if conn is not None:
                conn.execute("ROLLBACK")
        except Exception:
            pass
        logger.warning("interaction_log: Failed to log interaction: %s", e)


def _cleanup():
    """Close the database connection on exit."""
    global _CONNECTION
    if _CONNECTION is not None:
        try:
            _CONNECTION.close()
        except Exception as e:
            logger.warning("interaction_log: Failed to close DB connection: %s", e)
            pass
        finally:
            _CONNECTION = None

atexit.register(_cleanup)