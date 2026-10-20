"""Session search tool — search past conversations."""

import json
from mullet.core.agent.session import search_sessions, list_recent_sessions

SESSION_SEARCH_SCHEMA = {
    "description": "Search past conversation sessions for relevant context. Use when the user references something from a previous conversation or you need cross-session context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for in past sessions."},
            "max_results": {"type": "integer", "description": "Max results. Default: 5."},
        },
        "required": ["query"],
    },
}

SESSION_LIST_SCHEMA = {
    "description": "List recent conversation sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max sessions to list. Default: 10."},
        },
    },
}


def session_search(query: str, max_results: int = 5, **_kwargs) -> str:
    """Search past sessions."""
    results = search_sessions(query, max_results)
    if not results:
        return "No relevant past sessions found."

    lines = []
    for r in results:
        import datetime
        ts = datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M") if r["timestamp"] else "?"
        lines.append(f"[{ts}] ({r['role']}): {r['content']}")
    return "\n".join(lines)


def session_list(limit: int = 10, **_kwargs) -> str:
    """List recent sessions."""
    sessions = list_recent_sessions(limit)
    if not sessions:
        return "No past sessions found."
    return json.dumps(sessions, indent=2, default=str)


def register(registry):
    """Register session tools."""
    registry.register(name="session_search", func=session_search, schema=SESSION_SEARCH_SCHEMA, toolset="session")
    registry.register(name="session_list", func=session_list, schema=SESSION_LIST_SCHEMA, toolset="session")
