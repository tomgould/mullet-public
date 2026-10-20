"""Knowledge tools — Trove search, memory, and recall as agent-callable tools.

These are registered into the ToolRegistry when a --client is specified,
giving the agent access to project knowledge mid-execution.
"""

import json
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mullet.core.knowledge.manager import KnowledgeManager
    from mullet.core.tools.registry import ToolRegistry


TRAWL_SEARCH_SCHEMA = {
    "description": "Search project knowledge for relevant context. Use when you need information about the codebase, architecture, or project conventions.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "top_k": {"type": "integer", "description": "Max results. Default: 10."},
        },
        "required": ["query"],
    },
}

TRAWL_REMEMBER_SCHEMA = {
    "description": "Store a memory for future sessions. Use to save important discoveries, decisions, or patterns.",
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Short key identifying this memory."},
            "content": {"type": "string", "description": "The memory content to store."},
        },
        "required": ["key", "content"],
    },
}

TRAWL_RECALL_SCHEMA = {
    "description": "Recall stored memories relevant to a topic.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for in memories."},
        },
        "required": ["query"],
    },
}


def make_trawl_search(km: "KnowledgeManager"):
    """Create a trawl_search tool bound to a KnowledgeManager."""
    def trawl_search(query: str, top_k: int = 10, **_kwargs) -> str:
        """Search the project knowledge base for ``query`` (formatted markdown output)."""
        results = km.search(query, top_k=top_k)
        if not results:
            return f"No results found for: {query}"

        lines = []
        for i, r in enumerate(results, 1):
            content = r.get("content", r.get("text", ""))
            source = r.get("source", r.get("metadata", {}).get("source", "unknown"))
            score = r.get("score", 0)
            lines.append(f"**{i}. {source}** (score: {score:.2f})")
            # Truncate long chunks
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(content)
            lines.append("")

        return "\n".join(lines)
    return trawl_search


def make_trawl_remember(km: "KnowledgeManager"):
    """Create a trawl_remember tool bound to a KnowledgeManager."""
    def trawl_remember(key: str, content: str, **_kwargs) -> str:
        """Store ``content`` under ``key`` in the long-term memory."""
        km.remember(key, content)
        return json.dumps({"success": True, "key": key})
    return trawl_remember


def make_trawl_recall(km: "KnowledgeManager"):
    """Create a trawl_recall tool bound to a KnowledgeManager."""
    def trawl_recall(query: str, **_kwargs) -> str:
        """Recall stored memories matching ``query`` from the knowledge base."""
        results = km.recall(query)
        if not results:
            return "No memories found."
        return json.dumps(results, indent=2, default=str)
    return trawl_recall


TRAWL_SEARCH_MULTI_SCHEMA = {
    "description": "Search across multiple project knowledge bases simultaneously. Results are merged using Reciprocal Rank Fusion (RRF). Use when a task spans multiple projects.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "clients": {"type": "string", "description": "Comma-separated client names to search across."},
            "top_k": {"type": "integer", "description": "Max results per client. Default: 5."},
        },
        "required": ["query", "clients"],
    },
}


def make_trawl_search_multi():
    """Create a multi-client search tool using RRF merge."""
    def trawl_search_multi(query: str, clients: str, top_k: int = 5, **_kwargs) -> str:
        """Search ``query`` across multiple comma-separated knowledge bases and RRF-merge the results."""
        client_names = [c.strip() for c in clients.split(",") if c.strip()]
        all_results = []

        for client_name in client_names:
            try:
                from mullet.core.knowledge.manager import knowledge_session
                with knowledge_session(client_name) as km:
                    results = km.search(query, top_k=top_k)
                    for r in results:
                        r["_client"] = client_name
                    all_results.extend(results)
            except Exception as e:
                logger.warning("Error searching client %s: %s", client_name, e)

        # RRF merge: score = sum(1 / (k + rank)) across lists
        rrf_k = 60
        scored = {}
        for rank, r in enumerate(all_results):
            key = r.get("content", "")[:100]
            if key not in scored:
                scored[key] = {"result": r, "rrf_score": 0}
            scored[key]["rrf_score"] += 1.0 / (rrf_k + rank)

        merged = sorted(scored.values(), key=lambda x: -x["rrf_score"])

        lines = []
        for i, item in enumerate(merged[:top_k], 1):
            r = item["result"]
            client = r.get("_client", "?")
            content = r.get("content", r.get("text", ""))[:300]
            source = r.get("source", "")
            lines.append(f"**{i}. [{client}] {source}**")
            lines.append(content)
            lines.append("")

        return "\n".join(lines) if lines else "No results found across clients."

    return trawl_search_multi


def register(registry: "ToolRegistry", km: "KnowledgeManager"):
    """Register all knowledge tools into a ToolRegistry."""
    registry.register(
        name="trawl_search",
        func=make_trawl_search(km),
        schema=TRAWL_SEARCH_SCHEMA,
        toolset="trove",
    )
    registry.register(
        name="trawl_remember",
        func=make_trawl_remember(km),
        schema=TRAWL_REMEMBER_SCHEMA,
        toolset="trove",
    )
    registry.register(
        name="trawl_recall",
        func=make_trawl_recall(km),
        schema=TRAWL_RECALL_SCHEMA,
        toolset="trove",
    )
    registry.register(
        name="trawl_search_multi",
        func=make_trawl_search_multi(),
        schema=TRAWL_SEARCH_MULTI_SCHEMA,
        toolset="trove",
    )
