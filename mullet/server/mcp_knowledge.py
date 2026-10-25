"""MCP knowledge tools — Trawl knowledge tools — vector search, memory, ingest, graph.

Drop-in replacement for TRAG MCP server tools. Uses Mullet's knowledge engine
with the same tool names and parameter schemas so Claude Code works identically.
"""

import concurrent.futures
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# Must be set before any torch/sentence_transformers import — reduces VRAM
# fragmentation when searches and graph extraction share the same GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import ipaddress
import re
from urllib.parse import urlparse as _urlparse

from mullet.core.env import get_pools_dir
from mullet.knowledge.engine import pool_metadata as _pool_meta

logger = logging.getLogger(__name__)


def _is_safe_url(url: str) -> bool:
    """Block file://, private/loopback IPs, and localhost URLs."""
    try:
        parsed = _urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname or ""
    if hostname.lower() in ("localhost", ""):
        return False
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return False
    except ValueError:
        # Not an IP literal — check for localhost variants
        if hostname.lower().endswith(".localhost"):
            return False
    return True


_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_.\-]+$")
_VALID_CLIENT_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_pool_name(name: str | None) -> str | None:
    """Validate pool_name argument. Returns error message or None if valid."""
    if name is not None and not _VALID_CLIENT_NAME_RE.match(name):
        return f"Error: Invalid client name '{name}'. Only alphanumeric, hyphens, and underscores are allowed."
    return None


def estimate_tokens(text: str) -> int:
    """Estimate token count using character heuristic (1 token ~ 4 chars)."""
    return len(text) // 4


# ============================================================================
# Thread-safe Searcher Cache (ported from TRAG shared.py)
# ============================================================================

class SearcherCache:
    """LRU cache for loaded searchers (thread-safe)."""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self._cache: dict[str, tuple] = {}
        self._access_order: list[str] = []
        self._current_name: Optional[str] = None
        self._lock = threading.Lock()

    def get(self, name: str):
        """Retrieve a cached searcher by name and update its LRU order."""
        with self._lock:
            if name in self._cache:
                self._access_order.remove(name)
                self._access_order.append(name)
                return self._cache[name]
            return None

    def put(self, name: str, client, searcher):
        """Store a client and searcher in the cache, evicting the oldest if at capacity."""
        with self._lock:
            if name in self._cache:
                self._access_order.remove(name)
            elif len(self._cache) >= self.max_size:
                evict_name = self._access_order.pop(0)
                evicted = self._cache.pop(evict_name)
                try:
                    evicted[1].close()
                except Exception as _exc:
                    logger.warning("mcp_knowledge: error closing evicted searcher: %s", _exc)
                logger.info(f"Evicted searcher: {evict_name}")
            self._cache[name] = (client, searcher)
            self._access_order.append(name)

    def pop(self, name: str):
        """Remove a searcher from the cache by name."""
        with self._lock:
            if name in self._cache:
                self._cache.pop(name)
                if name in self._access_order:
                    self._access_order.remove(name)

    @property
    def current_name(self) -> Optional[str]:
        """Return the name of the currently active cached pool."""
        return self._current_name

    @current_name.setter
    def current_name(self, name: Optional[str]):
        """Set the name of the currently active cached pool."""
        self._current_name = name

    def close_all(self):
        """Close all cached searchers and clear cache."""
        with self._lock:
            for name, (_client, searcher) in self._cache.items():
                try:
                    searcher.close()
                except Exception as _exc:
                    logger.warning("mcp_knowledge: error closing cached searcher '%s': %s", name, _exc)
            self._cache.clear()
            self._access_order.clear()
            self._current_name = None


_cache_size = int(os.environ.get("TRAWL_CACHE_SIZE", "3"))
_pool_cache = SearcherCache(max_size=_cache_size)
_jobs: dict[str, dict] = {}
_MAX_JOBS = 100

_trawl_docs_lookup_instance: Any = None


def _trim_jobs() -> None:
    """Keep _jobs bounded to _MAX_JOBS entries by removing oldest finished jobs first."""
    if len(_jobs) <= _MAX_JOBS:
        return
    # First pass: remove completed/failed jobs (oldest first)
    finished = sorted(
        ((k, v) for k, v in _jobs.items() if v.get("status") in ("completed", "failed")),
        key=lambda kv: kv[1].get("started", 0),
    )
    for k, _ in finished:
        if len(_jobs) <= _MAX_JOBS:
            return
        del _jobs[k]
    # Second pass: remove oldest overall if still over limit
    oldest = sorted(_jobs.keys(), key=lambda k: _jobs[k].get("started", 0))
    for k in oldest:
        if len(_jobs) <= _MAX_JOBS:
            return
        del _jobs[k]
def _get_pools_dir() -> Path:
    return get_pools_dir()


def _resolve_pool_name(args: dict) -> str | None:
    """Resolve pool name from args, current pool, or TRAWL_DEFAULT_POOL env var."""
    name = args.get("pool_name", _pool_cache.current_name)
    if not name:
        name = os.environ.get("TRAWL_DEFAULT_POOL")
    return name


# Search mode presets
SEARCH_PRESETS = {
    "precise": {
        "bm25_weight": 0.7, "vector_weight": 0.3,
        "merge_strategy": "weighted", "mmr_lambda": None,
        "_search_params": {"top_k": 10, "min_score": -1.0},
    },
    "broad": {
        "bm25_weight": 0.3, "vector_weight": 0.7,
        "merge_strategy": "weighted", "mmr_lambda": None,
        "_search_params": {"top_k": 30, "min_score": None},
    },
    "code_first": {
        "code_priority": True, "code_priority_boost": 0.5,
        "cross_reference_code": True,
        "_search_params": {"chunk_types": ["code"]},
    },
    "exhaustive": {
        "merge_strategy": "rrf", "adaptive_rerank": False,
        "_search_params": {"top_k": 50, "rerank_top_n": 100},
    },
    "diverse": {
        "mmr_lambda": 0.7,
        "_search_params": {},
    },
}


def _apply_preset(searcher, mode: str) -> tuple[dict, dict]:
    """Apply a search preset, returning (saved_attrs, search_params)."""
    preset = SEARCH_PRESETS.get(mode)
    if not preset:
        return {}, {}
    saved = {}
    search_params = preset.get("_search_params", {}).copy()
    for key, value in preset.items():
        if key == "_search_params":
            continue
        if hasattr(searcher, key):
            saved[key] = getattr(searcher, key)
            setattr(searcher, key, value)
    return saved, search_params


def _restore_preset(searcher, saved_attrs: dict):
    """Restore searcher attributes after preset search."""
    for key, value in saved_attrs.items():
        setattr(searcher, key, value)


def _load_pool(pool_name: str) -> tuple:
    """Load a client + searcher, cached with thread-safe LRU."""
    cached = _pool_cache.get(pool_name)
    if cached:
        _pool_cache.current_name = pool_name
        return cached

    from mullet.knowledge.engine.client import load_client
    from mullet.knowledge.engine.search.searcher import HybridSearcher

    client = load_client(pool_name, clients_dir=_get_pools_dir())

    if not client.index_path.exists():
        raise ValueError(f"Client '{pool_name}' has no index. Run ingestion first.")

    # Create LLM provider for search features (HyDE, multi-query) if enabled
    search_llm = None
    if client.search_config.get("hyde", False) or client.search_config.get("multi_query", False):
        search_llm = _create_llm_provider(client)

    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=client.search_config.get("bm25_weight", 0.5),
        vector_weight=client.search_config.get("vector_weight", 0.5),
        min_score=client.search_config.get("min_score"),
        tokenizer_config=client.tokenization_config,
        embedding_config=client.embedding_config,
        adaptive_rerank=client.search_config.get("adaptive_rerank", True),
        adaptive_rerank_threshold=client.search_config.get("adaptive_rerank_threshold", 0.9),
        adaptive_rerank_depth=client.search_config.get("adaptive_rerank_depth", 15),
        merge_strategy=client.search_config.get("merge_strategy", "weighted"),
        rrf_k=client.search_config.get("rrf_k", 60),
        mmr_lambda=client.search_config.get("mmr_lambda"),
        code_priority=client.search_config.get("code_priority", False),
        code_priority_boost=client.search_config.get("code_priority_boost", 0.5),
        cross_reference_code=client.search_config.get("cross_reference_code", False),
        cross_reference_top_k=client.search_config.get("cross_reference_top_k", 3),
        recency_boost=client.search_config.get("recency_boost", 0.0),
        recency_half_life_days=client.search_config.get("recency_half_life_days", 180),
        prioritize_data=client.search_config.get("prioritize_data", True),
        reranker_backend=client.search_config.get("reranker", "local"),
        reranker_endpoint=client.search_config.get("reranker_endpoint"),
        bm25_k1=client.search_config.get("bm25_k1", 1.5),
        bm25_b=client.search_config.get("bm25_b", 0.75),
        score_normalization=client.search_config.get("score_normalization", "minmax"),
        hyde=client.search_config.get("hyde", False),
        llm_provider=search_llm,
        multi_query=client.search_config.get("multi_query", False),
        multi_query_count=client.search_config.get("multi_query_count", 3),
        hnsw_ef_construction=client.embedding_config.get("hnsw_ef_construction", 200),
        hnsw_max_neighbors=client.embedding_config.get("hnsw_max_neighbors", 16),
        hnsw_ef_search=client.embedding_config.get("hnsw_ef_search", 100),
        cache_ttl=client.search_config.get("cache_ttl", 300),
        cache_max_size=client.search_config.get("cache_max_size", 128),
    )

    _pool_cache.put(pool_name, client, searcher)
    _pool_cache.current_name = pool_name
    return client, searcher


def _format_results(results: list[dict], max_results: int = 20, include_content: bool = True, max_tokens: int | None = None) -> str:
    """Format search results with optional token budget enforcement."""
    if not results:
        return "No results found."

    total_results = len(results)
    lines = [f"Found {total_results} results:\n"]
    token_count = estimate_tokens(lines[0])
    shown = 0

    for i, r in enumerate(results[:max_results]):
        metadata = r.get("metadata", {})
        score = r.get("_rerank_score", r.get("_combined_score", r.get("score", 0)))
        source = metadata.get("source_file", r.get("source", "unknown"))
        chunk_type = metadata.get("chunk_type", r.get("chunk_type", "document"))
        source_client = r.get("_pool") or r.get("_source_client")

        result_lines = [f"## Result {i + 1}"]
        if source_client:
            result_lines.append(f"**Client:** {source_client}")
        result_lines.append(f"**Source:** {source}")
        if metadata.get("source_url"):
            result_lines.append(f"**URL:** {metadata['source_url']}")
        if metadata.get("published_at"):
            result_lines.append(f"**Published:** {metadata['published_at']}")
        result_lines.append(f"**Score:** {score:.3f}")
        result_lines.append(f"**Type:** {chunk_type}")

        if chunk_type == "code":
            lang = metadata.get("language", "")
            sym_type = metadata.get("symbol_type", "")
            sym_name = metadata.get("symbol_name", "")
            line_start = metadata.get("line_start", metadata.get("start_line", ""))
            repo = metadata.get("repo_name", "")
            if lang:
                result_lines.append(f"**Language:** {lang}")
            if sym_type and sym_name:
                result_lines.append(f"**Symbol:** {sym_type} `{sym_name}`")
            if line_start:
                result_lines.append(f"**Lines:** {line_start}-{metadata.get('line_end', metadata.get('end_line', line_start))}")
            if repo:
                result_lines.append(f"**Repo:** {repo}")

        if include_content:
            content = r.get("content", r.get("text", ""))
            if len(content) > 2000:
                content = content[:2000] + "...\n[truncated]"
            if chunk_type == "code":
                result_lines.append(f"\n```{metadata.get('language', '')}\n{content}\n```")
            else:
                result_lines.append(f"\n{content}\n")

        result_lines.append("---")
        result_block = "\n".join(result_lines)
        result_tokens = estimate_tokens(result_block)

        if max_tokens and (token_count + result_tokens) > max_tokens:
            break

        lines.append(result_block)
        token_count += result_tokens
        shown += 1

    if max_tokens:
        lines.append(f"\n**Tokens:** ~{token_count}/{max_tokens} (showing {shown}/{total_results} results)")

    return "\n".join(lines)


# ============================================================================
# Tool Definitions
# ============================================================================

MCP_KNOWLEDGE_TOOLS = [
    # --- Client management ---
    {
        "name": "trawl_list_pools",
        "description": "List all available knowledge pools. Each client has its own indexed document collection.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "trawl_load_pool",
        "description": "Load a specific pool for subsequent searches.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Name of the client to load"},
            },
            "required": ["pool_name"],
        },
    },
    {
        "name": "trawl_graph_prune",
        "description": "Remove orphaned entities, relationships, and processed-chunk tracking whose source chunks no longer exist in the collection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool name"},
            },
            "required": ["pool_name"],
        },
    },
    {
        "name": "trawl_get_pool",
        "description": "Get information about the currently loaded pool.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "trawl_pool_delete",
        "description": "Permanently delete a Trawl pool (directory + index). Requires confirm=true.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Name of the pool directory to delete."},
                "confirm": {"type": "boolean", "description": "Must be true to actually delete.", "default": False},
            },
            "required": ["pool_name"],
        },
    },
    # --- Search ---
    {
        "name": "trawl_search",
        "description": "Search across all indexed documents using hybrid BM25 + vector search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "pool_name": {"type": "string", "description": "Pool to search (optional if already loaded)"},
                "top_k": {"type": "integer", "description": "Number of results (default: 40, max: 50)"},
                "mode": {"type": "string", "description": "Search mode: precise, broad, code_first, exhaustive, diverse, graph_local, graph_global, graph_hybrid, graph_mix"},
                "include_content": {"type": "boolean", "description": "Include full content in results (default: true)"},
                "max_tokens": {"type": "integer", "description": "Max token budget for results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "trawl_search_pool",
        "description": "One-shot search: load a client and search in one call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool to search"},
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results (default: 20)"},
            },
            "required": ["pool_name", "query"],
        },
    },
    {
        "name": "trawl_recent",
        "description": (
            "List a pool's documents newest first, by the best date known for each: "
            "a page's own published date, a PDF's creation date, when the web crawler "
            "first saw it, or when its text last changed. Use this, not search, for "
            "'what's new' / 'recent news' questions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool to list"},
                "limit": {"type": "integer", "description": "Number of documents (default: 20, max: 100)"},
                "since": {"type": "string", "description": "Only documents dated on or after this ISO date, e.g. 2026-09-01"},
            },
            "required": ["pool_name"],
        },
    },
    {
        "name": "trawl_search_code",
        "description": "Search code specifically with language and symbol type filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Code search query"},
                "pool_name": {"type": "string", "description": "Pool to search"},
                "languages": {"type": "array", "items": {"type": "string"}, "description": "Filter by languages (e.g., python, typescript)"},
                "symbol_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by symbol types (function, class, method)"},
                "top_k": {"type": "integer", "description": "Number of results (default: 40, max: 50)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "trawl_search_multi",
        "description": "Search across multiple clients with reciprocal rank fusion. Omit clients to search all.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "clients": {"type": "array", "items": {"type": "string"}, "description": "Clients to search (empty = all with indices)"},
                "top_k": {"type": "integer", "description": "Results per client (default: 20, max: 50)"},
                "mode": {"type": "string", "description": "Search mode preset"},
                "max_tokens": {"type": "integer", "description": "Max token budget for results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "trawl_find_symbol",
        "description": "Find a specific code symbol (function, class, variable) by name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol_name": {"type": "string", "description": "Name of the symbol to find"},
                "pool_name": {"type": "string", "description": "Pool to search"},
                "symbol_type": {"type": "string", "description": "Filter: function, class, method, variable"},
                "language": {"type": "string", "description": "Filter by language (python, js, etc.)"},
            },
            "required": ["symbol_name"],
        },
    },
    # --- Ingestion ---
    {
        "name": "trawl_reindex_pool",
        "description": "Trigger a full reindex of a client's documents. Runs in the background.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Name of the client to reindex"},
            },
            "required": ["pool_name"],
        },
    },
    {
        "name": "trawl_sync_pool",
        "description": "Incremental reindex — only process new/modified files since last index. Fast.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool to sync"},
            },
            "required": ["pool_name"],
        },
    },
    {
        "name": "trawl_ingest_status",
        "description": "Check the status of an ingestion job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID from trawl_reindex_pool"},
                "pool_name": {"type": "string", "description": "Pool name (alternative to job_id)"},
            },
        },
    },
    {
        "name": "trawl_ingest_file",
        "description": "Ingest a single file into a pool.s collection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Target pool"},
                "file_path": {"type": "string", "description": "Path to the file to ingest"},
            },
            "required": ["pool_name", "file_path"],
        },
    },
    {
        "name": "trawl_ingest_url",
        "description": "Ingest content from a URL into a pool.s collection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Target pool"},
                "url": {"type": "string", "description": "URL to fetch and ingest"},
            },
            "required": ["pool_name", "url"],
        },
    },
    # --- Memory ---
    {
        "name": "trawl_memory_store",
        "description": "Store a memory/note with semantic search capability.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Namespace for grouping (default: mcp)"},
                "key": {"type": "string", "description": "Unique key for the memory"},
                "content": {"type": "string", "description": "Content to store"},
                "pool_name": {"type": "string", "description": "Pool scope (optional, global if omitted)"},
                "metadata": {"type": "object", "description": "Optional metadata dict"},
                "ttl": {"type": "integer", "description": "Time-to-live in seconds (optional, no expiry if omitted)"},
            },
            "required": ["key", "content"],
        },
    },
    {
        "name": "trawl_memory_search",
        "description": "Search stored memories semantically.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "pool_name": {"type": "string", "description": "Pool scope (optional, global if omitted)"},
                "namespace": {"type": "string", "description": "Filter by namespace"},
                "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "trawl_memory_list",
        "description": "List all stored memories, optionally filtered by namespace.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool scope (optional, global if omitted)"},
                "namespace": {"type": "string", "description": "Filter by namespace"},
            },
        },
    },
    {
        "name": "trawl_memory_delete",
        "description": "Delete a memory by namespace and key.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Namespace of the memory (default: mcp)"},
                "key": {"type": "string", "description": "Key of the memory to delete"},
                "pool_name": {"type": "string", "description": "Pool scope (optional, global if omitted)"},
            },
            "required": ["key"],
        },
    },
    # --- Knowledge Graph ---
    {
        "name": "trawl_build_graph",
        "description": "Build or rebuild the knowledge graph for a client. Runs in background.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool to build graph for"},
                "rebuild": {"type": "boolean", "description": "Clear and rebuild from scratch (default: false)"},
                "sample": {"type": "integer", "description": "Process only N chunks (for testing)"},
                "workers": {"type": "integer", "description": "Number of extraction workers"},
                "providers": {
                    "type": "string",
                    "description": "Optional comma-separated provider allowlist (e.g. 'kimi,haiku'). When set, the LLM used for graph extraction must come from this list or the job fails. Accepts: local, kimi, gemini, haiku, sonnet, opus.",
                },
            },
            "required": ["pool_name"],
        },
    },
    {
        "name": "trawl_graph_stats",
        "description": "Get knowledge graph statistics (node/edge counts, entity types).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool name"},
            },
            "required": ["pool_name"],
        },
    },
    {
        "name": "trawl_search_entities",
        "description": "Search for entities in the knowledge graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Entity search query"},
                "pool_name": {"type": "string", "description": "Pool name"},
                "entity_type": {"type": "string", "description": "Filter by type (person, concept, etc.)"},
                "top_k": {"type": "integer", "description": "Max results (default: 20, max: 100)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "trawl_search_relationships",
        "description": "Search for relationships between entities in the knowledge graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity to find relationships for"},
                "pool_name": {"type": "string", "description": "Pool name"},
                "relationship_type": {"type": "string", "description": "Filter by relationship type"},
                "max_hops": {"type": "integer", "description": "Max traversal depth (default: 2, max: 4)"},
            },
            "required": ["entity"],
        },
    },
    # --- Sources ---
    {
        "name": "trawl_list_sources",
        "description": "List document sources for a client.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool name"},
            },
            "required": ["pool_name"],
        },
    },
    {
        "name": "trawl_sync_sources",
        "description": "Sync all sources for a client (pull latest from git repos, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool name"},
            },
            "required": ["pool_name"],
        },
    },
    {
        "name": "trawl_connect_github",
        "description": "Add a GitHub account, org, or repo as a source for a pool.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool name"},
                "owner": {"type": "string", "description": "GitHub username or org"},
                "repo": {"type": "string", "description": "Specific repo (optional)"},
                "include_repos": {"type": "array", "items": {"type": "string"}, "description": "Whitelist repos"},
                "exclude_repos": {"type": "array", "items": {"type": "string"}, "description": "Blacklist repos"},
            },
            "required": ["pool_name", "owner"],
        },
    },
    {
        "name": "trawl_crawl_web",
        "description": "Crawl a website and save pages as Markdown for a client.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool name"},
                "url": {"type": "string", "description": "URL to crawl"},
                "max_pages": {"type": "integer", "description": "Max pages to crawl (default: 100)"},
                "stay_in_path": {"type": "boolean", "description": "Only crawl under starting URL path (default: true)"},
            },
            "required": ["pool_name", "url"],
        },
    },
    # --- Summarize ---
    {
        "name": "trawl_summarize",
        "description": "Search and return concise summaries of results using LLM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query to summarize results for"},
                "top_k": {"type": "integer", "description": "Number of results to summarize (default: 5)"},
                "pool_name": {"type": "string", "description": "Pool to search (optional if loaded)"},
                "providers": {
                    "type": "string",
                    "description": "Optional comma-separated provider allowlist (e.g. 'kimi,haiku'). When set, the LLM used for summarization must come from this list or the call fails. Accepts: local, kimi, gemini, haiku, sonnet, opus.",
                },
            },
            "required": ["query"],
        },
    },
    # --- Repos ---
    {
        "name": "trawl_list_repos",
        "description": "List GitHub repos indexed for a client.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool name"},
            },
            "required": ["pool_name"],
        },
    },
    # --- Search Repo ---
    {
        "name": "trawl_search_repo",
        "description": "Search within a specific repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_name": {"type": "string", "description": "Repository name to search within"},
                "query": {"type": "string", "description": "Search query"},
                "pool_name": {"type": "string", "description": "Pool name"},
                "top_k": {"type": "integer", "description": "Number of results (default: 20)"},
                "chunk_types": {"type": "array", "items": {"type": "string"}, "description": "Filter: code, document"},
            },
            "required": ["repo_name", "query"],
        },
    },
    # --- Graph Export ---
    {
        "name": "trawl_graph_export",
        "description": "Export knowledge graph as D3.js-compatible JSON for visualization.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool name"},
                "max_nodes": {"type": "integer", "description": "Max nodes to export (default: 500)"},
            },
        },
    },
    # --- Workflows ---
    {
        "name": "trawl_workflow_create",
        "description": "Create a workflow with named steps and dependency ordering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Workflow name"},
                "steps": {"type": "array", "items": {"type": "object"}, "description": "Steps: [{name, description?, depends_on?}]"},
                "pool_name": {"type": "string", "description": "Pool scope (optional, global if omitted)"},
            },
            "required": ["name", "steps"],
        },
    },
    {
        "name": "trawl_workflow_status",
        "description": "Get workflow status and step progress.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "description": "Workflow ID"},
                "pool_name": {"type": "string", "description": "Pool scope"},
            },
            "required": ["workflow_id"],
        },
    },
    {
        "name": "trawl_workflow_update",
        "description": "Update a workflow step's status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "description": "Workflow ID"},
                "step_id": {"type": "string", "description": "Step ID to update"},
                "status": {"type": "string", "description": "New status: pending, running, completed, failed, skipped"},
                "result": {"type": "string", "description": "Optional result text"},
                "pool_name": {"type": "string", "description": "Pool scope"},
            },
            "required": ["workflow_id", "step_id", "status"],
        },
    },
    {
        "name": "trawl_workflow_list",
        "description": "List all workflows, optionally filtered by status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pool_name": {"type": "string", "description": "Pool scope"},
                "status": {"type": "string", "description": "Filter by status: pending, running, completed, failed"},
            },
        },
    },
    # --- Agents ---
    {
        "name": "trawl_agents_list",
        "description": "List available agent templates, optionally filtered by tag.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "Filter by tag"},
            },
        },
    },
    {
        "name": "trawl_agents_get",
        "description": "Get full agent template config including system prompt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Agent name or filename slug"},
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "trawl_run_agent",
        "description": "Run a Trove agent with local LLM — searches context, applies agent prompt, generates response.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Agent to run"},
                "query": {"type": "string", "description": "Query or task for the agent"},
                "pool_name": {"type": "string", "description": "Client for context"},
                "top_k": {"type": "integer", "description": "Context chunks to retrieve (default: 10)"},
                "providers": {
                    "type": "string",
                    "description": "Optional comma-separated provider allowlist (e.g. 'kimi,haiku'). When set, the LLM used by the agent must come from this list or the call fails. Accepts: local, kimi, gemini, haiku, sonnet, opus.",
                },
            },
            "required": ["agent_name", "query"],
        },
    },
    # --- Prompts ---
    {
        "name": "trawl_prompts_search_live",
        "description": "Search the prompts.chat API for prompt templates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "type": {"type": "string", "description": "Filter by type: TEXT, IMAGE, VIDEO, AUDIO, SKILL"},
                "category": {"type": "string", "description": "Filter by category"},
                "tag": {"type": "string", "description": "Filter by tag"},
                "sort": {"type": "string", "description": "Sort order: newest, popular (default: relevance)"},
                "limit": {"type": "integer", "description": "Max results (default: 10, max: 50)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "trawl_prompts_find",
        "description": "Find prompts by combining local index and live API search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "type": {"type": "string", "description": "Filter by type: TEXT, IMAGE, VIDEO, AUDIO, SKILL"},
                "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "trawl_prompts_get",
        "description": "Get a specific prompt template by ID with optional variable substitution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt_id": {"type": "string", "description": "Prompt ID from prompts.chat"},
                "variables": {"type": "object", "description": "Variables to substitute in the prompt"},
            },
            "required": ["prompt_id"],
        },
    },
    {
        "name": "trawl_docs_lookup",
        "description": "Look up documentation for a library using Context7 (version-aware) with fallback to local Trawl pools. Use this when you need API docs, usage examples, or version-specific information for any library.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "library": {"type": "string", "description": "Library name (e.g. 'react', 'fastapi', 'numpy')"},
                "version": {"type": "string", "description": "Library version (e.g. '18.0', 'latest'). Use 'latest' if unsure."},
                "query": {"type": "string", "description": "What you want to know (e.g. 'hooks API', 'async middleware', 'array operations')"},
                "sources": {"type": "array", "items": {"type": "string"}, "description": "Source preference order. Default: ['context7', 'trawl', 'web']"},
            },
            "required": ["library", "query"],
        },
    },
]


# ============================================================================
# Handler Implementations
# ============================================================================

# current client tracked via _pool_cache.current_name


def handle_list_pools(args: dict) -> str:
    """List all configured Trawl pools with summary info."""
    clients_dir = _get_pools_dir()
    if not clients_dir.exists():
        return "No pools directory found."
    clients = []
    for d in sorted(clients_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "config.yaml").exists():
            yaml_path = d / "config.yaml"
        elif (d / "client.yaml").exists():
            yaml_path = d / "client.yaml"  # legacy fallback
        else:
            continue
        name = d.name
        desc = ""
        try:
            import yaml
            cfg = yaml.safe_load(yaml_path.read_text())
            desc = cfg.get("description", "")
            name = cfg.get("name", d.name)
        except Exception as _exc:
            logger.warning("mcp_knowledge: error reading pool config for '%s': %s", d.name, _exc)
        meta = _pool_meta.read_metadata(d)
        parts = [f"- **{name}**: {desc}"]
        details = []
        if meta.get("last_searched_at"):
            details.append(f"searched {meta['last_searched_at']}")
        if meta.get("search_count"):
            details.append(f"{meta['search_count']} searches")
        if meta.get("last_indexed_at"):
            details.append(f"indexed {meta['last_indexed_at']}")
        if details:
            parts.append(" [" + ", ".join(details) + "]")
        else:
            parts.append(" [no usage yet]")
        clients.append("".join(parts))
    if not clients:
        return "No pools found."
    return "Available Trawl pools:\n\n" + "\n".join(clients)


def handle_load_pool(args: dict) -> str:
    """Load a Trawl pool by name and set it as current."""
    # _pool_cache.current_name is set via property
    name = args.get("pool_name", "")
    if not name:
        return "Error: pool_name is required."
    try:
        _pool_cache.pop(name)  # force fresh load, not stale cache
        client, searcher = _load_pool(name)
        _pool_cache.current_name = name
        try:
            # Record against the resolved pool dir (client.path), not the raw
            # display name — otherwise a display-name load spawns a stub dir.
            _pool_meta.mark_searched(client.path)
        except Exception as _exc:
            logger.warning("mcp_knowledge: error marking pool '%s' as searched: %s", name, _exc)
        return f"Loaded client **{name}**. {searcher.collection.count()} chunks indexed."
    except Exception as e:
        return f"Error loading client '{name}': {e}"


def handle_get_pool(args: dict) -> str:
    """Get info about the currently loaded Trawl pool."""
    if not _pool_cache.current_name:
        return "No pool currently loaded. Use trawl_load_pool first."
    try:
        client, searcher = _load_pool(_pool_cache.current_name)
        return f"Current client: **{_pool_cache.current_name}**\nChunks: {searcher.collection.count()}\nDescription: {client.config.get('description', 'N/A')}"
    except Exception as e:
        return f"Error: {e}"


def handle_delete_pool(args: dict) -> str:
    """Delete a Trawl pool and its index after confirmation."""
    pool_name = args.get("pool_name", "").strip()
    confirm = args.get("confirm", False)
    if not pool_name:
        return "Error: pool_name is required."
    pool_dir = _get_pools_dir() / pool_name
    if not pool_dir.exists() or not pool_dir.is_dir():
        return f"Error: pool '{pool_name}' not found."
    if not confirm:
        return (
            f"Refusing to delete pool '{pool_name}' without confirmation. "
            f"Pass confirm=true to proceed. This will permanently remove {pool_dir}."
        )
    try:
        _pool_meta.delete_pool(pool_dir)
        try:
            if _pool_cache.current_name == pool_name:
                _pool_cache.current_name = None
        except Exception as _exc:
            logger.warning("mcp_knowledge: error clearing current pool name after delete: %s", _exc)
    except Exception as e:
        return f"Error deleting pool '{pool_name}': {e}"
    return f"Pool '{pool_name}' deleted."


def handle_search(args: dict) -> str:
    """Search the current Trawl pool with optional mode and quality metrics."""
    query = args.get("query", "")
    pool_name = _resolve_pool_name(args)
    top_k = min(int(args.get("top_k", 40)), 50)
    mode = args.get("mode")
    include_content = args.get("include_content", True)
    max_tokens = int(args["max_tokens"]) if args.get("max_tokens") else None
    if not pool_name:
        return "Error: No pool specified. Use pool_name parameter or trawl_load_pool first."
    if not query:
        return "Error: query is required."
    try:
        client, searcher = _load_pool(pool_name)
        saved_attrs = {}
        extra_params = {}

        # Handle graph search modes
        graph_mode = None
        if mode and mode.startswith("graph_"):
            graph_mode = mode.replace("graph_", "")
            mode = None

        if mode:
            saved_attrs, extra_params = _apply_preset(searcher, mode)
        search_timeout = int(args.get("timeout", 45))
        try:
            from mullet.knowledge.engine.search.types import SearchResult, SearchQuality
            search_top_k = extra_params.pop("top_k", top_k)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                if graph_mode and hasattr(searcher, 'graph_search'):
                    _fut = _ex.submit(searcher.graph_search, query=query, mode=graph_mode, top_k=search_top_k)
                else:
                    _fut = _ex.submit(searcher.search_with_quality, query=query, top_k=search_top_k, **extra_params)
                try:
                    search_result = _fut.result(timeout=search_timeout)
                except concurrent.futures.TimeoutError:
                    return (
                        f"Error: Search timed out after {search_timeout}s. "
                        "The index may be busy (GPU conflict during reindex). "
                        "Try again shortly, reduce top_k, or pass a shorter timeout value."
                    )
                if graph_mode and hasattr(searcher, 'graph_search') and not isinstance(search_result, SearchResult):
                    search_result = SearchResult(results=search_result, quality=SearchQuality())
        finally:
            if saved_attrs:
                _restore_preset(searcher, saved_attrs)
        q = search_result.quality
        output = f"Searching **{client.name}** for: *{query}*"
        if args.get("mode"):
            output += f" (mode: {args['mode']})"
        output += "\n\n"
        output += (
            f"**Quality:** avg={q.avg_rerank_score:.2f}, top={q.top_score:.2f}, "
            f"spread={q.score_spread:.2f}, sources={q.unique_sources}, "
            f"candidates={q.total_candidates}, strategy={q.merge_strategy}, "
            f"time={q.timing.get('total', 0):.2f}s\n\n"
        )
        output += _format_results(search_result.results, max_results=top_k, include_content=include_content, max_tokens=max_tokens)

        # Append citation map if enabled in pool config
        if client.llm_config.get("citations", False) and search_result.results:
            try:
                from mullet.knowledge.engine.llm import build_citation_map
                cmap = build_citation_map(search_result.results)
                output += "\n\n**Citation Map:**\n"
                for num, entry in cmap.entries.items():
                    page = entry.get("page")
                    page_str = f", p.{page}" if page else ""
                    output += f"[{num}] {entry['source']}{page_str}\n"
            except (ImportError, AttributeError):
                pass
        try:
            # Record against the resolved pool dir (client.path), not the raw
            # display name — otherwise a display-name search spawns a stub dir.
            _pool_meta.mark_searched(client.path)
        except Exception as _exc:
            logger.warning("mcp_knowledge: error marking pool '%s' as searched: %s", pool_name, _exc)
        return output
    except Exception as e:
        return f"Search error: {e}"


def handle_search_pool(args: dict) -> str:
    """Search a specific Trawl pool (alias for handle_search)."""
    args["pool_name"] = args.get("pool_name", "")
    return handle_search(args)


_RECENCY_LABELS = {
    "published": "published",
    "pdf_created": "PDF created",
    "first_seen": "first seen by crawler",
    "updated": "page updated",
}


def _recent_documents(collection, since_ts: int | None) -> list[dict]:
    """One metadata dict per dated document, newest first."""
    metadatas: list = []
    # published_ts covers indexes built before recency_ts existed
    for key in ("recency_ts", "published_ts"):
        where = {key: {"$gte": since_ts}} if since_ts is not None else {key: {"$gt": 0}}
        metadatas = collection.get(where=where, include=["metadatas"]).get("metadatas") or []
        if metadatas:
            break
    docs: dict[str, dict] = {}
    for meta in metadatas:
        key = meta.get("source_path") or meta.get("source_file")
        if key and key not in docs:
            docs[key] = meta
    return sorted(
        docs.values(),
        key=lambda m: m.get("recency_ts", m.get("published_ts", 0)),
        reverse=True,
    )


def handle_recent(args: dict) -> str:
    """List a pool's documents newest first by their best-known date."""
    from datetime import datetime, timezone

    pool_name = _resolve_pool_name(args)
    if not pool_name:
        return "Error: No pool specified."
    limit = max(1, min(int(args.get("limit", 20)), 100))
    since_ts = None
    if args.get("since"):
        try:
            since = datetime.fromisoformat(str(args["since"]).replace("Z", "+00:00"))
        except ValueError:
            return "Error: since must be an ISO date, e.g. 2026-09-01."
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        since_ts = int(since.timestamp())

    try:
        _client, searcher = _load_pool(pool_name)
        docs = _recent_documents(searcher.collection, since_ts)
    except Exception as e:
        return f"Recent documents error: {e}"
    if not docs:
        return (
            f"No dated documents in **{pool_name}**"
            + (f" since {args['since']}." if since_ts is not None else ".")
            + " Dates come from published dates, PDF metadata and the web crawler; "
            "reindex the pool if it was built before these were recorded."
        )

    lines = [f"Newest documents in **{pool_name}** ({len(docs)} dated):\n"]
    for i, meta in enumerate(docs[:limit], 1):
        ts = meta.get("recency_ts", meta.get("published_ts"))
        day = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
        source = _RECENCY_LABELS.get(meta.get("recency_source", "published"), "dated")
        name = meta.get("doc_title") or Path(meta.get("source_file", "")).name
        lines.append(f"{i}. {day} | {source} | {name}")
        if meta.get("source_url"):
            lines.append(f"   {meta['source_url']}")
    return "\n".join(lines)


def handle_search_code(args: dict) -> str:
    """Search for code chunks with language and symbol type filtering."""
    query = args.get("query", "")
    pool_name = _resolve_pool_name(args)
    # Support both singular and array forms for compatibility
    languages = args.get("languages", [])
    if not languages and args.get("language"):
        languages = [args["language"]]
    symbol_types = args.get("symbol_types", [])
    if not symbol_types and args.get("symbol_type"):
        symbol_types = [args["symbol_type"]]
    top_k = min(int(args.get("top_k", 40)), 50)
    if not pool_name:
        return "Error: No pool specified."
    if not query:
        return "Error: query is required."
    try:
        client, searcher = _load_pool(pool_name)
        results = searcher.search(query, top_k=top_k * 3, chunk_types=["code"])
        # Filter by languages/symbol_types
        if languages:
            langs_lower = [lang.lower() for lang in languages]
            results = [r for r in results if r.get("metadata", {}).get("language", "").lower() in langs_lower]
        if symbol_types:
            types_lower = [t.lower() for t in symbol_types]
            results = [r for r in results if r.get("metadata", {}).get("symbol_type", "").lower() in types_lower]
        return _format_results(results[:top_k], max_results=top_k)
    except Exception as e:
        return f"Code search error: {e}"


def handle_search_multi(args: dict) -> str:
    """Search across multiple Trawl pools and merge results with RRF."""
    query = args.get("query", "")
    pool_names = args.get("clients", args.get("pool_names", []))
    top_k = min(int(args.get("top_k", 20)), 50)
    mode = args.get("mode")
    max_tokens = int(args["max_tokens"]) if args.get("max_tokens") else None
    if not query:
        return "Error: query is required."
    # Auto-discover pools if none specified
    if not pool_names:
        from mullet.knowledge.engine.client import discover_clients
        all_pools = discover_clients(clients_dir=_get_pools_dir())
        pool_names = [c for c in all_pools if (_get_pools_dir() / c / "index").exists()]
    if not pool_names:
        return "No pools with indices found."

    client_results = {}
    quality_lines = []
    errors = []
    for name in pool_names:
        try:
            client, searcher = _load_pool(name)
            saved_attrs = {}
            extra_params = {}
            if mode:
                saved_attrs, extra_params = _apply_preset(searcher, mode)
            try:
                search_top_k = extra_params.pop("top_k", top_k)
                sr = searcher.search_with_quality(query=query, top_k=search_top_k, **extra_params)
            finally:
                if saved_attrs:
                    _restore_preset(searcher, saved_attrs)
            client_results[name] = sr.results
            q = sr.quality
            quality_lines.append(
                f"  - **{name}**: {len(sr.results)} results, "
                f"top={q.top_score:.2f}, avg={q.avg_rerank_score:.2f}, "
                f"time={q.timing.get('total', 0):.2f}s"
            )
        except Exception as e:
            errors.append(f"  - **{name}**: Error - {e}")
            logger.warning(f"Multi-pool search failed for {name}: {e}")

    if not client_results:
        return "All pools failed:\n" + "\n".join(errors)

    # RRF merge across pools with source-file dedup + configurable knobs.
    # - `rrf_k`: clamp to [1, 200]; default 60.
    # - `pool_weights`: dict[pool_name, float]; defaults to 1.0 per pool.
    # - Dedup key is `metadata.source_file` when present (collapsing the same
    #   file indexed in multiple pools), else the pool-namespaced chunk id so
    #   unrelated chunks without a source_file stay distinct.
    try:
        rrf_k = int(args.get("rrf_k", 60))
    except (TypeError, ValueError):
        rrf_k = 60
    rrf_k = max(1, min(200, rrf_k))

    pool_weights_arg = args.get("pool_weights") or {}
    if not isinstance(pool_weights_arg, dict):
        pool_weights_arg = {}

    rrf_scores: dict[str, float] = {}
    chunks_by_key: dict[str, dict] = {}
    merged_from: dict[str, list[str]] = {}

    for pname, results in client_results.items():
        weight = float(pool_weights_arg.get(pname, 1.0))
        for rank, chunk in enumerate(results):
            chunk["_source_client"] = pname
            source_file = (chunk.get("metadata") or {}).get("source_file")
            if source_file:
                dedup_key = f"src:{source_file}"
            else:
                dedup_key = f"id:{pname}:{chunk.get('id', rank)}"

            contribution = weight * (1.0 / (rrf_k + rank + 1))
            prev_score = rrf_scores.get(dedup_key, 0.0)
            rrf_scores[dedup_key] = prev_score + contribution

            # Track all pools the dedup_key was seen in (audit trail).
            trail = merged_from.setdefault(dedup_key, [])
            if pname not in trail:
                trail.append(pname)

            # Keep the occurrence with the highest single-occurrence RRF
            # contribution (i.e. the best-ranked / highest-weighted copy).
            existing = chunks_by_key.get(dedup_key)
            if existing is None or contribution > existing.get("_rrf_contribution", 0.0):
                chunk["_rrf_contribution"] = contribution
                chunks_by_key[dedup_key] = chunk

    # Stamp the audit trail onto the kept chunk when more than one pool
    # contributed (no noise for single-pool hits).
    for key, trail in merged_from.items():
        if len(trail) > 1 and key in chunks_by_key:
            kept = chunks_by_key[key]
            md = kept.setdefault("metadata", {})
            md["_merged_from"] = list(trail)

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    merged = [chunks_by_key[k] for k in sorted_keys[:top_k]]

    output = f"Searching {len(client_results)} pools for: *{query}*"
    if mode:
        output += f" (mode: {mode})"
    output += "\n\n**Per-pool quality:**\n"
    output += "\n".join(quality_lines)
    if errors:
        output += "\n" + "\n".join(errors)
    output += "\n\n"
    output += _format_results(merged, include_content=True, max_tokens=max_tokens)
    return output


def handle_find_symbol(args: dict) -> str:
    """Find a specific code symbol by name in the loaded pool."""
    symbol_name = args.get("symbol_name", "")
    pool_name = _resolve_pool_name(args)
    symbol_type = args.get("symbol_type")
    language = args.get("language")
    if not symbol_name:
        return "Error: symbol_name is required."
    if not pool_name:
        return "Error: No pool specified."
    try:
        client, searcher = _load_pool(pool_name)
        results = searcher.search(
            symbol_name, top_k=20, chunk_types=["code"],
            symbol_types=[symbol_type] if symbol_type else None,
            languages=[language] if language else None,
        )
        # Exact name match
        exact = [r for r in results if r.get("metadata", {}).get("symbol_name") == symbol_name]
        if not exact:
            # Partial match
            exact = [r for r in results if symbol_name.lower() in r.get("metadata", {}).get("symbol_name", "").lower()]
        if not exact:
            return f"Symbol '{symbol_name}' not found."
        return _format_results(exact[:10])
    except Exception as e:
        return f"Symbol search error: {e}"


def handle_reindex_pool(args: dict) -> str:
    """Trigger a full reindex of a pool's documents in the background."""
    pool_name = args.get("pool_name", "")
    if not pool_name:
        return "Error: pool_name is required."

    # Check for existing running job
    for job in _jobs.values():
        if job.get("pool") == pool_name and job.get("status") == "running":
            return "Error: Max concurrent ingestion jobs (1) reached. Wait for the current job to complete."

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"pool": pool_name, "status": "running", "progress": 0, "total": 0, "chunks": 0, "started": time.time()}
    _trim_jobs()

    def _run_reindex():
        try:
            from mullet.knowledge.engine.client import load_client
            from mullet.knowledge.engine.ingest import DocumentIngestor, ingest_directory
            from mullet.knowledge.engine.search.searcher import HybridSearcher

            client = load_client(pool_name, clients_dir=_get_pools_dir())
            emb_cfg = client.embedding_config
            ingestor = DocumentIngestor(
                embedding_model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
                passage_prefix=emb_cfg.get("passage_prefix", ""),
            )

            job = _jobs[job_id]
            doc_extensions = client.get_document_extensions()
            exclude = client.documents_config.get("exclude_patterns", [])

            def progress_cb(progress):
                """Copy ingestion progress fields into the live ``_jobs`` entry."""
                job["progress"] = progress.processed_files
                job["total"] = progress.total_files
                job["chunks"] = progress.total_chunks
                job["skipped"] = progress.skipped_files
                job["failed_files"] = progress.failed_files
                job["eta"] = progress.format_eta()
                job["percent"] = progress.percent
                # Also log to stderr for CLI visibility
                import sys
                print(
                    f"\r  [{progress.percent:.1f}%] {progress.processed_files}/{progress.total_files} files, "
                    f"{progress.total_chunks} chunks, ETA: {progress.format_eta()}",
                    end="", flush=True, file=sys.stderr,
                )

            chunks, embeddings = ingest_directory(
                client.documents_path, ingestor,
                extensions=doc_extensions,
                progress_callback=progress_cb,
                exclude_patterns=exclude,
            )

            import sys
            searcher = HybridSearcher(
                index_path=str(client.index_path),
                collection_name=client.machine_name,
                embedding_config=emb_cfg,
                tokenizer_config=client.tokenization_config,
            )
            # Clear existing collection
            try:
                searcher.chroma_client.delete_collection(client.machine_name)
                searcher.collection = searcher.chroma_client.get_or_create_collection(
                    name=client.machine_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as _exc:
                logger.warning("mcp_knowledge: error recreating ChromaDB collection for '%s': %s", pool_name, _exc)

            doc_count = len(chunks) if chunks else 0
            if chunks:
                print(f"\n  Indexing {len(chunks)} doc chunks into ChromaDB...", file=sys.stderr)
                searcher.index_chunks(chunks, embeddings)

            # Also index code if enabled
            code_count = 0
            code_config = client.config.get("code", {})
            if code_config.get("enabled", False):
                print("  Indexing code...", file=sys.stderr)
                try:
                    from mullet.knowledge.engine.ingest import ingest_client_code
                    code_chunks, code_embeddings = ingest_client_code(client, ingestor, rebuild=True)
                    if code_chunks:
                        searcher.index_chunks(code_chunks, code_embeddings)
                        code_count = len(code_chunks)
                except Exception as e:
                    print(f"  Code indexing error: {e}", file=sys.stderr)

            total = doc_count + code_count
            job["chunks"] = total
            if total > 0:
                print(f"  Done — {total} chunks indexed ({doc_count} docs, {code_count} code).", file=sys.stderr)
            else:
                print("\n  No documents or code found to index.", file=sys.stderr)

            # Save file manifest for incremental sync
            from mullet.knowledge.engine.ingest import build_file_manifest, save_file_manifest
            manifest = build_file_manifest(client.documents_path, doc_extensions, exclude_patterns=exclude)
            save_file_manifest(client, manifest)

            job["status"] = "completed"
            _pool_cache.pop(pool_name)
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            import traceback
            traceback.print_exc()

    t = threading.Thread(target=_run_reindex, daemon=True)
    t.start()
    return f"Reindex started for **{pool_name}**. Job ID: `{job_id}`\n\nUse `trawl_ingest_status` to check progress."


def handle_sync_pool(args: dict) -> str:
    """Incremental reindex — only process new/modified files since last index."""
    pool_name = args.get("pool_name", "")
    if not pool_name:
        return "Error: pool_name is required."
    try:
        from mullet.knowledge.engine.client import load_client
        from mullet.knowledge.engine.ingest import (
            DocumentIngestor, build_file_manifest,
            load_file_manifest, save_file_manifest, diff_file_manifest,
        )

        client = load_client(pool_name, clients_dir=_get_pools_dir())
        emb_cfg = client.embedding_config
        extensions = client.get_document_extensions()
        exclude = client.documents_config.get("exclude_patterns", [])

        # Compare current files vs manifest
        current = build_file_manifest(client.documents_path, extensions, exclude_patterns=exclude)
        indexed = load_file_manifest(client)
        new_files, modified_files, deleted_files = diff_file_manifest(current, indexed)

        if not new_files and not modified_files and not deleted_files:
            return f"**{pool_name}** is up to date. No changes detected."

        summary = [f"**{pool_name}** sync: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted"]

        # Ingest new + modified files
        files_to_ingest = new_files + modified_files
        if files_to_ingest:
            ingestor = DocumentIngestor(
                embedding_model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
                passage_prefix=emb_cfg.get("passage_prefix", ""),
            )
            client_obj, searcher = _load_pool(pool_name)
            total_chunks = 0
            for fpath in files_to_ingest:
                try:
                    chunks, embeddings = ingestor.ingest_document(Path(fpath))
                    if chunks:
                        searcher.index_chunks(chunks, embeddings)
                        total_chunks += len(chunks)
                except Exception as e:
                    summary.append(f"  Error: {Path(fpath).name}: {e}")
            summary.append(f"Indexed {total_chunks} chunks from {len(files_to_ingest)} files")
            _pool_cache.pop(pool_name)

        # Save updated manifest
        save_file_manifest(client, current)
        summary.append(f"Manifest updated: {len(current)} files tracked")
        return "\n".join(summary)
    except Exception as e:
        return f"Sync error: {e}"


def handle_ingest_status(args: dict) -> str:
    """Check the status of an ingestion job by ID or pool name."""
    job_id = args.get("job_id")
    pool_name = args.get("pool_name")

    job = None
    if job_id and job_id in _jobs:
        job = _jobs[job_id]
    elif pool_name:
        for jid in reversed(list(_jobs.keys())):
            if _jobs[jid].get("pool") == pool_name:
                job = _jobs[jid]
                job_id = jid
                break

    if not job:
        return "No ingestion job found."

    total = job.get("total", 0)
    progress = job.get("progress", 0)
    chunks = job.get("chunks", 0)
    skipped = job.get("skipped", 0)
    failed = job.get("failed_files", [])
    pct = job.get("percent", (progress / total * 100) if total > 0 else 0)
    eta = job.get("eta", "unknown")
    elapsed = time.time() - job.get("started", time.time())
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60):02d}s" if elapsed >= 60 else f"{int(elapsed)}s"
    job_type = job.get("type", "reindex")

    lines = [
        f"**Job {job_id}** ({job_type} for {job.get('pool', '?')})",
        f"**Status:** {job['status']}",
        f"**Progress:** {pct:.1f}% ({progress}/{total} files)",
        f"**Chunks:** {chunks}",
        f"**Elapsed:** {elapsed_str}",
        f"**ETA:** {eta}",
    ]
    if skipped:
        lines.append(f"**Skipped:** {skipped}")
    if failed:
        lines.append(f"**Failed ({len(failed)}):** {', '.join(str(Path(f).name) for f in failed[:5])}")
    if job.get("error"):
        lines.append(f"**Error:** {job['error']}")
    return "\n".join(lines)


def handle_ingest_file(args: dict) -> str:
    """Ingest a single file into a pool's collection."""
    pool_name = args.get("pool_name", "")
    file_path = args.get("file_path", "")
    if not pool_name or not file_path:
        return "Error: pool_name and file_path are required."
    try:
        client, searcher = _load_pool(pool_name)
        fp = Path(file_path).resolve()
        # Security: validate file exists
        if not fp.exists():
            return f"Error: File not found: {fp}"
        # Security: validate file is under pool's documents path
        docs_path = client.documents_path.resolve()
        try:
            fp.relative_to(docs_path)
        except ValueError:
            return f"Error: File must be under pool's documents directory ({docs_path})"
        start = time.time()
        from mullet.knowledge.engine.ingest import DocumentIngestor
        emb_cfg = client.embedding_config
        ingestor = DocumentIngestor(
            embedding_model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
            passage_prefix=emb_cfg.get("passage_prefix", ""),
        )
        chunks, embeddings = ingestor.ingest_document(fp)
        if chunks:
            searcher.index_chunks(chunks, embeddings)
        elapsed = time.time() - start
        _pool_cache.pop(pool_name)
        return f"Ingested **{fp.name}**: {len(chunks)} chunks in {elapsed:.1f}s"
    except Exception as e:
        return f"Ingest error: {e}"


def handle_ingest_url(args: dict) -> str:
    """Ingest content from a URL into a pool's collection."""
    pool_name = args.get("pool_name", "")
    url = args.get("url", "")
    if not pool_name or not url:
        return "Error: pool_name and url are required."
    if not _is_safe_url(url):
        return "Error: URL must be a public http/https address."
    try:
        client, searcher = _load_pool(pool_name)
        from mullet.knowledge.engine.ingest import DocumentIngestor
        import tempfile
        import urllib.request
        emb_cfg = client.embedding_config
        ingestor = DocumentIngestor(
            embedding_model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
            passage_prefix=emb_cfg.get("passage_prefix", ""),
        )
        # Download URL with size limit
        MAX_SIZE = 50 * 1024 * 1024  # 50MB
        req = urllib.request.Request(url, headers={"User-Agent": "Mullet/1.0"})
        start = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_SIZE:
                return f"Error: Content too large ({int(content_length) // 1024 // 1024}MB > 50MB limit)"
            content_type = resp.headers.get("Content-Type", "text/html")
            content = resp.read(MAX_SIZE + 1)
            if len(content) > MAX_SIZE:
                return "Error: Content exceeds 50MB limit"
        # Determine suffix from content-type
        ct_map = {"text/html": ".html", "application/pdf": ".pdf", "text/plain": ".txt",
                  "text/markdown": ".md", "application/json": ".json"}
        suffix = ct_map.get(content_type.split(";")[0].strip(), Path(url).suffix or ".html")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            chunks, embeddings = ingestor.ingest_document(tmp_path)
            for chunk in chunks:
                chunk["metadata"]["source_url"] = url
            if chunks:
                searcher.index_chunks(chunks, embeddings)
        finally:
            tmp_path.unlink(missing_ok=True)
        elapsed = time.time() - start
        _pool_cache.pop(pool_name)
        return f"Ingested **{url}** into {pool_name}: {len(chunks)} chunks, content-type: {content_type}, time: {elapsed:.1f}s"
    except Exception as e:
        return f"URL ingest error: {e}"


def _get_memory_store(pool_name: str | None = None):
    """Get MemoryStore scoped to client or global."""
    from mullet.knowledge.engine.memory import MemoryStore
    if pool_name:
        return MemoryStore(Path(_get_pools_dir()) / pool_name / "index" / ".memory")
    return MemoryStore(Path(_get_pools_dir()) / ".memory")


def handle_memory_store(args: dict) -> str:
    """Store a memory/note with semantic search capability."""
    key = args.get("key", "")
    content = args.get("content", "")
    namespace = args.get("namespace", "mcp")
    metadata = args.get("metadata", {})
    pool_name = args.get("pool_name")
    ttl = int(args["ttl"]) if args.get("ttl") else None
    if not key or not content:
        return "Error: key and content are required."
    try:
        store = _get_memory_store(pool_name)
        store.store(namespace=namespace, key=key, content=content, metadata=metadata, ttl=ttl)
        scope = f" (pool: {pool_name})" if pool_name else " (global)"
        ttl_str = f", TTL: {ttl}s" if ttl else ""
        return f"Stored memory: **{key}** (namespace: {namespace}){scope}{ttl_str}"
    except Exception as e:
        return f"Memory store error: {e}"


def handle_memory_search(args: dict) -> str:
    """Search stored memories semantically by query text."""
    query = args.get("query", "")
    namespace = args.get("namespace")
    top_k = int(args.get("top_k", 10))
    pool_name = args.get("pool_name")
    if not query:
        return "Error: query is required."
    try:
        store = _get_memory_store(pool_name)
        results = store.search(query, namespace=namespace, top_k=top_k)
        if not results:
            return "No memories found."
        parts = []
        for r in results:
            parts.append(f"- **{r.get('key', '?')}** ({r.get('namespace', 'default')}): {r.get('content', '')[:200]}")
        return "\n".join(parts)
    except Exception as e:
        return f"Memory search error: {e}"


def handle_memory_list(args: dict) -> str:
    """List all stored memories, optionally filtered by namespace."""
    namespace = args.get("namespace")
    pool_name = args.get("pool_name")
    try:
        store = _get_memory_store(pool_name)
        memories = store.list(namespace=namespace)
        if not memories:
            return "No memories stored."
        from datetime import datetime
        lines = ["| Namespace | Key | Content | Updated |", "| --- | --- | --- | --- |"]
        for m in memories:
            ns = m.get('namespace', 'default')
            key = m.get('key', '?')
            content = m.get('content', '')[:80].replace("|", "\\|").replace("\n", " ")
            updated = m.get('updated_at', m.get('created_at', ''))
            if isinstance(updated, (int, float)):
                updated = datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M")
            lines.append(f"| {ns} | **{key}** | {content} | {updated} |")
        scope = f" (pool: {pool_name})" if pool_name else " (global)"
        return f"Memories{scope}:\n\n" + "\n".join(lines)
    except Exception as e:
        return f"Memory list error: {e}"


def handle_memory_delete(args: dict) -> str:
    """Delete a memory by namespace and key."""
    key = args.get("key", "")
    namespace = args.get("namespace", "mcp")
    pool_name = args.get("pool_name")
    if not key:
        return "Error: key is required."
    try:
        store = _get_memory_store(pool_name)
        store.delete(namespace, key)
        return f"Deleted memory: **{key}** (namespace: {namespace})"
    except Exception as e:
        return f"Memory delete error: {e}"


def handle_build_graph(args: dict) -> str:
    """Build or rebuild the knowledge graph for a pool."""
    pool_name = args.get("pool_name", "")
    if not pool_name:
        return "Error: pool_name is required."
    rebuild = args.get("rebuild", False)
    sample = int(args["sample"]) if args.get("sample") else None
    workers_override = int(args["workers"]) if args.get("workers") else None
    allowed_providers = _normalize_allowed_providers(args.get("providers"))

    # Check for existing running graph job
    for job in _jobs.values():
        if job.get("pool") == pool_name and job.get("type") == "graph" and job.get("status") == "running":
            return "Error: Graph build already running for this pool."

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"pool": pool_name, "status": "running", "type": "graph", "started": time.time()}
    _trim_jobs()

    def _run_build():
        try:
            import random as _random
            import chromadb
            from mullet.knowledge.engine.client import load_client
            from mullet.knowledge.graph.store import KnowledgeGraph
            from mullet.knowledge.graph.extractor import GraphExtractor

            client = load_client(pool_name, clients_dir=_get_pools_dir())
            graph_config = client.graph_config

            # Load or clear graph
            graph = KnowledgeGraph(client.index_path)
            if rebuild:
                graph.clear()
            else:
                graph.load()

            # Get chunks from ChromaDB
            try:
                from chromadb.config import Settings
                chroma = chromadb.PersistentClient(
                    path=str(client.index_path),
                    settings=Settings(anonymized_telemetry=False),
                )
            except Exception:
                chroma = chromadb.PersistentClient(path=str(client.index_path))
            try:
                collection = chroma.get_collection(name=client.machine_name)
            except Exception:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = "No collection found"
                return

            count = collection.count()
            if count == 0:
                _jobs[job_id]["status"] = "completed"
                return

            # Incremental: only process unprocessed chunks
            processed_ids = set() if rebuild else graph.processed_chunk_ids
            exclude_exts = graph_config.get("exclude_extensions", [])
            unprocessed_ids = []

            for offset in range(0, count, 10000):
                results = collection.get(include=["metadatas"], limit=10000, offset=offset)
                for i, chunk_id in enumerate(results["ids"]):
                    md = results["metadatas"][i] if results["metadatas"] else {}
                    if md.get("chunk_type", "document") != "document":
                        continue
                    if exclude_exts:
                        src = (md.get("source_path") or md.get("source") or "").lower()
                        if any(src.endswith(ext.lower()) for ext in exclude_exts):
                            continue
                    if chunk_id not in processed_ids:
                        unprocessed_ids.append(chunk_id)

            # Fetch content for unprocessed chunks
            all_chunks = []
            for i in range(0, len(unprocessed_ids), 5000):
                batch_ids = unprocessed_ids[i:i + 5000]
                results = collection.get(ids=batch_ids, include=["documents", "metadatas"])
                for j in range(len(results["ids"])):
                    all_chunks.append({
                        "id": results["ids"][j],
                        "content": results["documents"][j],
                        "metadata": results["metadatas"][j] if results["metadatas"] else {},
                    })

            if sample and sample < len(all_chunks):
                all_chunks = _random.sample(all_chunks, sample)

            if not all_chunks:
                _jobs[job_id]["status"] = "completed"
                return

            # Release embedding model from VRAM before graph extraction starts.
            # Graph extraction uses LM Studio (a separate process) — the embedding
            # model and LM Studio fight for the same GPU memory if both are loaded.
            # The embedder will reload automatically on the next search request.
            try:
                import gc
                from mullet.knowledge.engine.model_cache import get_model_cache
                get_model_cache().clear()
                gc.collect()
                logger.info("Graph build: released embedding model from VRAM")
            except Exception as _exc:
                logger.warning("mcp_knowledge: error releasing embedding model from VRAM: %s", _exc)

            # Create LLM provider (honour caller allowlist)
            try:
                provider = _create_llm_provider(client, allowed_providers=allowed_providers)
                if provider is None:
                    _jobs[job_id]["status"] = "failed"
                    _jobs[job_id]["error"] = "No LLM provider available"
                    return
            except ValueError as _ve:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(_ve)
                return
            except Exception:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = "No LLM provider available"
                return

            def get_llm_response(messages, system_prompt, max_tokens):
                """Send a deterministic (temperature=0) chat call for graph-extraction."""
                full_messages = [{"role": "system", "content": system_prompt}] + messages if system_prompt else messages
                # temperature=0.0: deterministic JSON extraction; reduces SPECIAL_* token spam
                return provider.create(full_messages, max_tokens=max_tokens, temperature=0.0).content

            # Default 1 worker: graph extraction is GPU-bound, not I/O-bound.
            # Multiple concurrent requests compete for VRAM/CPU and cause tokenizer
            # corruption (SPECIAL_* spam) when running alongside other processes.
            num_workers = workers_override or graph_config.get("extract_workers", 1)
            save_every = graph_config.get("extract_save_every", 5)

            extractor = GraphExtractor(
                get_llm_response=get_llm_response,
                entity_types=graph_config.get("entity_types", ["person", "organization", "concept", "location"]),
                max_description_chars=graph_config.get("max_description_chars", 200),
                max_workers=num_workers,
                save_every=save_every,
                batch_text_budget=graph_config.get("extract_text_budget", 6000),
                inter_batch_sleep=graph_config.get("inter_batch_sleep", 0.0),
            )

            # Progress callback with periodic saves
            _graph_lock = threading.Lock()
            _last_entity_count = [0]
            _last_rel_count = [0]

            def _on_progress(progress, entities, relationships):
                with _graph_lock:
                    new_entities = entities[_last_entity_count[0]:]
                    new_rels = relationships[_last_rel_count[0]:]
                    _last_entity_count[0] = len(entities)
                    _last_rel_count[0] = len(relationships)
                    for entity in new_entities:
                        graph.add_entity(entity)
                    for rel in new_rels:
                        graph.add_relationship(rel)
                    graph.mark_chunks_processed([c["id"] for c in all_chunks[:progress.processed_chunks]])
                    graph.save()

            entities, relationships = extractor.extract_from_chunks(
                all_chunks, progress_callback=_on_progress
            )

            # Final save
            with _graph_lock:
                new_entities = entities[_last_entity_count[0]:]
                new_rels = relationships[_last_rel_count[0]:]
                for entity in new_entities:
                    graph.add_entity(entity)
                for rel in new_rels:
                    graph.add_relationship(rel)
                graph.mark_chunks_processed([c["id"] for c in all_chunks])
                graph.save()

            graph.close()
            try:
                _pool_meta.mark_indexed(_get_pools_dir() / pool_name)
            except Exception as _exc:
                logger.warning("mcp_knowledge: error marking pool '%s' as indexed: %s", pool_name, _exc)
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["chunks"] = len(all_chunks)

        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            import traceback
            traceback.print_exc()

    t = threading.Thread(target=_run_build, daemon=True)
    t.start()
    mode = "rebuild" if rebuild else "incremental"
    sample_str = f" (sample={sample})" if sample else ""
    return f"Graph build started for **{pool_name}** ({mode}){sample_str}. Job ID: `{job_id}`\n\nUse `trawl_ingest_status` to check progress."


def handle_graph_stats(args: dict) -> str:
    """Get knowledge graph statistics for a pool."""
    pool_name = _resolve_pool_name(args)
    if not pool_name:
        return "Error: pool_name is required."
    try:
        client, searcher = _load_pool(pool_name)
        graph = searcher.load_graph()
        if not graph:
            return f"No knowledge graph found for **{pool_name}**. Use trawl_build_graph first."

        from mullet.knowledge.graph.export import graph_stats
        s = graph_stats(graph)

        lines = [f"Knowledge Graph Stats for **{pool_name}**:\n"]
        lines.append(f"**Entities:** {s['entity_count']}")
        lines.append(f"**Relationships:** {s['relationship_count']}")
        lines.append(f"**Source Chunks:** {s['source_chunk_count']}")
        lines.append(f"**Density:** {s['density']:.6f}")

        if s.get("entity_type_distribution"):
            lines.append("\n**Entity Types:**")
            for etype, count in sorted(s["entity_type_distribution"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {etype}: {count}")

        if s.get("top_entities"):
            lines.append("\n**Most Connected Entities:**")
            for te in s["top_entities"]:
                lines.append(f"  - **{te['name']}** ({te['type']}): {te['connections']} connections")

        return "\n".join(lines)
    except Exception as e:
        return f"Graph stats error: {e}"


def handle_graph_prune(args: dict) -> str:
    """Remove orphaned graph entities and relationships."""
    pool_name = _resolve_pool_name(args)
    if not pool_name:
        return "Error: pool_name is required."
    try:
        client, searcher = _load_pool(pool_name)
        graph = searcher.load_graph()
        if not graph:
            return f"No knowledge graph found for **{pool_name}**."

        existing_ids: set[str] = set()
        count = searcher.collection.count()
        for off in range(0, count, 10000):
            r = searcher.collection.get(include=[], limit=10000, offset=off)
            existing_ids.update(r["ids"])
        pruned = graph.prune_orphaned_chunks(existing_ids)
        if pruned["removed_chunks"] or pruned["removed_entities"] or pruned["removed_relationships"]:
            graph.save()
        return (
            f"Pruned graph for **{pool_name}**:\n"
            f"- Removed chunks: {pruned['removed_chunks']}\n"
            f"- Removed entities: {pruned['removed_entities']}\n"
            f"- Removed relationships: {pruned['removed_relationships']}\n"
            f"- Updated entities: {pruned['updated_entities']}\n"
            f"- Updated relationships: {pruned['updated_relationships']}"
        )
    except Exception as e:
        return f"Graph prune error: {e}"


def handle_search_entities(args: dict) -> str:
    """Search for entities in the knowledge graph by name with optional type filtering."""
    query = args.get("query", "")
    pool_name = _resolve_pool_name(args)
    entity_type = args.get("entity_type")
    top_k = min(int(args.get("top_k", 20)), 100)
    if not query:
        return "Error: query is required."
    if not pool_name:
        return "Error: No pool loaded. Use trawl_load_pool first."
    try:
        client, searcher = _load_pool(pool_name)
        graph = searcher.load_graph()
        if not graph:
            return f"No knowledge graph found for **{pool_name}**. Use trawl_build_graph first."

        entities = graph.find_entities_by_name(query)
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type.lower().strip()]
        entities = entities[:top_k]

        if not entities:
            return f"No entities matching '{query}' in **{pool_name}** graph."

        lines = [f"Found {len(entities)} entities matching '{query}':\n"]
        for e in entities:
            aliases_str = f" (aliases: {', '.join(e.aliases)})" if e.aliases else ""
            chunks_str = f" [{len(e.source_chunk_ids)} chunks]"
            lines.append(f"- **{e.name}** ({e.entity_type}){aliases_str}{chunks_str}")
            if e.description:
                lines.append(f"  {e.description}")

        return "\n".join(lines)
    except Exception as e:
        return f"Entity search error: {e}"


def handle_search_relationships(args: dict) -> str:
    """Search for relationships of an entity in the knowledge graph up to a max depth."""
    entity_name = args.get("entity", "")
    pool_name = _resolve_pool_name(args)
    relationship_type = args.get("relationship_type")
    max_hops = min(int(args.get("max_hops", 2)), 4)
    if not entity_name:
        return "Error: entity is required."
    if not pool_name:
        return "Error: No pool loaded. Use trawl_load_pool first."
    try:
        client, searcher = _load_pool(pool_name)
        graph = searcher.load_graph()
        if not graph:
            return f"No knowledge graph found for **{pool_name}**. Use trawl_build_graph first."

        entities = graph.find_entities_by_name(entity_name)
        if not entities:
            return f"No entity matching '{entity_name}' found."

        seed = entities[0]
        neighbors = graph.get_neighbors(seed.key, max_depth=max_hops)

        if relationship_type:
            rt = relationship_type.lower().strip()
            neighbors = [(e, r) for e, r in neighbors if r.relationship_type.lower().strip() == rt]

        if not neighbors:
            return f"No relationships found for entity '{seed.name}' (hops={max_hops})."

        lines = [f"Relationships from **{seed.name}** ({seed.entity_type}, depth={max_hops}):\n"]
        for entity, rel in neighbors:
            direction = "outgoing" if rel.source_entity == seed.key else "incoming"
            lines.append(f"- **{entity.name}** ({entity.entity_type}) [{rel.relationship_type}] ({direction})")
            if rel.description:
                lines.append(f"  {rel.description}")
            lines.append(f"  weight: {rel.weight:.2f}, chunks: {len(rel.source_chunk_ids)}")

        return "\n".join(lines)
    except Exception as e:
        return f"Relationship search error: {e}"


def handle_list_sources(args: dict) -> str:
    """List document sources for a pool including GitHub and web sources."""
    pool_name = _resolve_pool_name(args)
    if not pool_name:
        return "Error: pool_name is required."
    try:
        client, _ = _load_pool(pool_name)
        sources_config = client.sources_config if hasattr(client, 'sources_config') else {}
        sources_dir = client.sources_path if hasattr(client, 'sources_path') else client.path / "sources"
        output = f"Sources for **{pool_name}**:\n\n"

        github_sources = sources_config.get("github", []) if sources_config else []
        web_sources = sources_config.get("web", []) if sources_config else []

        if not github_sources and not web_sources:
            # Check disk for unconfigured sources
            if sources_dir.exists():
                github_dir = sources_dir / "github"
                web_dir = sources_dir / "web"
                if github_dir.exists():
                    repos = [d.name for d in github_dir.iterdir() if d.is_dir()]
                    output += f"**GitHub** (unconfigured): {len(repos)} directories\n"
                if web_dir.exists():
                    domains = [d.name for d in web_dir.iterdir() if d.is_dir()]
                    output += f"**Web** (unconfigured): {len(domains)} domains\n"
                if not github_dir.exists() and not web_dir.exists():
                    output += "No sources configured.\n"
            else:
                # Fall back to documents dir listing
                docs_dir = client.documents_path
                if docs_dir.exists():
                    sources = [d.name for d in docs_dir.iterdir() if d.is_dir()]
                    output += "\n".join(f"- {s}" for s in sorted(sources)) if sources else "No sources found."
                else:
                    output += "No sources configured.\n"
        else:
            if github_sources:
                output += "**GitHub Sources:**\n"
                for gh in github_sources:
                    owner = gh.get("owner", "?")
                    repo = gh.get("repo", "")
                    has_token = bool(gh.get("token_env"))
                    desc = f"{owner}/{repo}" if repo else f"{owner} (all repos)"
                    auth = "(auth)" if has_token else "(public)"
                    output += f"  - {desc} {auth}\n"
            if web_sources:
                output += "\n**Web Sources:**\n"
                for web in web_sources:
                    url = web.get("url", "?")
                    mode = web.get("mode", "crawl")
                    output += f"  - {url} [mode: {mode}]\n"

        return output
    except Exception as e:
        return f"List sources error: {e}"


def handle_sync_sources(args: dict) -> str:
    """Sync all sources for a pool by pulling latest changes from remote."""
    pool_name = args.get("pool_name", "")
    if not pool_name:
        return "Error: pool_name is required."
    if not (_get_pools_dir() / pool_name).exists():
        return f"Error: Pool '{pool_name}' does not exist."
    try:
        client, _ = _load_pool(pool_name)
        sources_config = client.sources_config if hasattr(client, 'sources_config') else {}

        if not sources_config:
            # Fall back to git pull on documents dir
            docs_dir = client.documents_path
            synced = 0
            for source_dir in docs_dir.iterdir():
                if source_dir.is_dir() and (source_dir / ".git").exists():
                    import subprocess
                    subprocess.run(["git", "pull"], cwd=str(source_dir), capture_output=True, timeout=60)
                    synced += 1
            return f"Synced {synced} git sources for **{pool_name}**."

        all_results = []
        total_added = total_updated = total_errors = 0

        # Sync GitHub sources
        github_sources = sources_config.get("github", [])
        if github_sources:
            try:
                from mullet.knowledge.sources.github_connector import GitHubSourceConnector
                from mullet.knowledge.sources.github_models import GitHubSourceConfig
                sources_dir = str(client.documents_path)
                for gh_config in github_sources:
                    config = GitHubSourceConfig.from_dict(gh_config) if hasattr(GitHubSourceConfig, 'from_dict') else GitHubSourceConfig(**gh_config)
                    connector = GitHubSourceConnector(config)
                    result = connector.sync(sources_dir)
                    status = "OK" if result.success else f"{len(result.errors)} errors"
                    all_results.append(f"- [github] {config.owner}: +{result.files_added} added, ~{result.files_updated} updated ({status})")
                    total_added += result.files_added
                    total_updated += result.files_updated
                    total_errors += len(result.errors)
            except ImportError:
                all_results.append("- [github] Connector not available")

        # Sync web sources
        web_sources = sources_config.get("web", [])
        if web_sources:
            try:
                from mullet.knowledge.sources.web_crawler import WebSourceConfig, WebSourceConnector
                sources_dir = str(client.documents_path)
                for web_config in web_sources:
                    config = WebSourceConfig.from_dict(web_config) if hasattr(WebSourceConfig, 'from_dict') else WebSourceConfig(**web_config)
                    connector = WebSourceConnector(config)
                    result = connector.sync(sources_dir)
                    status = "OK" if result.success else f"{len(result.errors)} errors"
                    all_results.append(f"- [web] {config.domain}: +{result.files_added} added, ~{result.files_updated} updated ({status})")
                    total_added += result.files_added
                    total_updated += result.files_updated
                    total_errors += len(result.errors)
            except ImportError:
                all_results.append("- [web] Connector not available")

        output = f"Source sync complete for **{pool_name}**\n\n"
        output += "\n".join(all_results)
        output += f"\n\n**Totals:** {total_added} added, {total_updated} updated, {total_errors} errors"
        return output
    except Exception as e:
        return f"Sync error: {e}"


def handle_connect_github(args: dict) -> str:
    """Connect a GitHub owner or repo as a document source for a pool."""
    pool_name = args.get("pool_name", "")
    owner = args.get("owner", "")
    repo = args.get("repo", "")
    token_env = args.get("token_env", "GITHUB_TOKEN")
    include_repos = args.get("include_repos", [])
    exclude_repos = args.get("exclude_repos", [])
    if not pool_name or not owner:
        return "Error: pool_name and owner are required."
    if not _SAFE_NAME_RE.match(owner):
        return "Error: Invalid owner name."
    if repo and not _SAFE_NAME_RE.match(repo):
        return "Error: Invalid repo name."
    pool_dir = _get_pools_dir() / pool_name
    if not pool_dir.exists():
        return f"Error: Pool '{pool_name}' does not exist. Create it first with `mullet pool create`."
    try:
        client, _ = _load_pool(pool_name)
        # Try proper connector first
        try:
            from mullet.knowledge.sources.github_connector import GitHubSourceConnector
            from mullet.knowledge.sources.github_models import GitHubSourceConfig
            config = GitHubSourceConfig(
                owner=owner, repo=repo, token_env=token_env,
                include_repos=include_repos, exclude_repos=exclude_repos,
            )
            connector = GitHubSourceConnector(config)
            sources_dir = str(client.documents_path)
            result = connector.sync(sources_dir)
            output = f"GitHub source connected for **{pool_name}**\n\n"
            output += f"- Owner: {owner}\n"
            if repo:
                output += f"- Repo: {repo}\n"
            output += f"- Files added: {result.files_added}\n"
            output += f"- Files updated: {result.files_updated}\n"
            if result.errors:
                output += f"- Errors ({len(result.errors)}):\n"
                for err in result.errors[:10]:
                    output += f"  - {err}\n"
            return output
        except ImportError:
            # Fallback to git clone
            import subprocess
            if not repo:
                return "Error: Please specify a repo. Org-level discovery requires the GitHub connector."
            docs_dir = Path(client.path) / "documents"
            docs_dir.mkdir(parents=True, exist_ok=True)
            token = os.environ.get(token_env, "")
            if token:
                url = f"https://{token}@github.com/{owner}/{repo}.git"
            else:
                url = f"https://github.com/{owner}/{repo}.git"
            target = (docs_dir / repo).resolve()
            if not str(target).startswith(str(docs_dir.resolve())):
                return "Error: Invalid target path."
            if target.exists():
                subprocess.run(["git", "pull"], cwd=str(target), capture_output=True, timeout=120)
                return f"Updated **{owner}/{repo}** in {pool_name}."
            else:
                subprocess.run(["git", "clone", url, str(target)], capture_output=True, timeout=300)
                return f"Cloned **{owner}/{repo}** into {pool_name}."
    except Exception as e:
        return f"GitHub connect error: {e}"


def handle_crawl_web(args: dict) -> str:
    """Crawl a public website and save pages as Markdown into a pool."""
    pool_name = args.get("pool_name", "")
    url = args.get("url", "")
    max_pages = int(args.get("max_pages", 100))
    if not pool_name or not url:
        return "Error: pool_name and url are required."
    if not _is_safe_url(url):
        return "Error: URL must be a public http/https address."
    try:
        from mullet.knowledge.engine.ingest import DocumentIngestor
        client, _ = _load_pool(pool_name)
        ingestor = DocumentIngestor(client)
        stay_in_path = args.get("stay_in_path", True)
        pages = ingestor.crawl_url(url, max_pages=max_pages, stay_in_path=stay_in_path)
        return f"Crawled **{url}**: {pages} pages saved for {pool_name}."
    except Exception as e:
        return f"Crawl error: {e}"


def _normalize_allowed_providers(raw):
    """Parse a providers allowlist from MCP args into a frozenset or None."""
    if raw is None:
        return None
    if isinstance(raw, str):
        allowed = frozenset(p.strip().lower() for p in raw.split(",") if p.strip())
    elif isinstance(raw, (list, tuple, set, frozenset)):
        allowed = frozenset(str(p).strip().lower() for p in raw if str(p).strip())
    else:
        return None
    return allowed or None


def _pool_provider_key(provider_type):
    pt = (provider_type or "").lower()
    if pt in ("lmstudio", "lmstudio_native"):
        return "local"
    if pt == "claude":
        return "haiku"
    return pt


def _build_local_provider(llm_config):
    from mullet.knowledge.engine.llm import LMStudioProvider
    from mullet.core.env import local_llm_host
    host = llm_config.get("lmstudio_host") or local_llm_host()
    model = llm_config.get("lmstudio_model", "default")
    return LMStudioProvider(base_url=f"{host}/v1", model=model)


def _build_anthropic_provider():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    from mullet.core.providers.anthropic import AnthropicProvider
    return AnthropicProvider(api_key=api_key)


def _create_llm_provider(client, allowed_providers=None):
    """Create an LLM provider from pool config, honouring an optional allowlist.

    If allowed_providers is given and the pool pick is not in it, fall back
    to the first allowlisted provider that can serve. Raises ValueError if
    none can. Returns None only on unexpected construction errors.
    """
    llm_config = getattr(client, "llm_config", {}) or {}
    provider_type = llm_config.get("provider", "lmstudio")
    pool_key = _pool_provider_key(provider_type)

    if allowed_providers is not None and pool_key not in allowed_providers:
        for key in allowed_providers:
            try:
                if key == "local":
                    return _build_local_provider(llm_config)
                if key == "haiku":
                    prov = _build_anthropic_provider()
                    if prov is not None:
                        return prov
            except Exception as e:
                logger.warning(f"Knowledge LLM fallback to {key} failed: {e}")
                continue
        raise ValueError(
            f"Knowledge LLM pick blocked by allowlist {sorted(allowed_providers)}; "
            f"pool configured for {provider_type!r}"
        )

    try:
        if pool_key == "local":
            return _build_local_provider(llm_config)
        if pool_key == "haiku":
            prov = _build_anthropic_provider()
            if prov is not None:
                return prov
        return _build_local_provider(llm_config)
    except Exception as e:
        logger.warning(f"Failed to create LLM provider: {e}")
        return None


def _summarize_chunk(content: str, query: str, provider) -> str:
    """Summarize a chunk. Falls back to truncation if no LLM."""
    if provider:
        try:
            messages = [{"role": "user", "content": (
                f"Summarize the following text in 2-3 sentences, "
                f"focusing on aspects relevant to the query '{query}':\n\n{content[:2000]}"
            )}]
            return provider.create(messages, max_tokens=150).content
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
    if len(content) > 200:
        return content[:200] + "..."
    return content


def handle_summarize(args: dict) -> str:
    """Search a pool and return LLM-generated concise summaries of results."""
    query = args.get("query", "")
    if not query:
        return "Error: query is required."
    top_k = int(args.get("top_k", 5))
    pool_name = _resolve_pool_name(args)
    if not pool_name:
        return "Error: No pool loaded. Use trawl_load_pool first or specify pool_name."
    allowed = _normalize_allowed_providers(args.get("providers"))
    try:
        client, searcher = _load_pool(pool_name)
        search_result = searcher.search_with_quality(query=query, top_k=top_k)
        provider = _create_llm_provider(client, allowed_providers=allowed)
        lines = [f"Summary of results for: *{query}* (from **{client.name}**)\n"]
        for i, chunk in enumerate(search_result.results, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source_file", "unknown")
            score = chunk.get("_rerank_score", chunk.get("_combined_score", 0))
            content = chunk.get("content", "")
            summary = _summarize_chunk(content, query, provider)
            lines.append(f"**{i}. {source}** (score: {score:.3f})")
            lines.append(f"   {summary}")
            lines.append("")
        if not search_result.results:
            lines.append("No results found.")
        return "\n".join(lines)
    except Exception as e:
        return f"Summarize error: {e}"


def handle_list_repos(args: dict) -> str:
    """List GitHub repositories indexed as sources for a pool."""
    return handle_list_sources(args)


# ============================================================================
# Missing tools — ported from TRAG MCP server
# ============================================================================


def handle_search_repo(args: dict) -> str:
    """Search within a specific repository in a pool."""
    repo_name = args.get("repo_name", "")
    query = args.get("query", "")
    pool_name = _resolve_pool_name(args)
    top_k = int(args.get("top_k", 20))
    chunk_types = args.get("chunk_types", [])
    if not repo_name or not query:
        return "Error: repo_name and query are required."
    if not pool_name:
        return "Error: No pool specified."
    try:
        client, searcher = _load_pool(pool_name)
        results = searcher.search(query, top_k=top_k * 3, chunk_types=chunk_types if chunk_types else None)
        results = [r for r in results if r.get("metadata", {}).get("repo_name") == repo_name]
        return _format_results(results[:top_k])
    except Exception as e:
        return f"Repo search error: {e}"


def handle_graph_export(args: dict) -> str:
    """Export the knowledge graph as D3.js-compatible JSON."""
    pool_name = _resolve_pool_name(args)
    max_nodes = int(args.get("max_nodes", 500))
    if not pool_name:
        return "Error: pool_name is required."
    try:
        client, searcher = _load_pool(pool_name)
        graph = searcher.load_graph()
        if not graph:
            return f"No knowledge graph found for **{pool_name}**. Use trawl_build_graph first."

        from mullet.knowledge.graph.export import export_for_d3
        d3_data = export_for_d3(graph, max_nodes=max_nodes)
        return json.dumps(d3_data, indent=2)
    except Exception as e:
        return f"Graph export error: {e}"


def _get_workflow_engine(pool_name: str | None = None):
    """Get WorkflowEngine for a pool (or global)."""
    from mullet.knowledge.engine.workflow import WorkflowEngine
    db_path = Path(_get_pools_dir()) / (pool_name or ".global") / "workflows.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return WorkflowEngine(db_path)


def handle_workflow_create(args: dict) -> str:
    """Create a workflow with named steps and dependency ordering."""
    name = args.get("name", "")
    steps = args.get("steps", [])
    pool_name = args.get("pool_name")
    if not name or not steps:
        return "Error: name and steps are required."
    try:
        engine = _get_workflow_engine(pool_name)
        workflow_id = engine.create(name, steps)
        wf = engine.get(workflow_id)
        next_ready = engine.next_steps(workflow_id)

        lines = [f"Created workflow **{name}**", f"ID: `{workflow_id}`", f"Steps: {len(wf['steps'])}", ""]
        for s in wf["steps"]:
            deps = ""
            if s.get("depends_on"):
                dep_names = []
                for dep_id in s["depends_on"]:
                    dep_step = next((st for st in wf["steps"] if st["id"] == dep_id), None)
                    if dep_step:
                        dep_names.append(dep_step["name"])
                deps = f" (after: {', '.join(dep_names)})"
            lines.append(f"  {s.get('sort_order', 0)+1}. **{s['name']}**{deps}")
            lines.append(f"     ID: `{s['id']}`")
            if s.get("description"):
                lines.append(f"     {s['description']}")
        if next_ready:
            lines.append(f"\nReady to start: {', '.join(s['name'] for s in next_ready)}")
        return "\n".join(lines)
    except ValueError as e:
        return f"Validation error: {e}"
    except Exception as e:
        return f"Workflow create error: {e}"


def handle_workflow_status(args: dict) -> str:
    """Get workflow status and step progress."""
    workflow_id = args.get("workflow_id", "")
    pool_name = args.get("pool_name")
    if not workflow_id:
        return "Error: workflow_id is required."
    try:
        engine = _get_workflow_engine(pool_name)
        wf = engine.get(workflow_id)
        if not wf:
            return f"Workflow `{workflow_id}` not found."
        next_ready = engine.next_steps(workflow_id)

        icons = {"pending": "[ ]", "running": "[~]", "completed": "[x]", "failed": "[!]", "skipped": "[-]"}
        lines = [f"**{wf['name']}** ({wf['status']})", f"ID: `{wf['id']}`\n"]
        for s in wf["steps"]:
            icon = icons.get(s["status"], "[ ]")
            lines.append(f"{icon} **{s['name']}** — {s['status']}")
            lines.append(f"    ID: `{s['id']}`")
            if s.get("result"):
                lines.append(f"    Result: {s['result']}")
        if next_ready:
            lines.append(f"\nReady to start: {', '.join(s['name'] for s in next_ready)}")
        return "\n".join(lines)
    except Exception as e:
        return f"Workflow status error: {e}"


def handle_workflow_update(args: dict) -> str:
    """Update a workflow step's status."""
    workflow_id = args.get("workflow_id", "")
    step_id = args.get("step_id", "")
    status = args.get("status", "")
    result = args.get("result")
    pool_name = args.get("pool_name")
    if not workflow_id or not step_id or not status:
        return "Error: workflow_id, step_id, and status are required."
    try:
        engine = _get_workflow_engine(pool_name)
        updated = engine.update_step(workflow_id, step_id, status, result=result)
        if not updated:
            return f"Step `{step_id}` not found in workflow `{workflow_id}`."
        wf = engine.get(workflow_id)
        next_ready = engine.next_steps(workflow_id)
        text = f"Updated step to **{status}**. Workflow is now **{wf['status']}**."
        if next_ready:
            text += f"\nNext steps ready: {', '.join(s['name'] for s in next_ready)}"
        return text
    except ValueError as e:
        return f"Validation error: {e}"
    except Exception as e:
        return f"Workflow update error: {e}"


def handle_workflow_list(args: dict) -> str:
    """List all workflows, optionally filtered by status."""
    pool_name = args.get("pool_name")
    status_filter = args.get("status")
    try:
        engine = _get_workflow_engine(pool_name)
        workflows = engine.list(status=status_filter)
        if not workflows:
            return "No workflows found."
        lines = [f"**Workflows** ({len(workflows)} total):\n"]
        for wf in workflows:
            progress = f"{wf.get('completed_steps', 0)}/{wf.get('total_steps', 0)} steps"
            lines.append(f"- **{wf['name']}** ({wf['status']}) — {progress}")
            lines.append(f"  ID: `{wf['id']}`")
        return "\n".join(lines)
    except Exception as e:
        return f"Workflow list error: {e}"


def handle_agents_list(args: dict) -> str:
    """List available agent templates, optionally filtered by tag."""
    tag = args.get("tag")
    try:
        from mullet.knowledge.engine.agents import AgentLoader
        loader = AgentLoader(Path(__file__).parent.parent / "agents")
        agents = loader.list(tag=tag)
        if not agents:
            return "No agents found."
        parts = []
        for a in agents:
            tags_str = f" [{', '.join(a.get('tags', []))}]" if a.get('tags') else ""
            parts.append(f"- **{a['name']}**{tags_str}: {a.get('description', '')}")
        return "\n".join(parts)
    except Exception as e:
        return f"Agent list error: {e}"


def handle_agents_get(args: dict) -> str:
    """Get full agent template config including system prompt."""
    agent_name = args.get("agent_name", "")
    if not agent_name:
        return "Error: agent_name is required."
    try:
        from mullet.knowledge.engine.agents import AgentLoader
        loader = AgentLoader(Path(__file__).parent.parent / "agents")
        agent = loader.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found."
        parts = [f"**{agent['name']}**"]
        if agent.get("description"):
            parts.append(f"Description: {agent['description']}")
        if agent.get("tags"):
            parts.append(f"Tags: {', '.join(agent['tags'])}")
        if agent.get("system_prompt"):
            parts.append(f"\nSystem Prompt:\n```\n{agent['system_prompt']}\n```")
        return "\n".join(parts)
    except Exception as e:
        return f"Agent get error: {e}"


def handle_run_agent(args: dict) -> str:
    """Run a Trove agent with local LLM — searches context, applies agent prompt, generates response."""
    agent_name = args.get("agent_name", "")
    query = args.get("query", "")
    pool_name = _resolve_pool_name(args)
    top_k = int(args.get("top_k", 10))
    store_memory = args.get("store_memory", False)
    if not agent_name or not query:
        return "Error: agent_name and query are required."
    allowed = _normalize_allowed_providers(args.get("providers"))
    try:
        from mullet.knowledge.engine.agents import AgentLoader
        from mullet.knowledge.engine.runner import AgentRunner
        loader = AgentLoader(Path(__file__).parent.parent / "agents")
        agent = loader.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found."
        if not pool_name:
            return "Error: No pool loaded. Specify pool_name or load a pool first."
        client, searcher = _load_pool(pool_name)
        # Create LLM provider (honour caller allowlist)
        try:
            provider = _create_llm_provider(client, allowed_providers=allowed)
        except ValueError as _ve:
            return f"Error: {_ve}"
        if not provider:
            return "Error: No LLM provider available. Check pool LLM config."
        # Optional memory store
        memory_store = None
        if store_memory:
            from mullet.knowledge.engine.memory import MemoryStore
            memory_store = MemoryStore(client.index_path / "trawl_state.db")
        runner = AgentRunner(
            agent=agent,
            client=client,
            llm_provider=provider,
            searcher=searcher,
            memory_store=memory_store,
        )
        result = runner.run(query=query, top_k=top_k, store_memory=store_memory)
        # Format response matching TRAG
        lines = [
            f"**Agent: {result.agent_name}**",
            f"*Query: {result.query}*",
            f"Sources: {len(result.sources)} | "
            f"Search: {result.timing.get('search_ms', 0)}ms | "
            f"Generate: {result.timing.get('generate_ms', 0)}ms",
            "",
            result.response,
        ]
        if result.sources:
            lines.append("\n---\n**Sources:**")
            seen = set()
            for s in result.sources[:10]:
                src = s["source"]
                if src not in seen:
                    seen.add(src)
                    lines.append(f"- {src} (score: {s['score']:.3f})")
        return "\n".join(lines)
    except Exception as e:
        return f"Agent run error: {e}"


def handle_prompts_search_live(args: dict) -> str:
    """Search the prompts.chat API for prompt templates."""
    query = args.get("query", "")
    if not query:
        return "Error: query is required."
    try:
        import urllib.request
        import urllib.parse
        params = urllib.parse.urlencode({"q": query, "limit": args.get("limit", 10)})
        url = f"https://api.prompts.chat/api/prompts/search?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mullet/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = data.get("results", data) if isinstance(data, dict) else data
        if not results:
            return "No prompts found."
        parts = []
        for p in (results[:10] if isinstance(results, list) else []):
            title = p.get("title", "Untitled")
            parts.append(f"- **{title}** (ID: {p.get('id', '?')})")
        return "\n".join(parts) if parts else "No prompts found."
    except Exception as e:
        return f"Prompts search error: {e}"


def handle_prompts_find(args: dict) -> str:
    """Hybrid search: local prompts pool + live prompts.chat API with deduplication."""
    query = args.get("query", "")
    if not query:
        return "Error: query is required."
    top_k = min(int(args.get("top_k", 10)), 50)
    half_k = max(top_k // 2, 3)

    # Local pool search
    local_results = []
    try:
        client, searcher = _load_pool("prompts")
        sr = searcher.search_with_quality(query=query, top_k=half_k)
        for chunk in sr.results:
            content = chunk.get("content", chunk.get("text", ""))
            source = chunk.get("metadata", {}).get("source", "")
            title = chunk.get("metadata", {}).get("title", "")
            if not title:
                for line in content.split("\n"):
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
            local_results.append({
                "source": "local",
                "title": title or Path(source).stem.replace("_", " ") if source else "Untitled",
                "content": content,
                "score": chunk.get("_rerank_score", chunk.get("_combined_score", 0)),
            })
    except Exception as e:
        logger.warning(f"Local prompts search failed: {e}")

    # Live API search
    api_results = []
    try:
        import urllib.request
        import urllib.parse
        params = urllib.parse.urlencode({"q": query, "limit": half_k})
        url = f"https://api.prompts.chat/api/prompts/search?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mullet/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = data.get("results", data) if isinstance(data, dict) else data
        if isinstance(results, list):
            for p in results:
                api_results.append({
                    "source": "live",
                    "title": p.get("title", "Untitled"),
                    "content": p.get("content", ""),
                    "id": p.get("id"),
                    "category": p.get("category"),
                    "type": p.get("type"),
                    "votes": p.get("vote_count", p.get("votes", 0)),
                })
    except Exception as e:
        logger.warning(f"prompts.chat API search failed: {e}")

    # Deduplicate by normalized title
    seen_titles: set[str] = set()
    combined = []
    for r in local_results + api_results:
        norm = r["title"].lower().strip()
        if norm not in seen_titles:
            seen_titles.add(norm)
            combined.append(r)

    if not combined:
        return f"No prompts found for: {query}"

    lines = [
        f"**Prompt search** for *{query}* — "
        f"{len(local_results)} local, {len(api_results)} live, "
        f"{len(combined)} deduplicated",
        "",
    ]
    for i, r in enumerate(combined[:top_k], 1):
        src_tag = "[LOCAL]" if r["source"] == "local" else "[LIVE]"
        meta_parts = [src_tag]
        if r.get("category"):
            meta_parts.append(r["category"])
        if r.get("type") and r.get("type") != "TEXT":
            meta_parts.append(r["type"])
        if r.get("votes"):
            meta_parts.append(f"{r['votes']} votes")
        lines.append(f"**{i}. {r['title']}** ({' | '.join(meta_parts)})")
        if r.get("id"):
            lines.append(f"   ID: `{r['id']}`")
        preview = (r.get("content") or "")[:200].replace("\n", " ")
        if preview:
            lines.append(f"   {preview}{'...' if len(r.get('content', '')) > 200 else ''}")
        lines.append("")

    return "\n".join(lines)


def handle_prompts_get(args: dict) -> str:
    """Get a specific prompt template by ID with optional variable substitution."""
    prompt_id = args.get("prompt_id", "")
    variables = args.get("variables", {})
    if not prompt_id:
        return "Error: prompt_id is required."
    try:
        import urllib.request
        url = f"https://api.prompts.chat/api/prompts/{prompt_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mullet/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        content = data.get("content", data.get("prompt", ""))
        if variables and content:
            for var_name, var_value in variables.items():
                content = re.sub(r'\$\{' + re.escape(var_name) + r'(?::[^}]*)?\}', str(var_value), content)
        return f"**{data.get('title', 'Prompt')}**\n\n{content}"
    except Exception as e:
        return f"Prompt get error: {e}"


def handle_trawl_docs_lookup(args: dict) -> str:
    """Look up documentation for a library using Context7 with fallback to local Trawl pools."""
    library = args.get("library", "")
    version = args.get("version", "latest")
    query = args.get("query", "")
    sources = args.get("sources")
    if not library or not query:
        return json.dumps({"error": "library and query are required"}, indent=2)

    from mullet.core.tools.trawl_docs_lookup import TrawlDocsLookup

    global _trawl_docs_lookup_instance
    if _trawl_docs_lookup_instance is None:
        from mullet.core.tools.mcp_client import Context7Client
        _trawl_docs_lookup_instance = TrawlDocsLookup(context7_client=Context7Client())

    lookup = _trawl_docs_lookup_instance
    if sources:
        lookup = TrawlDocsLookup(
            context7_client=_trawl_docs_lookup_instance.context7_client,
            source_preference=sources,
        )

    try:
        result = lookup.trawl_docs_lookup(library=library, version=version, query=query)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.warning("trawl_docs_lookup error: %s", e)
        return json.dumps({"error": str(e)}, indent=2)


# ============================================================================
# Handler dispatch
# ============================================================================

MCP_KNOWLEDGE_HANDLERS = {
    "trawl_list_pools": handle_list_pools,
    "trawl_load_pool": handle_load_pool,
    "trawl_get_pool": handle_get_pool,
    "trawl_pool_delete": handle_delete_pool,
    "trawl_search": handle_search,
    "trawl_search_pool": handle_search_pool,
    "trawl_recent": handle_recent,
    "trawl_search_code": handle_search_code,
    "trawl_search_multi": handle_search_multi,
    "trawl_find_symbol": handle_find_symbol,
    "trawl_reindex_pool": handle_reindex_pool,
    "trawl_sync_pool": handle_sync_pool,
    "trawl_ingest_status": handle_ingest_status,
    "trawl_ingest_file": handle_ingest_file,
    "trawl_ingest_url": handle_ingest_url,
    "trawl_memory_store": handle_memory_store,
    "trawl_memory_search": handle_memory_search,
    "trawl_memory_list": handle_memory_list,
    "trawl_memory_delete": handle_memory_delete,
    "trawl_build_graph": handle_build_graph,
    "trawl_graph_stats": handle_graph_stats,
    "trawl_graph_prune": handle_graph_prune,
    "trawl_search_entities": handle_search_entities,
    "trawl_search_relationships": handle_search_relationships,
    "trawl_list_sources": handle_list_sources,
    "trawl_sync_sources": handle_sync_sources,
    "trawl_connect_github": handle_connect_github,
    "trawl_crawl_web": handle_crawl_web,
    "trawl_summarize": handle_summarize,
    "trawl_list_repos": handle_list_repos,
    "trawl_search_repo": handle_search_repo,
    "trawl_graph_export": handle_graph_export,
    "trawl_workflow_create": handle_workflow_create,
    "trawl_workflow_status": handle_workflow_status,
    "trawl_workflow_update": handle_workflow_update,
    "trawl_workflow_list": handle_workflow_list,
    "trawl_agents_list": handle_agents_list,
    "trawl_agents_get": handle_agents_get,
    "trawl_run_agent": handle_run_agent,
    "trawl_prompts_search_live": handle_prompts_search_live,
    "trawl_prompts_find": handle_prompts_find,
    "trawl_prompts_get": handle_prompts_get,
    "trawl_docs_lookup": handle_trawl_docs_lookup,
}
