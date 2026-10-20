"""Unified interface to Trove's knowledge capabilities.

Wraps Trove's Client, HybridSearcher, MemoryStore, and WorkflowEngine
behind a single manager scoped to a project. This is how Mullet agents
access project-specific knowledge.
"""

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mullet.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _get_engine_module(module_name: str):
    """Import from vendored engine or fall back to external Trove.

    Priority:
    1. Vendored engine (mullet.knowledge.engine.*)
    2. External Trove on TRAWL_PATH or sibling directory
    """
    # Try vendored first
    try:
        return __import__(f"mullet.knowledge.engine.{module_name}", fromlist=[module_name])
    except ImportError as _exc:
        logger.warning("manager: vendored module not found: %s", _exc)

    # Fall back to external Trove
    trove_paths = [
        os.environ.get("TRAWL_PATH", ""),
        str(Path(__file__).resolve().parents[4] / "trove"),
    ]
    for path in trove_paths:
        if path and Path(path).exists() and (Path(path) / "core" / "client.py").exists():
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                return __import__(f"core.{module_name}", fromlist=[module_name])
            except ImportError as _exc:
                logger.warning("manager: external module not found on path: %s", _exc)

    raise ImportError(f"Knowledge engine module '{module_name}' not found. "
                      "Install mullet with knowledge dependencies.")


@dataclass
class KnowledgeContext:
    """Formatted context ready for system prompt injection."""
    markdown: str = ""
    chunks_used: int = 0
    memories_used: int = 0
    tokens_estimated: int = 0


class KnowledgeManager:
    """Unified interface to Trove's knowledge capabilities.

    Wraps Client, HybridSearcher, MemoryStore, and WorkflowEngine
    behind a single manager scoped to a project client.
    """

    def __init__(self, client_name: str, clients_dir: str | None = None):
        client_mod = _get_engine_module("client")
        memory_mod = _get_engine_module("memory")
        workflow_mod = _get_engine_module("workflow")

        load_client = client_mod.load_client
        MemoryStore = memory_mod.MemoryStore
        WorkflowEngine = workflow_mod.WorkflowEngine

        self.client_name = client_name
        self.client = load_client(client_name, clients_dir=clients_dir)

        # Derive state DB path from index directory
        state_db = Path(self.client.index_path) / "mullet_state.db"
        self.memory = MemoryStore(db_path=str(state_db))
        self.workflows = WorkflowEngine(db_path=str(state_db))

        self._searcher = None
        self._searcher_initialized = False

    @property
    def searcher(self):
        """Lazy-init the HybridSearcher (heavy — loads embeddings model)."""
        if not self._searcher_initialized:
            try:
                search_mod = _get_engine_module("search.searcher")
                HybridSearcher = search_mod.HybridSearcher
                search_cfg = self.client.search_config or {}
                embed_cfg = self.client.embedding_config or {}

                self._searcher = HybridSearcher(
                    index_path=str(self.client.index_path),
                    embedding_model=embed_cfg.get("model"),
                    **{k: v for k, v in search_cfg.items()
                       if k in self._SEARCHER_PARAMS},
                )
            except Exception as e:
                logger.warning("Failed to initialize searcher for %s: %s", self.client_name, e)
                self._searcher = None
            self._searcher_initialized = True
        return self._searcher

    def build_context(
        self,
        query: str,
        max_tokens: int = 8000,
        include_memory: bool = True,
        include_graph: bool = False,
        search_mode: str = "precise",
        top_k: int = 10,
    ) -> KnowledgeContext:
        """Build a context block for agent system prompt injection.

        1. Search relevant chunks via hybrid search
        2. Load relevant memories
        3. Format into markdown with token budget
        """
        parts: list[str] = []
        chunks_used = 0
        memories_used = 0
        tokens_est = 0

        # Search for relevant chunks
        if self.searcher:
            try:
                results = self.searcher.search(query, top_k=top_k)
                if results:
                    parts.append("## Project Context")
                    for r in results:
                        content = r.get("content", r.get("text", ""))
                        source = r.get("source", r.get("metadata", {}).get("source", ""))
                        chunk_text = f"**{source}**\n{content}\n" if source else f"{content}\n"
                        chunk_tokens = len(chunk_text) // 4  # rough estimate
                        if tokens_est + chunk_tokens > max_tokens * 0.7:
                            break
                        parts.append(chunk_text)
                        tokens_est += chunk_tokens
                        chunks_used += 1
            except Exception as e:
                logger.warning("Search failed for %s: %s", self.client_name, e)

        # Load graph context (entity search)
        if include_graph:
            try:
                graph_search_mod = _get_engine_module("graph.search")
                graph_store_mod = _get_engine_module("graph.store")
                local_search = graph_search_mod.local_search
                KnowledgeGraph = graph_store_mod.KnowledgeGraph

                graph_path = Path(self.client.index_path)
                graph = KnowledgeGraph(str(graph_path))
                graph.load()

                if graph.stats.get("entities", 0) > 0 and self.searcher:
                    graph_results = local_search(query, graph, self.searcher, top_k=5)
                    if graph_results:
                        parts.append("\n## Related Entities")
                        for entity in graph_results[:5]:
                            name = entity.get("name", entity.get("key", ""))
                            etype = entity.get("type", "")
                            desc = entity.get("description", "")
                            if name:
                                entity_text = f"- **{name}** ({etype}): {desc}" if desc else f"- **{name}** ({etype})"
                                entity_tokens = len(entity_text) // 4
                                if tokens_est + entity_tokens > max_tokens * 0.9:
                                    break
                                parts.append(entity_text)
                                tokens_est += entity_tokens
            except Exception as e:
                logger.debug("Graph context failed: %s", e)

        # Load relevant memories
        if include_memory:
            try:
                memories = self.memory.search(query=query, top_k=5)
                if memories:
                    parts.append("\n## Memories")
                    for mem in memories:
                        content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
                        mem_tokens = len(content) // 4
                        if tokens_est + mem_tokens > max_tokens:
                            break
                        parts.append(f"- {content}")
                        tokens_est += mem_tokens
                        memories_used += 1
            except Exception as e:
                logger.debug("Memory search failed: %s", e)

        markdown = "\n".join(parts) if parts else ""

        return KnowledgeContext(
            markdown=markdown,
            chunks_used=chunks_used,
            memories_used=memories_used,
            tokens_estimated=tokens_est,
        )

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[dict]:
        """Direct search access for agent tools."""
        if not self.searcher:
            return []
        return self.searcher.search(query, top_k=top_k, **kwargs)

    def remember(self, key: str, content: str, namespace: str = "agent", ttl: int | None = None):
        """Store a memory from agent execution."""
        self.memory.store(namespace=namespace, key=key, content=content, ttl=ttl)

    def recall(self, query: str, namespace: str | None = None, top_k: int = 5) -> list[dict]:
        """Recall memories relevant to a query."""
        try:
            return self.memory.search(query=query, namespace=namespace, top_k=top_k)
        except Exception as e:
            logger.warning("Memory search failed in recall: %s", e)
            return self.memory.list(namespace=namespace) if namespace else []

    def create_workflow(self, name: str, steps: list[dict]) -> str:
        """Create a tracked workflow for multi-step tasks."""
        return self.workflows.create(name=name, steps=steps)

    def close(self):
        """Release resources."""
        if self._searcher:
            try:
                self._searcher.close()
            except Exception as e:
                logger.warning("Failed to close searcher: %s", e)

    # Parameters that HybridSearcher accepts — filter config keys to these
    _SEARCHER_PARAMS = {
        "bm25_weight", "vector_weight", "min_score",
        "merge_strategy", "adaptive_rerank", "mmr_lambda",
        "code_priority", "hyde", "multi_query",
        "cache_ttl", "cache_max_size",
    }


@contextmanager
def knowledge_session(
    client_name: str,
    *,
    register_tools: "ToolRegistry | None" = None,
    clients_dir: str | None = None,
):
    """Context manager for the standard KnowledgeManager lifecycle.

    Yields the KnowledgeManager instance. If a tool registry is passed,
    registers knowledge tools on it. KM.close() is invoked on exit; any
    failure there is logged but never raised.

    Usage:
        with knowledge_session(args.client, register_tools=registry) as km:
            context = km.build_context(task)
            ...
    """
    km = KnowledgeManager(client_name, clients_dir=clients_dir)
    if register_tools is not None:
        from mullet.core.knowledge import tools as knowledge_tools
        knowledge_tools.register(register_tools, km)
    try:
        yield km
    finally:
        close = getattr(km, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning("knowledge_session: KM.close failed: %s", exc)


def build_knowledge_environment(
    client_name: str,
    task: str,
    registry: "ToolRegistry",
    system_prompt: str,
    clients_dir: str | None = None,
) -> tuple["KnowledgeManager", KnowledgeContext, str]:
    """Construct KM, register knowledge tools, build context, augment prompt.

    Returns ``(km, context, augmented_system_prompt)``. The KM is NOT
    closed automatically — callers should either call ``km.close()`` or
    use ``knowledge_session()`` for full lifecycle management.
    """
    from mullet.core.knowledge.context import build_system_prompt as _build_sp
    from mullet.core.knowledge import tools as knowledge_tools

    km = KnowledgeManager(client_name, clients_dir=clients_dir)
    knowledge_tools.register(registry, km)
    context = km.build_context(task)
    augmented = _build_sp(system_prompt, context, client_name)
    return km, context, augmented
