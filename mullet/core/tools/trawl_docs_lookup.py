"""Tool that looks up library/framework docs through the Trawl knowledge engine (Context7-style)."""
import logging
from typing import Any

from mullet.core.tools.mcp_client import Context7Client
from mullet.core.tools.web import web_search

logger = logging.getLogger(__name__)

DEFAULT_SOURCES = ["context7", "trawl", "web"]


class TrawlDocsLookup:
    """Documentation lookup using Context7 MCP with local Trawl pool fallback."""

    def __init__(
        self,
        context7_client: Context7Client | None = None,
        source_preference: list[str] | None = None,
    ):
        self._context7_client = context7_client
        self.source_preference = source_preference or DEFAULT_SOURCES

    @property
    def context7_client(self) -> Context7Client | None:
        """Lazily-constructed Context7 client; None when construction failed."""
        if self._context7_client is None:
            try:
                self._context7_client = Context7Client()
            except Exception as e:
                logger.warning("Failed to instantiate Context7Client: %s", e)
        return self._context7_client

    @context7_client.setter
    def context7_client(self, value: Context7Client | None) -> None:
        """Override the cached Context7 client (used by tests for injection)."""
        self._context7_client = value

    def trawl_docs_lookup(
        self, library: str, version: str, query: str
    ) -> dict[str, Any]:
        """Look up documentation for a specific library version."""
        full_query = f"{library} {version} {query}"

        for source in self.source_preference:
            if source == "context7":
                client = self.context7_client
                if client:
                    try:
                        context7_results = client.search_docs(
                            library=library, version=version, query=query
                        )
                        if context7_results:
                            return {
                                "source": "context7",
                                "library": library,
                                "version": version,
                                "query": query,
                                "results": context7_results,
                            }
                    except Exception as e:
                        logger.warning("Context7 lookup failed: %s", e)
            elif source == "trawl":
                try:
                    trawl_results = self._search_trawl_pools(full_query)
                    if trawl_results:
                        return {
                            "source": "trawl",
                            "library": library,
                            "version": version,
                            "query": query,
                            "results": trawl_results,
                        }
                except Exception as e:
                    logger.warning("Trawl pool lookup failed: %s", e)
            elif source == "web":
                try:
                    web_results = web_search(query=full_query, max_results=3)
                    if web_results:
                        return {
                            "source": "web",
                            "library": library,
                            "version": version,
                            "query": query,
                            "results": web_results,
                        }
                except Exception as e:
                    logger.warning("Web search fallback failed: %s", e)

        return {
            "source": "none",
            "library": library,
            "version": version,
            "query": query,
            "results": [],
        }

    def _search_trawl_pools(self, query: str) -> list[dict] | None:
        """Search across loaded/local Trawl pools using the knowledge engine."""
        try:
            from mullet.knowledge.engine.client import discover_clients, load_client
            from mullet.knowledge.engine.search.searcher import HybridSearcher
            from mullet.core.env import get_pools_dir

            pools_dir = get_pools_dir()
            if not pools_dir.exists():
                return None

            all_pools = discover_clients(clients_dir=pools_dir)
            pools_with_index = [
                p for p in all_pools if (pools_dir / p / "index").exists()
            ]
            if not pools_with_index:
                return None

            results: list[dict] = []
            for pool_name in pools_with_index:
                try:
                    client = load_client(pool_name, clients_dir=pools_dir)
                    searcher = HybridSearcher(
                        index_path=str(client.index_path),
                        collection_name=client.machine_name,
                        embedding_config=client.embedding_config,
                        tokenizer_config=client.tokenization_config,
                    )
                    pool_results = searcher.search(query, top_k=5)
                    for r in pool_results:
                        r["_pool"] = pool_name
                    results.extend(pool_results)
                except Exception as e:
                    logger.debug("Trawl search failed for pool '%s': %s", pool_name, e)

            return results if results else None
        except ImportError:
            logger.debug("Knowledge engine not available for Trawl pool search")
            return None

    def auto_suggest_versions(self, library: str) -> list[str]:
        """Auto-suggest available versions for a library."""
        client = self.context7_client
        if client:
            try:
                versions = client.get_versions(library=library)
                if versions:
                    return versions
            except Exception as e:
                logger.warning("Context7 version auto-suggest failed: %s", e)
        return []

    def auto_suggest_libraries(self, query: str) -> list[str]:
        """Auto-suggest libraries matching a query."""
        client = self.context7_client
        if client:
            try:
                libraries = client.search_libraries(query=query)
                if libraries:
                    return libraries
            except Exception as e:
                logger.warning("Context7 library auto-suggest failed: %s", e)
        return []
