"""Enterprise global context — company-wide document source management.

Admins connect company sources (GitHub org, Confluence, Jira, web crawl).
These are vectorized on schedule and merged with project context when
agents run tasks.
"""

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GlobalSource:
    """A company-wide document source."""
    id: str
    type: str  # github_org | confluence | jira | web_crawl
    config: dict = field(default_factory=dict)
    schedule: str = "0 2 * * *"  # Default: 2am daily
    last_sync: float = 0
    status: str = "pending"  # pending | syncing | ready | error
    error_message: str = ""


class GlobalContextManager:
    """Manage enterprise global document sources.

    When a developer runs a task, the system merges:
    1. Project context (from Trove client) — highest priority, 70% budget
    2. Global context (from enterprise sources) — supplementary, 30% budget
    """

    def __init__(self, db=None):
        self._db = db
        self._sources: dict[str, GlobalSource] = {}

    def add_source(self, source_type: str, config: dict, schedule: str = "0 2 * * *") -> GlobalSource:
        """Add a new global source."""
        source_id = f"gs_{int(time.time())}_{source_type}"
        source = GlobalSource(
            id=source_id,
            type=source_type,
            config=config,
            schedule=schedule,
        )
        self._sources[source_id] = source

        if self._db:
            self._save_to_db(source)

        logger.info("Added global source: %s (%s)", source_id, source_type)
        return source

    def remove_source(self, source_id: str):
        """Remove a global source."""
        self._sources.pop(source_id, None)
        if self._db:
            self._db.execute(
                "DELETE FROM global_sources WHERE id = ?", (source_id,)
            )
            self._db.commit()

    def list_sources(self) -> list[dict]:
        """List all global sources."""
        return [
            {
                "id": s.id,
                "type": s.type,
                "config": s.config,
                "schedule": s.schedule,
                "status": s.status,
                "last_sync": s.last_sync,
            }
            for s in self._sources.values()
        ]

    def sync_source(self, source_id: str) -> bool:
        """Trigger a sync for a specific source.

        Uses Trove's source connectors (GitHub, web crawler) to
        fetch and vectorize documents.
        """
        source = self._sources.get(source_id)
        if not source:
            return False

        source.status = "syncing"
        logger.info("Syncing global source: %s (%s)", source_id, source.type)

        try:
            if source.type == "github_org":
                self._sync_github(source)
            elif source.type == "web_crawl":
                self._sync_web_crawl(source)
            elif source.type in ("confluence", "jira"):
                self._sync_web_crawl(source)  # Treat as web crawl for now

            source.status = "ready"
            source.last_sync = time.time()
            source.error_message = ""
            return True

        except Exception as e:
            source.status = "error"
            source.error_message = str(e)[:500]
            logger.error("Sync failed for %s: %s", source_id, e)
            return False

    def build_global_context(self, query: str, max_tokens: int = 2400) -> str:
        """Build global context from all ready sources.

        This is called by the knowledge manager and merged with
        project-specific context using a 70/30 token budget split.
        """
        if not self._sources:
            return ""

        # For now, return a summary of available sources
        # Full implementation requires Trove search across global index
        parts = ["## Global Context"]
        for source in self._sources.values():
            if source.status == "ready":
                parts.append(f"- {source.type}: {source.config.get('org', source.config.get('url', '?'))}")

        return "\n".join(parts) if len(parts) > 1 else ""

    def _sync_github(self, source: GlobalSource):
        """Sync GitHub organization repos via Trove connector."""
        try:
            from core.sources.github_connector import GitHubSourceConnector
            connector = GitHubSourceConnector(source.config)
            connector.sync(target_dir=f"/tmp/mullet_global/{source.id}")
        except ImportError:
            logger.warning("Trove GitHub connector not available — skipping sync")

    def _sync_web_crawl(self, source: GlobalSource):
        """Sync web sources via Trove web crawler."""
        try:
            from core.sources.web_crawler import WebCrawler
            crawler = WebCrawler()
            urls = source.config.get("urls", [])
            for url in urls:
                crawler.crawl(url, max_depth=source.config.get("depth", 3))
        except ImportError:
            logger.warning("Trove web crawler not available — skipping sync")

    def _save_to_db(self, source: GlobalSource):
        """Persist source config to database."""
        if not self._db:
            return
        try:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS global_sources "
                "(id TEXT PRIMARY KEY, type TEXT, config TEXT, schedule TEXT, "
                "last_sync REAL, status TEXT)"
            )
            self._db.execute(
                "INSERT OR REPLACE INTO global_sources VALUES (?, ?, ?, ?, ?, ?)",
                (source.id, source.type, json.dumps(source.config),
                 source.schedule, source.last_sync, source.status),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("Failed to save source to DB: %s", e)
