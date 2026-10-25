"""Native version-aware library pool registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class LibraryPoolConfig:
    """Configuration for a versioned library pool."""
    name: str
    language: str
    version: str
    source_url: str
    last_synced: datetime | None = None


class LibraryPoolRegistry:
    """Registry of library pools persisted to JSON."""

    _path: Path
    _pools: dict[str, LibraryPoolConfig]

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from mullet.core.paths import data_dir
            path = data_dir() / "library_pools.json"
        self._path = path
        self._pools = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for item in data.get("pools", []):
                last_synced = None
                if item.get("last_synced"):
                    last_synced = datetime.fromisoformat(item["last_synced"])
                config = LibraryPoolConfig(
                    name=item["name"],
                    language=item["language"],
                    version=item["version"],
                    source_url=item["source_url"],
                    last_synced=last_synced,
                )
                self._pools[config.name] = config
        except Exception:
            # If the file is corrupt, start fresh.
            self._pools = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, Any]] = []
        for config in self._pools.values():
            item = {
                "name": config.name,
                "language": config.language,
                "version": config.version,
                "source_url": config.source_url,
                "last_synced": config.last_synced.isoformat() if config.last_synced else None,
            }
            payload.append(item)
        self._path.write_text(json.dumps({"pools": payload}, indent=2))

    def register(self, config: LibraryPoolConfig) -> None:
        """Register a library pool and persist to disk."""
        self._pools[config.name] = config
        self._save()

    def get(self, name: str) -> LibraryPoolConfig | None:
        """Retrieve a library pool by name."""
        return self._pools.get(name)

    def list(self) -> list[LibraryPoolConfig]:
        """Return all registered library pools."""
        return list(self._pools.values())

    def unregister(self, name: str) -> bool:
        """Remove a pool by name. Returns True if it existed."""
        if name in self._pools:
            del self._pools[name]
            self._save()
            return True
        return False
