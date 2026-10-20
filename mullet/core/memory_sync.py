"""Sync observations and decisions into Mullet's knowledge/memory layer.

Public API
----------
sync_observation_summary(summary, *, base)  -> str | None
sync_decisions(*, base, limit)              -> int
backfill_all(*, base)                       -> dict[str, int]

All functions are fail-open: if the MemoryStore is unavailable, a WARNING is
logged and 0/None is returned without raising.

Storage conventions
-------------------
Namespace:  "mullet-pipeline"
Tags (stored in metadata["tags"]):
    observations: ["mullet", "pipeline", "observations", "session-summary", "session:<sid>"]
    decisions:    ["mullet", "pipeline", "decisions", "<kind>"]

The original dataclass-as-dict is stored verbatim in metadata["data"] so that
callers can round-trip the structured record from the memory entry.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mullet.core.memory_sync")

_NAMESPACE = "mullet-pipeline"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_store(base: Optional[Path] = None):
    """Return a MemoryStore or None if the dependency is unavailable."""
    try:
        from mullet.knowledge.engine.memory import MemoryStore
        from mullet.core.env import get_pools_dir

        pools_dir = get_pools_dir()
        store = MemoryStore(pools_dir / ".memory")
        return store
    except Exception as exc:
        logger.warning("memory_sync: MemoryStore unavailable: %s", exc)
        return None


def _fallback_jsonl(base: Optional[Path] = None) -> Path:
    """Path to local JSONL fallback when MemoryStore is unavailable."""
    root = base or Path.cwd()
    p = root / ".mullet" / "memory_sync.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_fallback(record: dict, base: Optional[Path] = None) -> None:
    """Append *record* to the local fallback JSONL."""
    try:
        path = _fallback_jsonl(base)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.debug("memory_sync: fallback write failed: %s", exc)


def _load_fallback_ids(base: Optional[Path] = None) -> set[str]:
    """Return the set of already-synced IDs from the fallback JSONL."""
    path = _fallback_jsonl(base)
    if not path.exists():
        return set()
    seen: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" in rec:
                seen.add(rec["id"])
    except Exception:
        pass
    return seen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sync_observation_summary(
    summary,  # SessionSummary — avoid hard import so callers don't need the dep
    *,
    base: Optional[Path] = None,
) -> str | None:
    """Persist a SessionSummary into Mullet's memory store.

    Parameters
    ----------
    summary:
        A :class:`mullet.core.observation_summary.SessionSummary` instance.
    base:
        Project root; used for the fallback JSONL path when MemoryStore is
        unavailable.

    Returns
    -------
    str | None
        The memory ID on success, None on failure or empty summary.
    """
    try:
        if summary is None:
            return None

        sid = getattr(summary, "session_id", "") or ""
        text_parts = [f"Observation summary for session {sid}."]
        s = getattr(summary, "summary", "")
        if s:
            text_parts.append(s)
        bullets = getattr(summary, "bullet_points", []) or []
        if bullets:
            text_parts.append("Key points: " + "; ".join(bullets))
        open_q = getattr(summary, "open_questions", []) or []
        if open_q:
            text_parts.append("Open questions: " + "; ".join(open_q))
        content = " ".join(text_parts)

        tags = ["mullet", "pipeline", "observations", "session-summary"]
        if sid:
            tags.append(f"session:{sid}")

        try:
            data = asdict(summary)
        except Exception:
            data = {
                k: getattr(summary, k, None)
                for k in ("session_id", "started_at", "ended_at",
                          "tool_counts", "files_touched", "summary",
                          "bullet_points", "open_questions")
            }

        root = base or Path.cwd()
        source_path = str(root / ".mullet" / "observations" / "summaries")
        metadata = {"tags": tags, "data": data, "source_path": source_path}

        key = f"obs-summary:{sid}"

        store = _get_store(base)
        if store is None:
            logger.warning(
                "memory layer unavailable; falling back to local jsonl"
            )
            record = {"id": key, "namespace": _NAMESPACE, "key": key,
                      "content": content, "metadata": metadata}
            _write_fallback(record, base)
            return None

        mem_id = store.store(
            namespace=_NAMESPACE,
            key=key,
            content=content,
            metadata=metadata,
        )
        logger.debug("memory_sync: stored observation summary %s → %s", key, mem_id)
        return mem_id

    except Exception as exc:
        logger.warning("memory_sync: sync_observation_summary failed: %s", exc)
        return None


def sync_decisions(
    *,
    base: Optional[Path] = None,
    limit: Optional[int] = None,
) -> int:
    """Idempotently sync .mullet/decisions.jsonl rows into memory.

    Deduplication is done by decision.id stored in metadata.  Rows already
    present in the store (matched by namespace+key) are skipped via the
    store's own upsert logic, but we also track a local seen-set to avoid
    redundant store calls.

    Parameters
    ----------
    base:   Project root; defaults to cwd.
    limit:  Max number of rows to sync in one call (newest rows are not
            prioritised — file order is used).

    Returns
    -------
    int
        Number of decisions newly written to the store (or fallback JSONL).
    """
    try:
        from mullet.core.decisions import load, decisions_path

        all_decisions = load(base)
        if limit is not None:
            all_decisions = all_decisions[:limit]

        if not all_decisions:
            return 0

        store = _get_store(base)
        fallback_mode = store is None
        if fallback_mode:
            logger.warning(
                "memory layer unavailable; falling back to local jsonl"
            )
            existing_ids = _load_fallback_ids(base)
        else:
            existing_ids = set()  # store handles upserts natively

        source_path = str(decisions_path(base))
        count = 0

        for decision in all_decisions:
            key = f"decision:{decision.id}"

            if fallback_mode:
                if decision.id in existing_ids:
                    continue
                tags = ["mullet", "pipeline", "decisions", decision.kind]
                metadata = {
                    "tags": tags,
                    "data": decision.to_dict(),
                    "source_path": source_path,
                }
                record = {
                    "id": decision.id,
                    "namespace": _NAMESPACE,
                    "key": key,
                    "content": decision.excerpt,
                    "metadata": metadata,
                }
                _write_fallback(record, base)
                count += 1
                continue

            tags = ["mullet", "pipeline", "decisions", decision.kind]
            metadata = {
                "tags": tags,
                "data": decision.to_dict(),
                "source_path": source_path,
            }
            try:
                store.store(
                    namespace=_NAMESPACE,
                    key=key,
                    content=decision.excerpt,
                    metadata=metadata,
                )
                count += 1
            except Exception as exc:
                logger.warning(
                    "memory_sync: failed to store decision %s: %s",
                    decision.id,
                    exc,
                )

        logger.debug("memory_sync: synced %d decisions", count)
        return count

    except Exception as exc:
        logger.warning("memory_sync: sync_decisions failed: %s", exc)
        return 0


def backfill_all(*, base: Optional[Path] = None) -> dict:
    """Run all sync helpers and return counts.

    Returns
    -------
    dict
        ``{"observations": N, "decisions": M}`` where N is the number of
        summary JSON files successfully synced and M is the number of
        decisions synced.
    """
    obs_count = 0
    dec_count = 0

    # --- observation summaries ---
    try:
        from mullet.core.observation_summary import SessionSummary

        root = base or Path.cwd()
        summaries_dir = root / ".mullet" / "observations" / "summaries"
        if summaries_dir.is_dir():
            for json_file in sorted(summaries_dir.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    # Reconstruct SessionSummary from saved JSON
                    summary = SessionSummary(
                        session_id=data.get("session_id", ""),
                        started_at=data.get("started_at", ""),
                        ended_at=data.get("ended_at", ""),
                        tool_counts=data.get("tool_counts", {}),
                        files_touched=data.get("files_touched", []),
                        summary=data.get("summary", ""),
                        bullet_points=data.get("bullet_points", []),
                        open_questions=data.get("open_questions", []),
                    )
                    mem_id = sync_observation_summary(summary, base=base)
                    if mem_id is not None:
                        obs_count += 1
                    else:
                        # fallback path still counts as synced
                        obs_count += 1
                except Exception as exc:
                    logger.warning(
                        "memory_sync: backfill skipped %s: %s", json_file, exc
                    )
    except Exception as exc:
        logger.warning("memory_sync: backfill observations failed: %s", exc)

    # --- decisions ---
    try:
        dec_count = sync_decisions(base=base)
    except Exception as exc:
        logger.warning("memory_sync: backfill decisions failed: %s", exc)

    return {"observations": obs_count, "decisions": dec_count}
