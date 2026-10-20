"""mullet pool audit — per-pool diagnostic report."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import NamedTuple

# Canonical path of the github_auditer data directory.
# Override via the MULLET_AUDITER_PATH env var; if unset, auditer integration
# is treated as not configured rather than crashing on a missing directory.
_AUDITER_ROOT = Path(os.environ["MULLET_AUDITER_PATH"]) if os.environ.get("MULLET_AUDITER_PATH") else None
_AUDITER_DATA = (_AUDITER_ROOT / "data") if _AUDITER_ROOT else None


class _DocEntry(NamedTuple):
    name: str
    is_symlink: bool
    target: str | None
    target_exists: bool


class _VectorInfo(NamedTuple):
    indexed: bool
    chunk_count: int
    has_bm25: bool
    error: str | None


class _GraphInfo(NamedTuple):
    built: bool
    entity_count: int
    relationship_count: int
    db_size_mb: float
    error: str | None


class PoolAudit(NamedTuple):
    """Diagnostic snapshot of a single knowledge pool."""
    name: str
    pool_dir: Path
    has_client_yaml: bool
    description: str
    machine_name: str
    metadata: dict
    docs_symlink_target: str | None   # set when documents/ itself is a symlink
    docs_symlink_exists: bool
    doc_entries: list[_DocEntry]
    vector: _VectorInfo
    graph: _GraphInfo
    index_size_mb: float
    manager: str          # "auditer" | "manual" | "mixed" | "none"
    sync_commands: list[str]
from typing import NamedTuple

import logging
logger = logging.getLogger(__name__)


def _dir_size_mb(path: Path) -> float:
    total = 0
    try:
        for dirpath, _dirs, files in os.walk(path, followlinks=False):
            for fname in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fname))
                except OSError:
                    pass
    except Exception as _exc:
        logger.warning("pool_audit: _dir_size_mb: %s", _exc)
    return total / (1024 * 1024)

def _read_yaml_fields(yaml_path: Path) -> tuple[str, str]:
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text()) or {}
        return data.get("description", ""), data.get("machine_name", "")
    except Exception as _exc:
        logger.warning("pool_audit: _read_yaml_fields: %s", _exc)
        return "", ""


def _get_doc_entries(documents_path: Path) -> list[_DocEntry]:
    """List entries inside documents/. Call only when documents/ is a real directory."""
    entries = []
    if not documents_path.exists() or documents_path.is_symlink():
        return entries
    try:
        for entry in sorted(documents_path.iterdir()):
            is_sym = entry.is_symlink()
            target = None
            target_exists = True
            if is_sym:
                try:
                    target = str(os.readlink(entry))
                    target_exists = entry.exists()
                except Exception:
                    target = "?"
                    target_exists = False
            entries.append(_DocEntry(entry.name, is_sym, target, target_exists))
    except Exception as _exc:
        logger.warning("pool_audit: _get_doc_entries: %s", _exc)
    return entries


def _get_vector_info(index_path: Path, timeout: float = 10.0) -> _VectorInfo:
    if not index_path.exists():
        return _VectorInfo(False, 0, False, None)
    has_bm25 = (index_path / "bm25_cache.json").exists()

    def _query() -> tuple[int, str | None]:
        import chromadb
        # Disable telemetry — it spawns threads that make network calls and
        # cannot be interrupted by Ctrl+C, causing the audit to hang.
        try:
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=str(index_path),
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception:
            client = chromadb.PersistentClient(path=str(index_path))
        colls = client.list_collections()
        return sum(c.count() for c in colls), None

    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TE
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_query)
            try:
                chunk_count, err = fut.result(timeout=timeout)
            except _TE:
                return _VectorInfo(False, 0, has_bm25, f"timed out after {timeout:.0f}s")
        return _VectorInfo(chunk_count > 0, chunk_count, has_bm25, None)
    except ImportError:
        return _VectorInfo(False, 0, has_bm25, "chromadb not installed")
    except Exception as e:
        return _VectorInfo(False, 0, has_bm25, str(e)[:60])


def _get_graph_info(index_path: Path) -> _GraphInfo:
    db_path = index_path / "knowledge_graph.db"
    if not db_path.exists():
        return _GraphInfo(False, 0, 0, 0.0, None)
    db_size = db_path.stat().st_size / (1024 * 1024)
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        conn.close()
        return _GraphInfo(True, entities, relationships, db_size, None)
    except Exception as e:
        return _GraphInfo(True, 0, 0, db_size, str(e)[:60])


def _auditer_slug_for(target_str: str | None) -> str | None:
    """Return the auditer account slug if target resolves under github_auditer/data/, else None.

    Tolerates None / empty / sentinel inputs (``"?"`` from broken symlinks).
    Also tolerates the auditer integration being unconfigured — when
    ``MULLET_AUDITER_PATH`` is unset, ``_AUDITER_DATA`` is None and the
    canonical path-match below would raise TypeError on every audit pass.
    """
    if not target_str or target_str == "?":
        return None
    try:
        # Path-resolved check only when the auditer root is configured.
        if _AUDITER_DATA is not None:
            real = Path(target_str).resolve()
            if real.is_relative_to(_AUDITER_DATA):
                return real.relative_to(_AUDITER_DATA).parts[0]
        # Textual fallback works regardless of MULLET_AUDITER_PATH:
        # any path containing .../github_auditer/data/<slug>/... counts.
        p = Path(target_str)
        parts = p.parts
        for i, part in enumerate(parts):
            if part == "data" and i > 0 and parts[i - 1] == "github_auditer":
                if i + 1 < len(parts):
                    return parts[i + 1]
    except Exception as _exc:
        logger.warning("pool_audit: _auditer_slug_for(%r): %s", target_str, _exc)
    return None


def _classify_sources(
    docs_symlink_target: str | None,
    doc_entries: list[_DocEntry],
) -> tuple[str, list[str]]:
    """
    Return (manager, sync_commands).

    manager: "auditer" | "manual" | "mixed" | "none"
    sync_commands: ready-to-run shell commands to pull latest changes
    """
    all_targets: list[str] = []
    if docs_symlink_target:
        all_targets.append(docs_symlink_target)
    else:
        all_targets.extend(e.target for e in doc_entries if e.is_symlink and e.target)

    if not all_targets:
        return "none", []

    auditer_slugs: list[str] = []
    manual_paths: list[str] = []

    for t in all_targets:
        slug = _auditer_slug_for(t)
        if slug:
            if slug not in auditer_slugs:
                auditer_slugs.append(slug)
        else:
            # Resolve to real path for the git pull command
            real = str(Path(t).resolve())
            if real not in manual_paths:
                manual_paths.append(real)

    if auditer_slugs and not manual_paths:
        manager = "auditer"
    elif manual_paths and not auditer_slugs:
        manager = "manual"
    elif auditer_slugs or manual_paths:
        manager = "mixed"
    else:
        manager = "none"

    cmds: list[str] = []
    for slug in auditer_slugs:
        cmds.append(f"auditer checkout --account {slug}")
    for path in manual_paths:
        cmds.append(f"git -C {path} pull")

    return manager, cmds


def audit_pools(pools_dir: Path, verbose: bool = True) -> list[PoolAudit]:
    """Scan pools_dir and return a diagnostic audit for each pool."""
    from mullet.knowledge.engine.pool_metadata import read_metadata

    audits = []
    if not pools_dir.exists():
        return audits
    pool_dirs = sorted(d for d in pools_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    total = len(pool_dirs)
    for i, d in enumerate(pool_dirs, 1):
        if verbose:
            print(f"  [{i}/{total}] {d.name} …", end="\r", flush=True)
        yaml_path = d / "config.yaml"
        has_yaml = yaml_path.exists()
        if not has_yaml:
            # Legacy fallback
            yaml_path = d / "client.yaml"
            has_yaml = yaml_path.exists()
        desc, machine_name = _read_yaml_fields(yaml_path) if has_yaml else ("", "")
        metadata = read_metadata(d)
        documents_path = d / "documents"
        docs_symlink_target: str | None = None
        docs_symlink_exists = True
        if documents_path.is_symlink():
            try:
                docs_symlink_target = str(os.readlink(documents_path))
                docs_symlink_exists = documents_path.exists()
            except Exception:
                docs_symlink_target = "?"
                docs_symlink_exists = False
        doc_entries = _get_doc_entries(documents_path)
        manager, sync_commands = _classify_sources(docs_symlink_target, doc_entries)
        index_path = d / "index"
        vector = _get_vector_info(index_path)
        graph = _get_graph_info(index_path)
        index_size = _dir_size_mb(index_path)
        audits.append(PoolAudit(
            name=d.name,
            pool_dir=d,
            has_client_yaml=has_yaml,
            description=desc,
            machine_name=machine_name,
            metadata=metadata,
            docs_symlink_target=docs_symlink_target,
            docs_symlink_exists=docs_symlink_exists,
            doc_entries=doc_entries,
            vector=vector,
            graph=graph,
            index_size_mb=index_size,
            manager=manager,
            sync_commands=sync_commands,
        ))
    if verbose and total:
        print(" " * 60, end="\r", flush=True)  # clear progress line
    return audits


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "never"
    return iso[:10]


_MANAGER_LABEL = {
    "auditer": "\033[34mGithub Auditer\033[0m",
    "manual":  "\033[33mManual\033[0m",
    "mixed":   "\033[35mMixed\033[0m",
    "none":    "\033[90mnone\033[0m",
}


def format_audit_report(audits: list[PoolAudit]) -> str:
    """Format a list of pool audits into a human-readable report."""
    lines = []
    BAR = "─" * 64

    for a in audits:
        yaml_flag = "\033[32m✓\033[0m" if a.has_client_yaml else "\033[31m✗\033[0m"
        manager_label = _MANAGER_LABEL.get(a.manager, a.manager)
        lines.append(f"\n{BAR}")
        lines.append(f"  \033[1m{a.name}\033[0m  [yaml {yaml_flag}]  [{manager_label}]")
        if a.description:
            lines.append(f"  \033[90m{a.description}\033[0m")
        if a.machine_name and a.machine_name != a.name:
            lines.append(f"  machine_name: {a.machine_name}")
        lines.append("")

        # Metadata
        created = _fmt_date(a.metadata.get("created_at"))
        indexed = _fmt_date(a.metadata.get("last_indexed_at"))
        searched = _fmt_date(a.metadata.get("last_searched_at"))
        searches = a.metadata.get("search_count", 0)
        lines.append(
            f"  \033[90mDates\033[0m    created {created}  "
            f"indexed {indexed}  "
            f"searched {searched} ({searches}x)"
        )

        # Documents
        if a.docs_symlink_target is not None:
            status = "\033[32m✓\033[0m" if a.docs_symlink_exists else "\033[31m✗ BROKEN\033[0m"
            lines.append(f"  \033[90mDocs\033[0m     → {a.docs_symlink_target}  [{status}]")
        elif a.doc_entries:
            lines.append(f"  \033[90mDocs\033[0m     {len(a.doc_entries)} entr{'y' if len(a.doc_entries) == 1 else 'ies'}")
            for e in a.doc_entries:
                if e.is_symlink:
                    status = "\033[32m✓\033[0m" if e.target_exists else "\033[31m✗ BROKEN\033[0m"
                    lines.append(f"           → {e.name}  →  {e.target}  [{status}]")
                else:
                    lines.append(f"           · {e.name}")
        else:
            lines.append("  \033[90mDocs\033[0m     (empty)")

        # Sync commands
        if a.sync_commands:
            lines.append("  \033[90mSync\033[0m     " + a.sync_commands[0])
            for cmd in a.sync_commands[1:]:
                lines.append("           " + cmd)

        # Vector
        if a.vector.error and not a.vector.indexed:
            lines.append(f"  \033[90mVector\033[0m   \033[31merror:\033[0m {a.vector.error}")
        elif a.vector.indexed:
            bm25 = "\033[32mbm25 ✓\033[0m" if a.vector.has_bm25 else "\033[90mbm25 ✗\033[0m"
            lines.append(
                f"  \033[90mVector\033[0m   {a.vector.chunk_count:,} chunks  {bm25}  "
                f"{a.index_size_mb:.1f} MB"
            )
        else:
            lines.append("  \033[90mVector\033[0m   (not indexed)")

        # Graph
        if a.graph.built:
            graph_detail = (
                f"{a.graph.entity_count} entities  "
                f"{a.graph.relationship_count} relationships  "
                f"{a.graph.db_size_mb:.1f} MB"
            )
            if a.graph.error:
                graph_detail += f"  \033[31merror: {a.graph.error}\033[0m"
            lines.append(f"  \033[90mGraph\033[0m    {graph_detail}")
        else:
            lines.append("  \033[90mGraph\033[0m    (not built)")

    lines.append(f"\n{BAR}")

    # Summary counts
    total = len(audits)
    indexed = sum(1 for a in audits if a.vector.indexed)
    graphed = sum(1 for a in audits if a.graph.built)
    no_yaml = sum(1 for a in audits if not a.has_client_yaml)
    lines.append(
        f"\n{total} pools  |  {indexed} indexed  |  {graphed} with graph"
        + (f"  |  \033[33m{no_yaml} missing config.yaml\033[0m" if no_yaml else "")
    )
    return "\n".join(lines)


def run_pool_audit(pools_dir: Path) -> None:
    """Run a pool audit and print the formatted report."""
    audits = audit_pools(pools_dir)
    if not audits:
        print(f"No pools found in {pools_dir}")
        return
    print(format_audit_report(audits))


def _purge_pool_sidecars(pool_name: str, pool_slug: str, mullet_home: Path) -> tuple[int, list[str]]:
    """Purge mullet.db + interactions.db rows and analytics/caches keyed on pool name.

    Returns (rows_deleted, warnings). Best-effort: missing DBs/tables are silent,
    but SQL errors are captured in warnings so the user sees them.
    """
    rows_deleted = 0
    warnings: list[str] = []

    candidates = {pool_name, pool_slug}

    for db_name in ("mullet.db", "interactions.db"):
        db_path = mullet_home / db_name
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA foreign_keys = ON")
            cur = conn.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            for tbl in tables:
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{tbl}')").fetchall()]
                for col in ("pool_name", "pool", "client", "client_name"):
                    if col in cols:
                        for name in candidates:
                            cur.execute(f"DELETE FROM '{tbl}' WHERE {col} = ?", (name,))
                            rows_deleted += cur.rowcount or 0
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            warnings.append(f"{db_name}: {e}")

    for sidecar in (
        mullet_home / "analytics" / f"{pool_slug}.json",
        mullet_home / "analytics" / pool_slug,
        mullet_home / "state" / f"{pool_slug}.json",
        mullet_home / "routing_history" / f"{pool_slug}.json",
    ):
        try:
            if sidecar.is_dir():
                import shutil
                shutil.rmtree(sidecar)
            elif sidecar.exists() or sidecar.is_symlink():
                sidecar.unlink()
        except Exception as e:
            warnings.append(f"{sidecar}: {e}")

    return rows_deleted, warnings


def _force_rmtree(path: Path) -> tuple[bool, str | None]:
    """Remove directory tree forcefully. Cleans broken symlinks first.

    Returns (success, error_message).
    """
    import shutil
    import stat

    if not path.exists() and not path.is_symlink():
        return True, None

    docs = path / "documents"
    if docs.is_symlink() and not docs.exists():
        try:
            docs.unlink()
        except Exception:
            pass

    def _on_rm_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            func(p)
        except Exception:
            raise

    try:
        if path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path, onerror=_on_rm_error)
    except Exception as e:
        return False, str(e)

    if path.exists():
        return False, f"rmtree completed but path still exists: {path}"
    return True, None


# ───────────────────────────────────────────────────────────────────────────
# Pool health (P1-4)
# ───────────────────────────────────────────────────────────────────────────

# Size (in bytes) of a freshly-initialised, empty ChromaDB sqlite file. Used
# to detect "never-indexed" pools whose index/chroma.sqlite3 is just the stock
# schema with no chunks written. Empirically ~168 KiB; we compare against a
# tolerance window.
_EMPTY_CHROMA_SIZE_MIN = 160 * 1024
_EMPTY_CHROMA_SIZE_MAX = 180 * 1024


class PoolHealthIssue(NamedTuple):
    """A single health issue found during pool health checks."""
    code: str       # "empty_but_configured" | "stale_manifest" | "manifest_chroma_mismatch"
                    # | "broken_symlink" | "never_indexed"
    message: str


class PoolHealth(NamedTuple):
    """Overall health status of a single knowledge pool."""
    name: str
    pool_dir: Path
    ok: bool
    chunk_count: int
    file_count: int
    last_indexed_human: str | None
    issues: list[PoolHealthIssue]


def _fmt_delta_ago(seconds: float) -> str:
    """Format an age (seconds) as a human delta like '2h ago' / '3d ago'."""
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _collect_symlink_issues(pool_dir: Path) -> list[PoolHealthIssue]:
    """Detect broken symlinks in documents/ (top-level or under it)."""
    issues: list[PoolHealthIssue] = []
    docs = pool_dir / "documents"
    if docs.is_symlink() and not docs.exists():
        try:
            target = str(os.readlink(docs))
        except Exception:
            target = "?"
        issues.append(PoolHealthIssue(
            "broken_symlink",
            f"documents/ → {target} (target missing)",
        ))
        return issues

    if not docs.exists() or not docs.is_dir():
        return issues

    try:
        for dirpath, _dirs, files in os.walk(str(docs), followlinks=False):
            dp = Path(dirpath)
            for name in list(_dirs) + files:
                entry = dp / name
                if entry.is_symlink() and not entry.exists():
                    try:
                        target = str(os.readlink(entry))
                    except Exception:
                        target = "?"
                    try:
                        rel = entry.relative_to(docs)
                    except ValueError:
                        rel = entry.name
                    issues.append(PoolHealthIssue(
                        "broken_symlink",
                        f"documents/{rel} → {target} (target missing)",
                    ))
    except Exception as _exc:
        logger.warning("pool_audit: _collect_symlink_issues: %s", _exc)

    return issues


def _chroma_chunk_count(index_path: Path) -> tuple[int, bool, str | None]:
    """Return (chunk_count, chroma_present, error_message).

    chroma_present is True when a chroma.sqlite3 file exists AND is bigger
    than the empty-init threshold. chunk_count is the actual count from
    chroma (0 if unavailable).
    """
    chroma_sqlite = index_path / "chroma.sqlite3"
    if not chroma_sqlite.exists():
        return 0, False, None

    try:
        size = chroma_sqlite.stat().st_size
    except Exception:
        size = 0
    # "stock empty" sqlite is ~168KB — treat anything in that window as
    # never-indexed. Anything bigger is real data.
    is_empty_init = _EMPTY_CHROMA_SIZE_MIN <= size <= _EMPTY_CHROMA_SIZE_MAX
    chroma_present = not is_empty_init

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(index_path))
        colls = client.list_collections()
        chunks = sum(c.count() for c in colls)
        return chunks, chroma_present, None
    except ImportError:
        return 0, chroma_present, "chromadb not installed"
    except Exception as e:
        return 0, chroma_present, str(e)[:120]


def _max_mtime(path: Path) -> float:
    """Max mtime of any file under path (recursive). Returns 0.0 if empty."""
    latest = 0.0
    if not path.exists():
        return latest
    try:
        if path.is_file():
            return path.stat().st_mtime
        for dirpath, _dirs, files in os.walk(str(path), followlinks=False):
            for fname in files:
                try:
                    m = os.path.getmtime(os.path.join(dirpath, fname))
                    if m > latest:
                        latest = m
                except (OSError, FileNotFoundError):
                    continue
    except Exception as _exc:
        logger.warning("pool_audit: _max_mtime: %s", _exc)
    return latest


def _count_doc_files(docs_path: Path) -> int:
    """Count real files under documents/ (excludes broken symlinks, does not follow symlinks)."""
    if not docs_path.exists() or not docs_path.is_dir():
        return 0
    count = 0
    try:
        for _dirpath, _dirs, files in os.walk(str(docs_path), followlinks=False):
            for fname in files:
                if not fname.startswith("."):
                    count += 1
    except Exception as _exc:
        logger.warning("pool_audit: _count_doc_files: %s", _exc)
    return count


def _load_manifest_meta(index_path: Path) -> tuple[int, float | None]:
    """Return (manifest_file_count, manifest_mtime) or (0, None) if absent."""
    manifest = index_path / "file_manifest.json"
    if not manifest.exists():
        return 0, None
    try:
        import json
        data = json.loads(manifest.read_text())
        files = data.get("files", {})
        return len(files), manifest.stat().st_mtime
    except Exception as _exc:
        logger.warning("pool_audit: _load_manifest_meta: %s", _exc)
        return 0, None


def _check_config_validity(config: dict) -> list[PoolHealthIssue]:
    """Validate required config.yaml fields. Returns issues for missing/empty values."""
    issues: list[PoolHealthIssue] = []
    if not config.get("machine_name"):
        issues.append(PoolHealthIssue(
            "invalid_config",
            "config.yaml missing required field 'machine_name'",
        ))
    embedding = config.get("embedding") or {}
    if not embedding.get("model"):
        issues.append(PoolHealthIssue(
            "invalid_config",
            "config.yaml missing embedding.model (needed for vector search)",
        ))
    return issues


def _check_exclude_patterns(pool_dir: Path, config: dict) -> list[PoolHealthIssue]:
    """Flag exclude_patterns that would filter out ≥90% of documents/ files."""
    import fnmatch
    patterns = (config.get("documents") or {}).get("exclude_patterns") or []
    if not patterns:
        return []
    docs = pool_dir / "documents"
    if not docs.exists() or not docs.is_dir():
        return []
    try:
        all_files = [
            f for f in docs.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        ]
    except Exception as _exc:
        logger.warning("pool_audit: _check_exclude_patterns: %s", _exc)
        return []
    if not all_files:
        return []
    excluded = sum(
        1 for f in all_files
        if any(fnmatch.fnmatch(str(f), p) for p in patterns)
    )
    pct = excluded / len(all_files)
    if pct >= 0.9:
        return [PoolHealthIssue(
            "bad_exclude_pattern",
            f"exclude_patterns filter out {excluded}/{len(all_files)} files "
            f"({pct:.0%}) — pool would index near-nothing; review documents.exclude_patterns",
        )]
    return []


def _max_indexed_mtime(docs_path: Path, config: dict) -> float:
    """Max mtime of files the indexer would actually process.

    Filters by documents.extensions / exclude_extensions / exclude_patterns so
    non-indexed files (images, binaries, no-ext files) don't trigger false
    stale-manifest warnings. Falls back to _max_mtime when no filtering configured.
    """
    extensions = set((config.get("documents") or {}).get("extensions") or [])
    exclude_exts = set((config.get("documents") or {}).get("exclude_extensions") or [])
    exclude_patterns = (config.get("documents") or {}).get("exclude_patterns") or []

    if not extensions and not exclude_exts and not exclude_patterns:
        return _max_mtime(docs_path)

    import fnmatch as _fnmatch
    latest = 0.0
    if not docs_path.exists():
        return latest
    try:
        for dirpath, _dirs, files in os.walk(str(docs_path), followlinks=False):
            for fname in files:
                fpath = os.path.join(dirpath, fname)
                ext = Path(fname).suffix.lower()
                if extensions and ext not in extensions:
                    continue
                if ext in exclude_exts:
                    continue
                if any(_fnmatch.fnmatch(fpath, p) for p in exclude_patterns):
                    continue
                try:
                    m = os.path.getmtime(fpath)
                    if m > latest:
                        latest = m
                except (OSError, FileNotFoundError):
                    continue
    except Exception as _exc:
        logger.warning("pool_audit: _max_indexed_mtime: %s", _exc)
    return latest


def check_pool_health(pool_dir: Path) -> PoolHealth:
    """Diagnose a single pool directory. Returns a PoolHealth report."""
    issues: list[PoolHealthIssue] = []

    name = pool_dir.name
    config_path = pool_dir / "config.yaml"
    if not config_path.exists():
        config_path = pool_dir / "client.yaml"
    has_config = config_path.exists()

    # Parse config dict for validation checks
    config: dict = {}
    if has_config:
        try:
            import yaml
            config = yaml.safe_load(config_path.read_text()) or {}
        except Exception as _exc:
            logger.warning("pool_audit: check_pool_health: failed to parse config: %s", _exc)

    docs_path = pool_dir / "documents"
    index_path = pool_dir / "index"

    # Config validity (machine_name, embedding.model)
    if has_config:
        issues.extend(_check_config_validity(config))

    # Exclude pattern over-filtering
    if has_config:
        issues.extend(_check_exclude_patterns(pool_dir, config))

    # (d) broken symlinks
    issues.extend(_collect_symlink_issues(pool_dir))

    # File count under documents/ (excluding broken symlinks)
    file_count = _count_doc_files(docs_path)

    # Chroma chunk count + presence
    chunk_count, chroma_present, chroma_err = _chroma_chunk_count(index_path)
    if chroma_err:
        issues.append(PoolHealthIssue(
            "chroma_error",
            f"chroma read failed: {chroma_err}",
        ))

    # (e) never-indexed: config present, but no (real) chroma
    if has_config and not chroma_present:
        issues.append(PoolHealthIssue(
            "never_indexed",
            "config present, no chroma (run 'mullet reindex --pool "
            f"{name}')",
        ))

    # (a) empty-but-configured: config + docs with files + zero chunks
    if has_config and file_count > 0 and chunk_count == 0 and chroma_present:
        issues.append(PoolHealthIssue(
            "empty_but_configured",
            f"documents/ has {file_count} files but chroma has 0 chunks",
        ))

    # (b) stale manifest: docs modified after last index
    manifest_count, manifest_mtime = _load_manifest_meta(index_path)
    docs_mtime = _max_indexed_mtime(docs_path, config)
    if manifest_mtime is not None and docs_mtime > manifest_mtime:
        delta_days = (docs_mtime - manifest_mtime) / 86400.0
        issues.append(PoolHealthIssue(
            "stale_manifest",
            f"docs modified after last index (docs +{delta_days:.1f}d)",
        ))

    # (c) manifest/chroma mismatch: chunks < 0.5 * manifest file count
    if manifest_count > 0 and chunk_count < 0.5 * manifest_count:
        issues.append(PoolHealthIssue(
            "manifest_chroma_mismatch",
            f"chroma has {chunk_count} chunks but manifest lists "
            f"{manifest_count} files (expected ≥ {int(0.5 * manifest_count)})",
        ))

    # Last-indexed human string (from manifest mtime — closest proxy)
    last_indexed_human: str | None = None
    if manifest_mtime is not None:
        last_indexed_human = _fmt_delta_ago(time.time() - manifest_mtime)

    ok = len(issues) == 0
    return PoolHealth(
        name=name,
        pool_dir=pool_dir,
        ok=ok,
        chunk_count=chunk_count,
        file_count=file_count,
        last_indexed_human=last_indexed_human,
        issues=issues,
    )


def run_pool_health(pools_dir: Path, pool_name: str | None = None, workers: int = 1) -> int:
    """CLI entry point. Returns 0 if all pools are healthy, 1 otherwise.

    Args:
        pools_dir: Pools root directory.
        pool_name: Optional single-pool filter (slug). When None, audits all.
        workers: Parallel workers for health checks. Default 1 (serial).
    """
    if not pools_dir.exists():
        print(f"No pools directory at {pools_dir}")
        return 1

    if pool_name:
        pool_dir = pools_dir / pool_name
        if not pool_dir.is_dir():
            print(f"Pool not found: {pool_name} (in {pools_dir})")
            return 1
        targets = [pool_dir]
    else:
        targets = sorted(
            d for d in pools_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    if not targets:
        print(f"No pools found in {pools_dir}")
        return 0

    exit_code = 0
    name_width = max((len(d.name) for d in targets), default=10)
    name_width = max(name_width, 20)

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(check_pool_health, d) for d in targets]
            reports = [f.result() for f in futures]
    else:
        reports = [check_pool_health(d) for d in targets]

    reports.sort(key=lambda r: r.name)

    healthy = 0
    for report in reports:
        name_col = f"{report.name:<{name_width}}"
        if report.ok:
            healthy += 1
            indexed = (
                f"last indexed {report.last_indexed_human}"
                if report.last_indexed_human else "never indexed"
            )
            print(
                f"{name_col} OK ({report.chunk_count:,} chunks, "
                f"{report.file_count:,} files, {indexed})"
            )
        else:
            exit_code = 1
            # First line: aggregate warning code summary
            codes = ", ".join(i.code.replace("_", "-") for i in report.issues)
            print(f"{name_col} \u26a0 {codes}")
            for issue in report.issues:
                print(f"{' ' * name_width}   - {issue.message}")

    total = len(targets)
    if total > 1:
        unhealthy = total - healthy
        summary = f"\n{total} pools checked: {healthy} healthy"
        if unhealthy:
            summary += f", {unhealthy} need attention"
        print(summary)

    return exit_code


def run_pool_prune(pools_dir: Path, yes: bool = False) -> None:
    """Delete pools with no config.yaml, plus their mullet.db + sidecar state."""
    audits = audit_pools(pools_dir)
    candidates = [a for a in audits if not a.has_client_yaml]

    if not candidates:
        print("Nothing to prune — all pools have a config.yaml.")
        return

    print(f"{'Would remove' if not yes else 'Removing'} {len(candidates)} pool(s) with no config.yaml:\n")
    for a in candidates:
        size_mb = _dir_size_mb(a.pool_dir)
        chunks = a.vector.chunk_count
        print(f"  {'→' if yes else '·'} {a.name}  ({size_mb:.1f} MB, {chunks} chunks)")

    if not yes:
        print("\nDry run — pass --yes to delete.")
        return

    mullet_home = Path.home() / ".mullet"
    print()
    for a in candidates:
        slug = a.machine_name or a.name.lower().replace(" ", "_").replace("-", "_")
        ok, err = _force_rmtree(a.pool_dir)
        if not ok:
            hint = ""
            if err and ("permission" in err.lower() or "denied" in err.lower()):
                hint = f"\n      try: sudo rm -rf '{a.pool_dir}'"
            print(f"  \033[31m✗\033[0m {a.name}: {err}{hint}")
            continue

        rows, warnings = _purge_pool_sidecars(a.name, slug, mullet_home)
        extra = []
        if rows:
            extra.append(f"{rows} db row{'s' if rows != 1 else ''}")
        if warnings:
            extra.append(f"{len(warnings)} warning{'s' if len(warnings) != 1 else ''}")
        suffix = f"  ({', '.join(extra)})" if extra else ""
        print(f"  \033[32m✓\033[0m deleted {a.name}{suffix}")
        for w in warnings:
            print(f"      \033[33m!\033[0m {w}")


def _check_extension_coverage(pool_dir: Path, config: dict) -> list[PoolHealthIssue]:
    """Find file extensions present in documents/ but absent from coverage.

    Considers three sources of coverage so we don't false-positive on extensions
    handled by the code indexer or the new extension-less filenames opt-in:
      - documents.extensions      (the docs ingest path)
      - code.enabled + EXTRACTOR_MAP keys  (the code ingest path)
      - documents.filenames       (extension-less, picked up by name match)
    """
    docs = pool_dir / "documents"
    if not docs.exists() or not docs.is_dir():
        return []

    docs_cfg = config.get("documents") or {}
    configured_exts = set(docs_cfg.get("extensions") or [])
    exclude_exts = set(docs_cfg.get("exclude_extensions") or [])
    configured_filenames = set(docs_cfg.get("filenames") or [])

    if not configured_exts:
        return []  # no extension allowlist — all types go through

    # Extensions handled by the code-indexer (TypeScript, Python, JSON, etc.)
    code_exts: set[str] = set()
    code_cfg = config.get("code") or {}
    if code_cfg.get("enabled"):
        try:
            from mullet.knowledge.code.extractors import EXTRACTOR_MAP
            code_exts = set(EXTRACTOR_MAP.keys())
        except Exception:
            # If the import fails (rare — only if extractors module is broken),
            # fall back to a reasonable static superset so we don't false-flag.
            code_exts = {
                ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
                ".php", ".module", ".install", ".theme",
                ".html", ".htm", ".css", ".scss",
                ".json", ".yaml", ".yml", ".feature", ".vue", ".svelte",
            }

    import fnmatch
    exclude_patterns = docs_cfg.get("exclude_patterns") or []

    uncovered: dict[str, int] = {}
    try:
        for dirpath, _dirs, files in os.walk(str(docs), followlinks=True):
            for fname in files:
                if fname.startswith("."):
                    continue
                fpath = os.path.join(dirpath, fname)
                # Skip files excluded by exclude_patterns
                if any(fnmatch.fnmatch(fpath, p) for p in exclude_patterns):
                    continue
                # Skip files matching documents.filenames (extension-less opt-in)
                if fname in configured_filenames:
                    continue
                ext = Path(fname).suffix.lower()
                if (ext in configured_exts
                        or ext in exclude_exts
                        or ext in code_exts):
                    continue
                uncovered[ext] = uncovered.get(ext, 0) + 1
    except Exception as _exc:
        logger.warning("pool_audit: _check_extension_coverage: %s", _exc)

    total = sum(uncovered.values())
    if total < 5:
        return []

    top = sorted(uncovered.items(), key=lambda x: -x[1])[:8]
    ext_list = ", ".join(f"{e or '(no ext)'}({n})" for e, n in top)
    return [PoolHealthIssue(
        "uncovered_extensions",
        f"{total} files not indexed — extensions not in config: {ext_list}",
    )]


def _issue_fix_commands(
    issue: PoolHealthIssue,
    pool_dir: Path,
    config_path: Path,
    pool_name: str,
) -> list[str]:
    """Map one PoolHealthIssue to a list of actionable shell/comment fix lines."""
    code = issue.code

    if code in ("stale_manifest", "never_indexed", "empty_but_configured", "manifest_chroma_mismatch"):
        return [f"mullet reindex --pool {pool_name}"]

    if code == "broken_symlink":
        # message format: "documents/X/Y → target (target missing)"
        rel_str = issue.message.split(" → ")[0].strip()
        # Use the resolved pool_dir so the path works even when pools/ is itself a symlink
        abs_path = str(pool_dir.resolve() / rel_str)
        return [f"rm -f '{abs_path}'"]

    if code == "invalid_config":
        if "embedding.model" in issue.message:
            return [
                f"# Edit {config_path}",
                "# Add a top-level 'embedding' section, e.g.:",
                "#   embedding:",
                "#     model: all-MiniLM-L6-v2",
            ]
        if "machine_name" in issue.message:
            slug = pool_name.lower().replace("-", "_")
            return [
                f"# Edit {config_path}",
                f"# Add: machine_name: {slug}",
            ]
        return [f"# Edit {config_path}  # fix: {issue.message}"]

    if code == "uncovered_extensions":
        # Extract extension list from message
        try:
            ext_part = issue.message.split(": ", 2)[-1]
        except Exception:
            ext_part = ""
        return [
            f"# Edit {config_path}",
            "# Add missing extensions to documents.extensions",
            f"# Uncovered: {ext_part}",
        ]

    if code == "bad_exclude_pattern":
        return [
            f"# Edit {config_path}",
            "# Review/fix documents.exclude_patterns — currently filtering ≥90% of files",
        ]

    if code == "chroma_error":
        return [f"mullet reindex --pool {pool_name}  # chroma db error — rebuild index"]

    return [f"# {issue.message}"]


def run_pool_validate(pools_dir: Path, pool_name: str | None = None) -> int:
    """Deep pool validation with actionable fix commands.

    Returns 0 when all pools are clean, 1 when any have issues.
    """
    if not pools_dir.exists():
        print(f"No pools directory at {pools_dir}")
        return 1

    if pool_name:
        pool_dir = pools_dir / pool_name
        if not pool_dir.is_dir():
            print(f"Pool not found: {pool_name!r}")
            return 1
        targets = [pool_dir]
    else:
        # Skip:
        #   - hidden dirs (.cache, .git, etc.)
        #   - test_init_* pools created by tests/test_cli_e2e.py — they're
        #     test artifacts that leak into ~/.mullet/pools when the test
        #     suite runs the `init` wizard against a tmp_path without
        #     setting MULLET_POOLS_DIR. The validator should ignore them
        #     so nightly health reports don't get noise from leftover tests.
        targets = sorted(
            d for d in pools_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and not d.name.startswith("test_init_")
        )

    if not targets:
        print(f"No pools in {pools_dir}")
        return 0

    # Run health checks (parallel, 8 workers)
    from concurrent.futures import ThreadPoolExecutor

    def _check_one(pool_dir: Path):
        report = check_pool_health(pool_dir)
        # Load config for extension coverage check
        config_path = pool_dir / "config.yaml"
        if not config_path.exists():
            config_path = pool_dir / "client.yaml"
        config: dict = {}
        if config_path.exists():
            try:
                import yaml
                config = yaml.safe_load(config_path.read_text()) or {}
            except Exception:
                pass
        extra = _check_extension_coverage(pool_dir, config)
        all_issues = list(report.issues) + extra
        return report, config_path, all_issues

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_check_one, targets))

    # Print report
    bar = "─" * 66
    pools_with_issues = 0
    total_issues = 0
    all_fix_lines: list[str] = []

    for report, config_path, issues in sorted(results, key=lambda x: x[0].name):
        if not issues:
            continue
        pools_with_issues += 1
        total_issues += len(issues)
        print(f"\n── {report.name} {'─' * max(0, 64 - len(report.name))}")
        for issue in issues:
            icon = "✗" if issue.code == "invalid_config" else "⚠"
            print(f"  {icon} {issue.code.replace('_', '-')}: {issue.message}")
            fixes = _issue_fix_commands(issue, report.pool_dir, config_path, report.name)
            for fix in fixes:
                print(f"    {'FIX:':5} {fix}")
                all_fix_lines.append(fix)

    print(f"\n{bar}")
    total = len(targets)
    ok_count = total - pools_with_issues
    print(f"{ok_count} OK  ·  {pools_with_issues} with issues  ·  {total_issues} total issue(s)")
    if pools_with_issues == 0:
        print("All pools are healthy.")
        return 0

    # Consolidated fix block (runnable commands only, no comments)
    runnable = [ln for ln in all_fix_lines if not ln.startswith("#")]
    if runnable:
        print("\nRunnable fixes (copy-paste):")
        for cmd in runnable:
            print(f"  {cmd}")
    print("\nAfter fixing: mullet pool validate --all")
    return 1
