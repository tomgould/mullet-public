#!/usr/bin/env python3
"""
Trawl reindexer: document and code ingestion, graph building and pool sync.

Run it as ``mullet reindex`` or ``python -m mullet.knowledge.reindex``.
"""

import argparse
import fcntl
import logging
import os
import sys
import threading
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

# Prevent tokenizers fork warnings when using multiprocessing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Reduce VRAM fragmentation — PyTorch allocator grows segments instead of
# carving fixed blocks, which matters when embedding and graph extraction
# share the same GPU sequentially.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Deprioritise this reindex process so it can never starve interactive
# foreground work (e.g. Claude Code's API streaming). Reindex is
# throughput-oriented and tolerates low CPU/IO priority. Fail-open.
try:
    os.nice(19)
except Exception:
    pass
try:
    import subprocess as _sp
    from shutil import which as _which
    if _which("ionice"):
        _sp.run(["ionice", "-c", "3", "-p", str(os.getpid())],
                check=False, capture_output=True)
except Exception:
    pass

# Suppress ALL noise before any imports
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
# Preload langdetect profiles to avoid "Need to load profiles" spam
try:
    import langdetect
    langdetect.detect("preload")
except Exception:
    pass


from mullet.knowledge.engine.client import load_client, discover_clients
from mullet.knowledge.engine.search.searcher import HybridSearcher
from mullet.knowledge.engine.ingest import (
    DocumentIngestor,
    ingest_directory,
    ingest_client_code,
    build_file_manifest,
    load_file_manifest,
    diff_file_manifest,
    save_file_manifest,
)
from mullet.knowledge.graph.store import KnowledgeGraph
from mullet.core.env import get_pools_dir
from mullet.jobs import machine
from mullet.knowledge.graph.extractor import GraphExtractor

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)
# Suppress ALL library noise — only our print() output matters
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("mullet").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)  # suppress langdetect "No features in text."

# Worked out once when the script starts: TRAWL_POOLS_DIR, else the pools
# directory under Mullet's data directory (see mullet.core.paths).
TRAWL_POOLS_DIR = get_pools_dir()


class PhaseReporter:
    """Single-line live status for a pool during reindex.

    Uses \\r to overwrite the current line as phases advance, ending with
    a final newline via done(). Safe to call from any thread so long as
    only one reporter owns stdout at a time (we run pools serially).
    """
    WIDTH = 22      # Pool-name column width
    LINE_WIDTH = 100  # Clears long previous phase labels

    def __init__(self, pool_name: str, enabled: bool = True):
        import time as _t
        self._time = _t
        self.pool_name = pool_name
        self.enabled = enabled
        self.start = _t.monotonic()
        self.phase_start = self.start
        self.current_phase = ""
        # Use \r only when writing to a real terminal; file/pipe output gets \n
        self._cr = "\r" if (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()) else "\n"

    def phase(self, label: str) -> None:
        """Update the live line with a new phase label."""
        if not self.enabled:
            return
        elapsed = self._time.monotonic() - self.start
        self.current_phase = label
        self.phase_start = self._time.monotonic()
        msg = f"  {self.pool_name:<{self.WIDTH}} {label}  [{elapsed:.1f}s]"
        line = f"{msg:<{self.LINE_WIDTH}}" if self._cr == "\r" else msg
        sys.stdout.write(f"{self._cr}{line}")
        sys.stdout.flush()

    def tick(self, label: str) -> None:
        """Refresh the live line without resetting phase_start (for progress inside a phase)."""
        if not self.enabled:
            return
        elapsed = self._time.monotonic() - self.start
        msg = f"  {self.pool_name:<{self.WIDTH}} {label}  [{elapsed:.1f}s]"
        line = f"{msg:<{self.LINE_WIDTH}}" if self._cr == "\r" else msg
        sys.stdout.write(f"{self._cr}{line}")
        sys.stdout.flush()

    def done(self, summary: str) -> float:
        """Finalize with a summary line; returns total elapsed seconds."""
        elapsed = self._time.monotonic() - self.start
        msg = f"  {self.pool_name:<{self.WIDTH}} {summary}  {elapsed:.1f}s"
        if self.enabled:
            line = f"{msg:<{self.LINE_WIDTH}}" if self._cr == "\r" else msg
            sys.stdout.write(f"{self._cr}{line}\n")
            sys.stdout.flush()
        return elapsed

    def fail(self, error: str) -> float:
        elapsed = self._time.monotonic() - self.start
        msg = f"  {self.pool_name:<{self.WIDTH}} FAILED: {error}  {elapsed:.1f}s"
        if self.enabled:
            line = f"{msg:<{self.LINE_WIDTH}}" if self._cr == "\r" else msg
            sys.stdout.write(f"{self._cr}{line}\n")
            sys.stdout.flush()
        return elapsed


def _fmt_eta(seconds: float) -> str:
    """Format a duration as e.g. '3m42s' or '45s' or '1h05m'."""
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


_WAL_FLUSH_THRESHOLD = 500  # embeddings_queue rows above this will hang the compactor


def _check_chroma_wal(index_path, reporter=None) -> bool:
    """Detect and flush a stuck ChromaDB WAL before opening the collection.

    ChromaDB's Rust compactor deadlocks when embeddings_queue has a large
    backlog it cannot apply to the HNSW writer. We can safely DELETE those
    rows via sqlite3 (no ChromaDB process involved) — the embeddings are
    already persisted in the main embeddings table; the queue rows are only
    needed for the compactor's backfill pass.

    Returns True if a flush was performed.
    """
    import sqlite3 as _sqlite3

    db_path = Path(index_path) / "chroma.sqlite3"
    if not db_path.exists():
        return False
    try:
        conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        count = conn.execute("SELECT COUNT(*) FROM embeddings_queue").fetchone()[0]
        conn.close()
    except Exception:
        return False

    if count < _WAL_FLUSH_THRESHOLD:
        return False

    # Flush the stuck WAL
    msg = f"WARNING: {db_path.parent.name} has {count} stuck WAL rows — flushing before open"
    if reporter:
        reporter.phase(f"WAL flush ({count} rows)")
    else:
        print(f"\n  {msg}", flush=True)
        logger.warning(msg)
    try:
        conn = _sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM embeddings_queue")
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        logger.warning(f"WAL flush failed for {db_path}: {exc}")
        return False


def _cheap_collection(client):
    """Get a ChromaDB collection without constructing HybridSearcher.

    HybridSearcher.__init__ loads a SentenceTransformer (~2-3s per call). For
    callers that only need .collection (chunk count, graph source), this
    avoids the embedding-model warmup entirely.

    Must use identical Settings() to HybridSearcher — ChromaDB caches
    PersistentClient instances per (path, settings) and will raise
    "An instance of Chroma already exists" if a call uses a different
    Settings object than an earlier call for the same path.
    """
    import chromadb
    from chromadb.config import Settings
    cc = chromadb.PersistentClient(
        path=str(client.index_path),
        settings=Settings(anonymized_telemetry=False),
    )
    return cc.get_or_create_collection(
        name=client.machine_name,
        metadata={"hnsw:space": "cosine"},
    )


def _bm25_cache_is_stale(client) -> bool:
    """Heuristic: return True if the pool's BM25 cache is older than any HNSW
    binary, i.e. Chroma chunks were added/removed after the cache was written.

    This catches the case where an earlier reindex saved the manifest + upserted
    to Chroma but was killed before the final BM25 rebuild. The file manifest
    would look "up to date" on the next run, but the BM25 cache is stale.
    """
    index_dir = Path(client.index_path)
    cache_path = index_dir / "bm25_cache.json"
    if not cache_path.exists():
        return True  # no cache to be stale, but rebuild is needed anyway
    try:
        cache_mtime = cache_path.stat().st_mtime
        for sub in index_dir.iterdir():
            if not sub.is_dir():
                continue
            for name in ("data_level0.bin", "header.bin", "length.bin", "link_lists.bin"):
                f = sub / name
                if f.exists() and f.stat().st_mtime > cache_mtime:
                    return True
        return False
    except Exception:
        return False

def sync_client_sources(client_name: str) -> dict:
    """
    Sync all configured sources for a client.
    """
    client = load_client(client_name, clients_dir=TRAWL_POOLS_DIR)
    sources_config = client.sources_config
    if not sources_config:
        
        return {"synced": False}

    sources_dir = str(client.sources_path)
    results = []

    for gh_cfg in sources_config.get("github", []):
        try:
            from mullet.knowledge.sources.github_models import GitHubSourceConfig
            from mullet.knowledge.sources.github_connector import GitHubSourceConnector
            config = GitHubSourceConfig.from_dict(gh_cfg)
            connector = GitHubSourceConnector(config)
            result = connector.sync(sources_dir)
            results.append(result)
            logger.info(
                "GitHub sync (%s): +%d updated=%d errors=%d",
                config.owner, result.files_added, result.files_updated, len(result.errors),
            )
            for err in result.errors:
                logger.warning("  %s", err)
        except Exception as e:
            logger.error("GitHub source sync failed: %s", e)

    for web_cfg in sources_config.get("web", []):
        try:
            from mullet.knowledge.sources.web_crawler import WebSourceConfig, WebSourceConnector
            config = WebSourceConfig.from_dict(web_cfg)
            connector = WebSourceConnector(config)
            result = connector.sync(sources_dir)
            results.append(result)
            logger.info(
                "Web sync (%s): +%d updated=%d errors=%d",
                config.domain, result.files_added, result.files_updated, len(result.errors),
            )
            for err in result.errors:
                logger.warning("  %s", err)
        except Exception as e:
            logger.error("Web source sync failed: %s", e)

    total_added = sum(r.files_added for r in results)
    total_updated = sum(r.files_updated for r in results)
    total_errors = sum(len(r.errors) for r in results)
    logger.info(
        "Source sync complete: +%d updated=%d errors=%d",
        total_added, total_updated, total_errors,
    )

    return {
        "synced": True,
        "files_added": total_added,
        "files_updated": total_updated,
        "errors": total_errors,
    }

def _should_sync_on_ingest(client):
    """
    Check if any configured source has sync_on_ingest enabled.
    """
    sources_config = client.sources_config
    if not sources_config:
        return False
    for gh_cfg in sources_config.get("github", []):
        if gh_cfg.get("sync_on_ingest", False):
            return True
    for web_cfg in sources_config.get("web", []):
        if web_cfg.get("sync_on_ingest", False):
            return True
    return False

def get_chunk_ids_for_file(searcher: HybridSearcher, file_path: str) -> list[str]:
    """
    Get existing chunk IDs for a file from ChromaDB by source_path metadata.
    """
    try:
        results = searcher.collection.get(
            where={"source_path": file_path},
            include=[],
        )
        return results.get("ids", [])
    except Exception:
        return []

def incremental_reindex(
    client_name: str,
    include_code: bool = True,
    reporter: "PhaseReporter | None" = None,
) -> str:
    """
    Incrementally reindex only changed files for a client.
    """
    client = load_client(client_name, clients_dir=TRAWL_POOLS_DIR)

    docs_path = client.documents_path
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_path}")
        return

    embedding_config = client.embedding_config
    embedding_model = embedding_config.get("model", "all-MiniLM-L6-v2")
    passage_prefix = embedding_config.get("passage_prefix", "")

    doc_extensions = client.get_document_extensions()
    doc_filenames = client.get_document_filenames()
    doc_exclude_patterns = client.documents_config.get("exclude_patterns", [])

    if _should_sync_on_ingest(client):
        if reporter:
            reporter.phase("syncing sources")
        sync_client_sources(client_name)

    if reporter:
        reporter.phase("scanning files")
    current_files = build_file_manifest(
        docs_path, doc_extensions,
        exclude_patterns=doc_exclude_patterns,
        filenames=doc_filenames,
    )
    sources_path = client.sources_path
    if sources_path.exists():
        sources_manifest = build_file_manifest(
            sources_path, doc_extensions,
            exclude_patterns=doc_exclude_patterns,
            filenames=doc_filenames,
        )
        current_files.update(sources_manifest)

    indexed_files = load_file_manifest(client)

    if not indexed_files:
        # No manifest but index exists — just save current manifest for future diffs
        # Don't rebuild — the existing index data is fine
        try:
            from mullet.knowledge.engine.search.searcher import HybridSearcher as _HS
            s = _HS(index_path=str(client.index_path), collection_name=client.machine_name, embedding_config=embedding_config, skip_bm25_load=True)
            existing = s.collection.count()
        except Exception:
            existing = 0
        if existing > 0:
            save_file_manifest(client, current_files)
            return "up_to_date"
        # Truly empty — do a full rebuild
        print(f"  {client.name:<22} empty index, rebuilding...", flush=True)
        rebuild_index(client_name, include_code=include_code)
        return "rebuilt"

    if reporter:
        reporter.phase("diffing manifest")
    new_files, modified_files, deleted_files = diff_file_manifest(current_files, indexed_files)

    if not new_files and not modified_files and not deleted_files:
        # Files are aligned with Chroma, but verify the BM25 cache isn't stale
        # (e.g., from an earlier killed reindex that saved the manifest but
        # didn't finish the BM25 rebuild). If it's stale, rebuild just BM25.
        if _bm25_cache_is_stale(client):
            if reporter:
                reporter.phase("BM25 cache stale — rebuilding only")
            # Open collection with skip_bm25_load and trigger a rebuild
            searcher = HybridSearcher(
                index_path=str(client.index_path),
                collection_name=client.machine_name,
                bm25_weight=client.search_config.get("bm25_weight", 0.5),
                vector_weight=client.search_config.get("vector_weight", 0.5),
                tokenizer_config=client.tokenization_config,
                embedding_config=embedding_config,
                skip_bm25_load=True,
                progress_reporter=reporter,
            )
            searcher.rebuild_bm25_index()
            return "bm25_rebuilt"
        return "up_to_date"

    if reporter:
        reporter.phase(
            f"changes: +{len(new_files)} ~{len(modified_files)} -{len(deleted_files)}"
        )
    # Create searcher and ingestor
    if reporter:
        reporter.phase("loading embedding model")
    ingestor = DocumentIngestor(
        embedding_model=embedding_model,
        passage_prefix=passage_prefix,
    )
    # skip_bm25_load=True: we're about to mutate the corpus. Loading the old
    # cache and re-validating it would waste minutes on a big pool, only to
    # throw it away. We rebuild BM25 ONCE at the end of the ingest loop.
    _check_chroma_wal(client.index_path, reporter)
    if reporter:
        reporter.phase("opening collection (deferred BM25 rebuild)")
    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=client.search_config.get("bm25_weight", 0.5),
        vector_weight=client.search_config.get("vector_weight", 0.5),
        tokenizer_config=client.tokenization_config,
        embedding_config=embedding_config,
        skip_bm25_load=True,
        progress_reporter=reporter,
    )

    total_added = 0
    total_removed = 0

    # Delete chunks for removed files
    for fp in deleted_files:
        try:
            results = searcher.collection.get(where={"source_path": fp}, include=[])
            old_ids = results.get("ids", [])
            if old_ids:
                searcher.collection.delete(ids=old_ids)
                total_removed += len(old_ids)
        except Exception:
            pass

    # Delete old chunks for modified files
    for fp in modified_files:
        try:
            results = searcher.collection.get(where={"source_path": str(fp)}, include=[])
            old_ids = results.get("ids", [])
            if old_ids:
                searcher.collection.delete(ids=old_ids)
                total_removed += len(old_ids)
        except Exception:
            pass

    # Ingest new + modified files. defer_bm25_rebuild=True prevents the per-file
    # full-BM25-rebuild + 213 MB JSON-save cycle (measured at 20-30s per file on
    # a 40k-chunk pool; 159 files = 60+ minutes wasted). One rebuild at the end.
    files_to_ingest = [Path(f) for f in new_files + modified_files]
    total_files = len(files_to_ingest)
    if files_to_ingest:
        for idx, fp in enumerate(files_to_ingest, 1):
            if reporter and (idx == 1 or idx % 5 == 0 or idx == total_files):
                reporter.tick(f"ingesting {idx}/{total_files}  ({fp.name[:40]})")
            try:
                chunks, embeddings = ingestor.ingest_document(fp)
                if chunks:
                    searcher.index_chunks(chunks, embeddings, defer_bm25_rebuild=True)
                    total_added += len(chunks)
            except Exception as e:
                print(f"    warn: {fp.name}: {e}", file=sys.stderr)

        # Save manifest BEFORE the BM25 rebuild — if the rebuild is killed or
        # crashes, Chroma is already updated, and the next run will see an
        # up-to-date manifest rather than re-ingesting the same files. BM25
        # can always be rebuilt from Chroma state alone.
        if reporter:
            reporter.phase("saving manifest")
        save_file_manifest(client, current_files)

        # One BM25 rebuild for the whole batch. Touches Chroma once, writes
        # the cache once, instead of per-file.
        if reporter:
            reporter.phase(f"rebuilding BM25 ({total_added} new chunks)")
        searcher.rebuild_bm25_index()
    else:
        # Nothing ingested — still save a fresh manifest so future runs see
        # the current state (important after an interrupted rebuild).
        if reporter:
            reporter.phase("saving manifest")
        save_file_manifest(client, current_files)

    if include_code:
        if reporter:
            reporter.phase("ingesting code")
        ingest_client_code_files(client_name, rebuild=False, reporter=reporter)


def rebuild_index(client_name: str, include_code: bool = True, reporter=None) -> None:
    """Full rebuild — clear index and re-ingest everything."""
    client = load_client(client_name, clients_dir=TRAWL_POOLS_DIR)
    if _should_sync_on_ingest(client):
        if reporter:
            reporter.phase("syncing sources")
        sync_client_sources(client_name)
    ingest_client_documents(client_name, resume=False, reporter=reporter)
    if include_code:
        ingest_client_code_files(client_name, rebuild=True, reporter=reporter)

def ingest_client_documents(client_name: str, resume: bool = False, reporter=None) -> None:
    """Ingest documents for a pool using progressive pipeline."""
    client = load_client(client_name, clients_dir=TRAWL_POOLS_DIR)
    docs_path = client.documents_path
    if not docs_path.exists():
        print(f"  {client.name:<22} ERROR: documents dir not found")
        return

    emb_cfg = client.embedding_config
    if reporter:
        reporter.phase("loading embedding model")
    ingestor = DocumentIngestor(
        embedding_model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
        passage_prefix=emb_cfg.get("passage_prefix", ""),
    )
    doc_extensions = client.get_document_extensions()
    doc_filenames = client.get_document_filenames()
    doc_exclude = client.documents_config.get("exclude_patterns", [])

    # skip_bm25_load=True: this path CLEARS the index and rebuilds from scratch
    # when resume=False. Loading the old cache just to throw it away is pure waste.
    # We rebuild BM25 once at the end after all ingestion.
    _check_chroma_wal(client.index_path, reporter)
    if reporter:
        reporter.phase("opening collection (deferred BM25 rebuild)")
    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=client.search_config.get("bm25_weight", 0.5),
        vector_weight=client.search_config.get("vector_weight", 0.5),
        tokenizer_config=client.tokenization_config,
        embedding_config=emb_cfg,
        skip_bm25_load=True,
        progress_reporter=reporter,
    )

    # Clear existing index if not resuming
    if not resume:
        try:
            existing = searcher.collection.count()
            if existing > 0:
                if reporter:
                    reporter.phase(f"clearing index ({existing} chunks)")
                searcher.chroma_client.delete_collection(client.machine_name)
                searcher.collection = searcher.chroma_client.get_or_create_collection(
                    name=client.machine_name, metadata={"hnsw:space": "cosine"})
        except Exception:
            pass

    # Ingest with pipeline (documents/)
    if reporter:
        reporter.phase("ingesting documents/")
    ingest_directory(
        docs_path, ingestor, extensions=doc_extensions,
        exclude_patterns=doc_exclude,
        pipeline_searcher=searcher,
        contextualize=emb_cfg.get("contextualize", False),
        filenames=doc_filenames,
    )

    # Also ingest from sources/ if it exists
    sources_path = client.sources_path
    if sources_path.exists() and any(sources_path.iterdir()):
        if reporter:
            reporter.phase("ingesting sources/")
        ingest_directory(
            sources_path, ingestor, extensions=doc_extensions,
            exclude_patterns=doc_exclude,
            pipeline_searcher=searcher,
            filenames=doc_filenames,
        )

    # Final BM25 rebuild — pipeline ingest goes direct to Chroma and never
    # touches BM25. Without this call the cache stays stale after a rebuild.
    if reporter:
        reporter.phase("rebuilding BM25")
    searcher.rebuild_bm25_index()

    # Save manifest
    if reporter:
        reporter.phase("saving manifest")
    manifest = build_file_manifest(
        docs_path, doc_extensions,
        exclude_patterns=doc_exclude, filenames=doc_filenames,
    )
    if sources_path.exists():
        manifest.update(build_file_manifest(
            sources_path, doc_extensions,
            exclude_patterns=doc_exclude, filenames=doc_filenames,
        ))
    save_file_manifest(client, manifest)


def ingest_client_code_files(client_name: str, rebuild: bool = False, reporter=None) -> None:
    """Ingest code files for a pool if code indexing is enabled."""
    client = load_client(client_name, clients_dir=TRAWL_POOLS_DIR)
    code_config = client.config.get("code", {})
    if not code_config.get("enabled", False):
        return

    emb_cfg = client.embedding_config
    if reporter:
        reporter.phase("code: loading embedding model")
    ingestor = DocumentIngestor(
        embedding_model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
        passage_prefix=emb_cfg.get("passage_prefix", ""),
    )

    # skip_bm25_load=True: pipeline ingest goes direct to Chroma and never
    # touches BM25. We rebuild once at the end instead of loading + validating
    # + rebuilding inside the constructor.
    if reporter:
        reporter.phase("code: opening collection (deferred BM25 rebuild)")
    searcher = HybridSearcher(
        index_path=str(client.index_path),
        collection_name=client.machine_name,
        bm25_weight=client.search_config.get("bm25_weight", 0.5),
        vector_weight=client.search_config.get("vector_weight", 0.5),
        tokenizer_config=client.tokenization_config,
        embedding_config=emb_cfg,
        skip_bm25_load=True,
        progress_reporter=reporter,
    )

    if reporter:
        reporter.phase("code: scanning source files")
    ingest_client_code(client, ingestor, rebuild=rebuild, pipeline_searcher=searcher)

    # Final BM25 rebuild after all code chunks are in Chroma
    if reporter:
        reporter.phase("code: rebuilding BM25")
    searcher.rebuild_bm25_index()

def _create_graph_provider(client, reporter=None):
    """Create a local LLM provider for graph extraction (LM Studio only).

    Checks that LM Studio is running and has a model loaded before proceeding.
    Graph extraction never uses API providers — it would rack up unexpected bills.
    """
    import urllib.request
    import json as _json
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider

    llm_config = getattr(client, "llm_config", {}) or {}
    host = llm_config.get("lmstudio_host", "http://localhost:1234")

    try:
        with urllib.request.urlopen(f"{host}/v1/models", timeout=2) as resp:
            models = [
                m for m in _json.loads(resp.read()).get("data", [])
                if "embedding" not in m["id"].lower()
            ]
    except Exception:
        models = []

    if not models:
        msg = "graph: skipped — no model loaded in LM Studio. Load a model and re-run."
        if reporter:
            reporter.phase(msg)
        else:
            print(f"  {client.name:<22} GRAPH SKIPPED: no local model loaded")
        return None

    graph_model = llm_config.get("graph_model") or models[0]["id"]
    if reporter:
        reporter.phase(f"graph: using local {graph_model}")
    return OpenAICompatibleProvider(base_url=f"{host}/v1", model=graph_model)


def build_graph(client_name: str, rebuild: bool = False, reporter: "PhaseReporter | None" = None, max_chunks: int | None = None) -> None:
    """Build or update knowledge graph for a pool."""
    client = load_client(client_name, clients_dir=TRAWL_POOLS_DIR)
    graph_config = client.graph_config if hasattr(client, 'graph_config') else {}
    if not graph_config.get("enabled", False):
        return

    # Suppress extractor noise
    logging.getLogger("mullet.knowledge.graph").setLevel(logging.ERROR)

    if reporter:
        reporter.phase("graph: loading")
    graph = KnowledgeGraph(client.index_path)
    mode = "rebuild" if rebuild else "update"
    if rebuild:
        graph.clear()
    else:
        graph.load()

    # Get collection directly — graph build only needs .collection, and HybridSearcher
    # would waste ~2-3s loading a SentenceTransformer plus (historically) the BM25 cache.
    try:
        collection = _cheap_collection(client)
    except Exception as _coll_e:
        if reporter:
            reporter.phase(f"graph: ERROR (no collection: {str(_coll_e)[:60]})")
        else:
            print(f"  {client.name:<22} GRAPH ERROR: no collection ({_coll_e})")
        return

    count = collection.count()
    if count == 0:
        if reporter:
            reporter.phase("graph: no chunks")
        else:
            print(f"  {client.name:<22} GRAPH: no chunks")
        return

    # Get unprocessed chunks
    processed_ids = set() if rebuild else graph.processed_chunk_ids
    exclude_exts = graph_config.get("exclude_extensions", [])
    unprocessed_ids = []

    if reporter:
        reporter.phase(f"graph: scanning {count} chunks")
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

    if max_chunks and len(unprocessed_ids) > max_chunks:
        if reporter:
            reporter.phase(f"graph: capped at {max_chunks}/{len(unprocessed_ids)} chunks (--max-graph-chunks)")
        unprocessed_ids = unprocessed_ids[:max_chunks]

    if not unprocessed_ids:
        s = graph.stats
        logger.info(f"Graph up to date: {s.get('entity_count', 0)} entities, {s.get('relationship_count', 0)} relationships")
        if reporter:
            reporter.phase(
                f"graph: up to date  ({s.get('entity_count', 0)} ent, {s.get('relationship_count', 0)} rel)"
            )
        # Prune orphaned chunks whose source no longer exists in the collection
        if reporter:
            reporter.phase("graph: pruning orphans")
        try:
            existing_ids: set[str] = set()
            count = collection.count()
            for _off in range(0, count, 10000):
                _r = collection.get(include=[], limit=10000, offset=_off)
                existing_ids.update(_r["ids"])
            pruned = graph.prune_orphaned_chunks(existing_ids)
            if pruned["removed_chunks"] or pruned["removed_entities"] or pruned["removed_relationships"]:
                graph.save()
                if reporter:
                    reporter.phase(
                        f"graph: pruned {pruned['removed_chunks']} chunks, "
                        f"{pruned['removed_entities']} entities, "
                        f"{pruned['removed_relationships']} rels"
                    )
        except Exception as _prune_e:
            logger.warning(f"Graph prune failed: {_prune_e}")
        return

    # Fetch content
    if reporter:
        reporter.phase(f"graph: fetching {len(unprocessed_ids)} chunks")
    all_chunks = []
    for i in range(0, len(unprocessed_ids), 5000):
        batch = unprocessed_ids[i:i + 5000]
        results = collection.get(ids=batch, include=["documents", "metadatas"])
        for j in range(len(results["ids"])):
            all_chunks.append({
                "id": results["ids"][j],
                "content": results["documents"][j],
                "metadata": results["metadatas"][j] if results["metadatas"] else {},
            })

    if not reporter:
        print(f"  {client.name:<22} GRAPH {mode}  {len(all_chunks)} chunks to process")

    # Create LLM provider — auto-select: local → Kimi → Gemini
    provider = _create_graph_provider(client, reporter)
    if provider is None:
        return

    def get_llm_response(messages, system_prompt, max_tokens):
        full_messages = [{"role": "system", "content": system_prompt}] + messages if system_prompt else messages
        # temperature=0.0: deterministic JSON extraction; reduces SPECIAL_* token spam.
        # Thinking off: extraction doesn't need it, and on a local reasoning model
        # it costs time and can use up max_tokens before any JSON is written.
        return provider.create(
            full_messages, max_tokens=max_tokens, temperature=0.0, reasoning_effort="none",
        ).content

    import threading
    num_workers = graph_config.get("extract_workers", 1)
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

    _graph_lock = threading.Lock()
    _last_e = [0]
    _last_r = [0]
    import time as _time_mod
    _extract_start = _time_mod.monotonic()

    def _on_progress(progress, entities, relationships):
        with _graph_lock:
            for e in entities[_last_e[0]:]:
                graph.add_entity(e)
            for r in relationships[_last_r[0]:]:
                graph.add_relationship(r)
            _last_e[0] = len(entities)
            _last_r[0] = len(relationships)
            graph.mark_chunks_processed([c["id"] for c in all_chunks[:progress.processed_chunks]])
            graph.save()
        pct = (progress.processed_chunks / len(all_chunks) * 100) if all_chunks else 0
        dt = _time_mod.monotonic() - _extract_start
        rate = progress.processed_chunks / dt if dt > 0 else 0.0
        remaining = (len(all_chunks) - progress.processed_chunks) / rate if rate > 0 else 0
        label = (
            f"graph: {mode} {progress.processed_chunks}/{len(all_chunks)} "
            f"({pct:.0f}%, {rate:.1f} ch/s, eta {_fmt_eta(remaining)})  "
            f"{len(entities)} ent  {len(relationships)} rel"
        )
        if reporter:
            reporter.tick(label)
        else:
            sys.stdout.write(f"\r  {client.name:<22} GRAPH {mode}  {pct:.0f}%  {len(entities)} entities  {len(relationships)} rels        ")
            sys.stdout.flush()

    entities, relationships = extractor.extract_from_chunks(all_chunks, progress_callback=_on_progress)

    # Final save
    with _graph_lock:
        for e in entities[_last_e[0]:]:
            graph.add_entity(e)
        for r in relationships[_last_r[0]:]:
            graph.add_relationship(r)
        # Only mark chunks as processed up to whatever we actually got through —
        # if we bailed out on a broken LLM, don't lie about the unprocessed ones.
        processed_so_far = [c["id"] for c in all_chunks[:len(entities) + len(relationships) + 1]]
        graph.mark_chunks_processed(processed_so_far)
        graph.save()

    # Prune orphaned chunks whose source no longer exists in the collection
    if reporter:
        reporter.phase("graph: pruning orphans")
    try:
        existing_ids: set[str] = set()
        count = collection.count()
        for _off in range(0, count, 10000):
            _r = collection.get(include=[], limit=10000, offset=_off)
            existing_ids.update(_r["ids"])
        pruned = graph.prune_orphaned_chunks(existing_ids)
        if pruned["removed_chunks"] or pruned["removed_entities"] or pruned["removed_relationships"]:
            graph.save()
            if reporter:
                reporter.phase(
                    f"graph: pruned {pruned['removed_chunks']} chunks, "
                    f"{pruned['removed_entities']} entities, "
                    f"{pruned['removed_relationships']} rels"
                )
    except Exception as _prune_e:
        logger.warning(f"Graph prune failed: {_prune_e}")

    fatal = getattr(extractor, "_fatal_error", None)
    s = graph.stats
    graph.close()
    if fatal:
        # Model is broken — surface this clearly and DON'T claim success.
        msg = f"graph: ABORTED ({fatal[:80]}…)  {s.get('entity_count', 0)} ent  {s.get('relationship_count', 0)} rel"
        if reporter:
            reporter.phase(msg)
        else:
            print(f"\r  {client.name:<22} {msg}")
    elif reporter:
        reporter.phase(
            f"graph: DONE  {s.get('entity_count', 0)} ent  {s.get('relationship_count', 0)} rel"
        )
    else:
        print(f"\r  {client.name:<22} GRAPH DONE  {s.get('entity_count', 0)} entities  {s.get('relationship_count', 0)} rels                    ")


@contextmanager
def _pool_lock(name: str):
    """Hold an exclusive per-pool lock; yields False if another process holds it.

    Two reindexes writing one Chroma index at once (e.g. the nightly timer and
    a manual ``--all``) can corrupt it or duplicate chunks.
    """
    lock_path = Path(TRAWL_POOLS_DIR) / name / ".reindex.lock"
    if not lock_path.parent.is_dir():
        yield True  # unknown pool: let the reindex report its usual error
        return
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# Exit status of `--no-wait` when a pool was skipped because the GPU or the
# machine was busy (EX_TEMPFAIL in sysexits.h: try again later).
EXIT_MACHINE_BUSY = 75

# After taking the GPU lease, how long to re-check that the machine still has room.
_RECHECK_S = 30.0

# Every line a pool prints outside its PhaseReporter, so --workers threads
# don't interleave them.
_print_lock = threading.Lock()


def _say(name: str, text: str) -> None:
    with _print_lock:
        print(f"  {name:<{PhaseReporter.WIDTH}} {text}", flush=True)


def _release_gpu_memory() -> None:
    """Drop cached embedding models and CUDA's cached memory, if a model was loaded."""
    model_cache = sys.modules.get("mullet.knowledge.engine.model_cache")
    if model_cache is None:
        return
    try:
        model_cache.get_model_cache().clear()
    except Exception as exc:
        logger.warning(f"reindex: couldn't release GPU memory: {exc}")


def _cpu_only():
    """Load this thread's embedding models on the CPU."""
    from mullet.knowledge.engine.model_cache import use_device
    return use_device("cpu")


@contextmanager
def _machine_turn(name: str, *, wait: bool = True):
    """Get the machine ready for one pool's indexing.

    With a GPU: wait for headroom, then take the GPU lease, then re-check
    briefly. The whole wait lasts ``MULLET_GPU_WAIT_MAX_S`` at most (900 s by
    default); if the GPU is still busy after that, the pool's embeddings run
    on the CPU instead, without the lease. Without a GPU there is no lease
    and only a short wait for RAM and load (see ``machine.wait_for_headroom``).

    Prints a line when it has to wait, saying what for, and one when it
    starts. With wait=False it raises ``machine.MachineBusy`` at once instead.
    Cached models are released before the lease is.
    """
    started = time.monotonic()
    budget = machine.max_wait_s() if wait else 0.0
    waited = False

    def left() -> str:
        return _fmt_eta(max(0.0, budget - (time.monotonic() - started)))

    def on_room(room, shortfalls) -> None:
        nonlocal waited
        waited = True
        reasons = "; ".join(s.explain() for s in shortfalls)
        if room.has_gpu:
            _say(name, f"waiting up to {left()} for headroom, then the CPU (--no-wait skips): {reasons}")
        else:
            _say(name, f"no GPU; waiting up to {_fmt_eta(min(budget, machine.NO_GPU_MAX_WAIT_S))} "
                       f"for RAM and load, then starting anyway: {reasons}")

    def on_gpu(holder) -> None:
        nonlocal waited
        waited = True
        who = holder.describe() if holder else "another Mullet process (it left no record)"
        _say(name, f"waiting up to {left()} for the GPU, then the CPU (--no-wait skips): held by {who}")

    lease = ExitStack()
    busy: machine.MachineBusy | None = None
    try:
        room = machine.wait_for_headroom(timeout=budget, on_wait=on_room)
        if room.has_gpu:
            remaining = max(0.0, budget - (time.monotonic() - started))
            lease.enter_context(machine.gpu_lease(f"reindex of pool {name}", wait=wait,
                                                  timeout=remaining, on_wait=on_gpu))
            machine.wait_for_headroom(timeout=min(_RECHECK_S, budget), on_wait=on_room)
    except machine.MachineBusy as exc:
        lease.close()
        if not wait:
            raise
        busy = exc
    except BaseException:
        lease.close()
        raise
    with lease, (_cpu_only() if busy else nullcontext()):
        if busy:
            _say(name, f"{busy}; indexing on the CPU instead")
        elif waited:
            _say(name, f"starting after waiting {_fmt_eta(time.monotonic() - started)}")
        try:
            yield
        finally:
            _release_gpu_memory()


def _reindex_one(name: str, args, quiet: bool = False, sync_sources: bool = False) -> float:
    """Reindex a single pool under its per-pool lock. Returns total elapsed seconds.

    quiet=True disables PhaseReporter's live ``\\r`` output (used in parallel
    mode where multiple pools would otherwise fight over stdout).
    sync_sources=True syncs the pool's external sources first (``--sync-sources``).
    The indexing itself runs once the machine has room (see ``_machine_turn``);
    with ``--no-wait`` a busy machine raises ``machine.MachineBusy`` instead.
    """
    with _pool_lock(name) as locked:
        if not locked:
            reporter = PhaseReporter(name, enabled=not quiet)
            return reporter.fail("skipped: another reindex of this pool is running")
        if sync_sources:
            sync_client_sources(name)
        with _machine_turn(name, wait=getattr(args, "no_wait", False) is not True):
            return _reindex_one_locked(name, args, quiet)


def _reindex_one_locked(name: str, args, quiet: bool = False) -> float:
    """Reindex a single pool; the caller holds its lock. Returns elapsed seconds."""
    reporter = PhaseReporter(name, enabled=not quiet)
    reporter.phase("starting")
    include_code = not getattr(args, 'no_code', False)

    result = None
    try:
        if getattr(args, 'code_only', False) or getattr(args, 'rebuild_code', False):
            reporter.phase("code-only reindex")
            ingest_client_code_files(name, rebuild=getattr(args, 'rebuild_code', False), reporter=reporter)
        elif getattr(args, 'rebuild', False):
            reporter.phase("full rebuild")
            rebuild_index(name, include_code=include_code, reporter=reporter)
        else:
            result = incremental_reindex(name, include_code=include_code, reporter=reporter)
    except Exception as e:
        return reporter.fail(str(e))

    # Cheap chunk count — bypass HybridSearcher entirely to skip SentenceTransformer load.
    # If chroma is unavailable (locked, mid-rebuild, corrupted), fall back to the
    # manifest file count so the summary is still informative instead of a bare "?".
    try:
        client = load_client(name, clients_dir=TRAWL_POOLS_DIR)
    except Exception as e:
        logger.warning(f"reindex: could not load client {name} for chunk count: {e}")
        chunks = "?"
    else:
        try:
            chunks = _cheap_collection(client).count()
        except Exception as e:
            logger.warning(
                f"reindex: chroma chunk count unavailable for {name}: {e}; "
                f"falling back to manifest file count"
            )
            try:
                manifest = load_file_manifest(client)
                chunks = f"~{len(manifest)} files" if manifest else "?"
            except Exception:
                chunks = "?"

    if result == "up_to_date":
        status = "UP TO DATE"
    elif result == "bm25_rebuilt":
        status = "BM25 REBUILT"
    else:
        status = "DONE"

    # Graph building (may print its own phases and final line)
    graph_enabled = getattr(args, 'graph', False) or getattr(args, 'graph_rebuild', False)
    if graph_enabled:
        try:
            _check_chroma_wal(client.index_path, reporter)
            build_graph(
                name,
                rebuild=getattr(args, 'graph_rebuild', False),
                reporter=reporter,
                max_chunks=getattr(args, 'max_graph_chunks', None),
            )
        except Exception as e:
            return reporter.fail(f"graph: {e}")

    # If `chunks` is a numeric count, append "chunks"; if it's already a
    # descriptive fallback string like "~711 files", don't double-label.
    if isinstance(chunks, int):
        summary_tail = f"{chunks} chunks"
    else:
        summary_tail = str(chunks)
    return reporter.done(f"{status}  {summary_tail}")


def main():
    # Suppress noisy warnings
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*pynvml.*")

    parser = argparse.ArgumentParser(description="Mullet Trawl Reindex — direct in-process ingestion")
    parser.add_argument("--pool", "-c", help="Pool name (directory in pools/)")
    parser.add_argument("--all", "-a", action="store_true", help="Process all pools")
    parser.add_argument("--rebuild", "-r", action="store_true", help="Full rebuild (delete index first)")
    parser.add_argument("--graph", action="store_true", help="Update knowledge graph (incremental)")
    parser.add_argument("--graph-rebuild", action="store_true", help="Rebuild knowledge graph from scratch")
    parser.add_argument("--code-only", action="store_true", help="Only index code files")
    parser.add_argument("--rebuild-code", action="store_true", help="Rebuild code index from scratch")
    parser.add_argument("--no-code", action="store_true", help="Skip code indexing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-graph-chunks", type=int, default=None, metavar="N",
                        help="Cap graph build at N unprocessed chunks per pool (prevents long overnight runs)")
    parser.add_argument("--sync-sources", action="store_true", help="Sync sources before ingestion")
    parser.add_argument("--list", "-l", action="store_true", help="List available pools")
    parser.add_argument("--stats", action="store_true", help="Show index stats")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel pool workers for --all (default: 1); with a GPU, pools take turns on it")
    parser.add_argument(
        "--processes", type=int, default=0,
        help="Parallel pool workers using a process pool (CUDA isolation). "
             "When >1, takes precedence over --workers. Default: 0 (disabled). "
             "With a GPU, pools take turns on it."
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Don't wait for the GPU lease or for headroom (free VRAM and RAM, GPU "
             "temperature, load): skip the pool and say what holds the GPU, exiting "
             f"{EXIT_MACHINE_BUSY} if any pool was skipped. Without it, pools wait up to "
             "MULLET_GPU_WAIT_MAX_S (default 900 s), then index on the CPU",
    )
    args = parser.parse_args()

    if args.list:
        pools = discover_clients(clients_dir=TRAWL_POOLS_DIR)
        print(f"Available pools ({len(pools)}):")
        for p in pools:
            print(f"  - {p}")
        return

    if args.all:
        import time
        pool_names = discover_clients(clients_dir=TRAWL_POOLS_DIR)
        if not pool_names:
            print("No pools found.")
            return
        mode = "Rebuilding" if args.rebuild else "Syncing"
        graph_str = " + graph rebuild" if args.graph_rebuild else " + graph" if args.graph else ""
        print(f"{mode} {len(pool_names)} pools{graph_str}...\n", flush=True)
        total_start = time.time()
        completed_times: list[float] = []
        # Pools skipped because the machine was busy (only with --no-wait).
        busy: list[str] = []

        if args.processes > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(
                f"  (process-pool mode, {args.processes} processes — live phase output disabled)\n",
                flush=True,
            )
            for n in pool_names:
                print(f"  {n:<22} queued...", flush=True)

            with ProcessPoolExecutor(max_workers=args.processes) as ex:
                future_to_name = {
                    ex.submit(_reindex_one, n, args, True): n for n in pool_names
                }
                done_count = 0
                for fut in as_completed(future_to_name):
                    n = future_to_name[fut]
                    try:
                        elapsed = fut.result()
                        if not isinstance(elapsed, (int, float)):
                            elapsed = 0.0
                        elapsed = float(elapsed)
                        completed_times.append(elapsed)
                        print(f"  {n:<22} DONE  {elapsed:.1f}s", flush=True)
                    except machine.MachineBusy as e:
                        completed_times.append(0.0)
                        busy.append(n)
                        print(f"  {n:<22} FAILED: skipped: {e}", flush=True)
                    except Exception as e:
                        completed_times.append(0.0)
                        print(f"  {n:<22} FAILED: {e}", flush=True)
                    done_count += 1
                    remaining = len(pool_names) - done_count
                    if remaining > 0 and completed_times:
                        avg = sum(completed_times) / len(completed_times)
                        eta = avg * remaining / max(1, args.processes)
                        total_so_far = time.time() - total_start
                        print(
                            f"  [{done_count}/{len(pool_names)} complete  "
                            f"elapsed {_fmt_eta(total_so_far)}  "
                            f"avg {avg:.1f}s/pool  "
                            f"eta {_fmt_eta(eta)}]",
                            flush=True,
                        )
            # Child processes own their own CUDA contexts; nothing to clear here.
            print(
                f"\nDone — {len(pool_names)} pools in "
                f"{_fmt_eta(time.time() - total_start)} ({args.processes} processes)"
            )
            if busy:
                sys.exit(EXIT_MACHINE_BUSY)
            return

        if args.workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            print_lock = _print_lock  # shared with the pools' wait lines
            with print_lock:
                print(
                    f"  (parallel mode, {args.workers} workers — live phase output disabled)\n",
                    flush=True,
                )

            def _run_one(n: str) -> float:
                try:
                    with print_lock:
                        print(f"  {n:<22} starting...", flush=True)
                    elapsed = _reindex_one(n, args, quiet=True)
                    if not isinstance(elapsed, (int, float)):
                        elapsed = 0.0
                    with print_lock:
                        print(f"  {n:<22} DONE  {float(elapsed):.1f}s", flush=True)
                    return float(elapsed)
                except machine.MachineBusy as e:
                    with print_lock:
                        busy.append(n)
                        print(f"  {n:<22} FAILED: skipped: {e}", flush=True)
                    return 0.0
                except Exception as e:
                    with print_lock:
                        print(f"  {n:<22} FAILED: {e}", flush=True)
                    return 0.0

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(_run_one, n) for n in pool_names]
                done_count = 0
                for fut in as_completed(futures):
                    elapsed = fut.result()
                    completed_times.append(elapsed)
                    done_count += 1
                    remaining = len(pool_names) - done_count
                    if remaining > 0 and completed_times:
                        avg = sum(completed_times) / len(completed_times)
                        eta = avg * remaining / max(1, args.workers)
                        total_so_far = time.time() - total_start
                        with print_lock:
                            print(
                                f"  [{done_count}/{len(pool_names)} complete  "
                                f"elapsed {_fmt_eta(total_so_far)}  "
                                f"avg {avg:.1f}s/pool  "
                                f"eta {_fmt_eta(eta)}]",
                                flush=True,
                            )
            # Release models now all parallel workers are done.
            try:
                from mullet.knowledge.engine.model_cache import get_model_cache
                import gc
                get_model_cache().clear()
                gc.collect()
            except Exception:
                pass
            print(
                f"\nDone — {len(pool_names)} pools in "
                f"{_fmt_eta(time.time() - total_start)} ({args.workers} workers)"
            )
            if busy:
                sys.exit(EXIT_MACHINE_BUSY)
            return

        for i, name in enumerate(pool_names):
            pool_elapsed = 0.0
            try:
                result = _reindex_one(name, args)
                if isinstance(result, (int, float)):
                    pool_elapsed = float(result)
                else:
                    pool_elapsed = 0.0
            except machine.MachineBusy as e:
                busy.append(name)
                print(f"  {name:<22} FAILED: skipped: {e}", flush=True)
            except Exception as e:
                print(f"  {name:<22} FAILED: {e}", flush=True)

            # Release embedding model + VRAM between pools so the next pool
            # starts with a clean GPU state (avoids VRAM fragmentation across
            # pools with different embedding models, and frees memory for
            # any graph extraction that runs as part of reindex).
            try:
                from mullet.knowledge.engine.model_cache import get_model_cache
                import gc
                get_model_cache().clear()
                gc.collect()
            except Exception:
                pass

            completed_times.append(pool_elapsed)
            # Rolling ETA after each pool
            done_count = i + 1
            remaining = len(pool_names) - done_count
            if remaining > 0 and completed_times:
                avg = sum(completed_times) / len(completed_times)
                eta = avg * remaining
                total_so_far = time.time() - total_start
                print(
                    f"  [{done_count}/{len(pool_names)} complete  "
                    f"elapsed {_fmt_eta(total_so_far)}  "
                    f"avg {avg:.1f}s/pool  "
                    f"eta {_fmt_eta(eta)}]",
                    flush=True,
                )
            sys.stdout.flush()
        print(f"\nDone — {len(pool_names)} pools in {_fmt_eta(time.time() - total_start)}")
        if busy:
            sys.exit(EXIT_MACHINE_BUSY)
        return

    if not args.pool:
        parser.error("--pool is required (or use --all or --list)")

    if args.stats:
        client = load_client(args.pool, clients_dir=TRAWL_POOLS_DIR)
        import chromadb
        cc = chromadb.PersistentClient(path=str(client.index_path))
        try:
            col = cc.get_collection(name=client.machine_name)
            print(f"Pool: {client.name}")
            print(f"  Chunks: {col.count()}")
        except Exception:
            print(f"No index for {args.pool}")
        return

    try:
        _reindex_one(args.pool, args, sync_sources=args.sync_sources)
    except machine.MachineBusy as exc:
        PhaseReporter(args.pool).fail(f"skipped: {exc}")
        sys.exit(EXIT_MACHINE_BUSY)


if __name__ == "__main__":
    main()
