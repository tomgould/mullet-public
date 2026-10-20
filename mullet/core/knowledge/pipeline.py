"""Producer-consumer ingestion pipeline.

Decouples file parsing (CPU, threaded) from embedding (batched) from
indexing (storage backend). Supports adaptive batch sizing via ResourceManager,
job tracking via JobManager, and graceful Ctrl+C interrupt.

Ported from Trove core/pipeline.py, simplified for Mullet's architecture.
"""

import logging
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from mullet.core.knowledge.ingest import (
    FileManifest,
    chunk_text,
    discover_files,
)
from mullet.core.knowledge.job_manager import JobManager

logger = logging.getLogger(__name__)


# ── Protocols for pluggable backends ─────────────────────────────────────────

class Embedder(Protocol):
    """Interface for an embedding backend."""
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings, returning vectors."""
        ...


class IndexStore(Protocol):
    """Interface for an indexing/storage backend."""
    def upsert(self, ids: list[str], embeddings: list[list[float]],
               documents: list[str], metadatas: list[dict]) -> None:
        """Store chunks with their embeddings."""
        ...


# ── Chunk representation ─────────────────────────────────────────────────────

@dataclass
class PipelineChunk:
    """A chunk flowing through the pipeline."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_id(source: str, index: int) -> str:
        """Deterministic chunk ID from source path and index."""
        import hashlib
        h = hashlib.sha256(f"{source}:{index}".encode()).hexdigest()[:16]
        return f"{source}:{index}:{h}"


# ── Pipeline stages ──────────────────────────────────────────────────────────

@dataclass
class StageStats:
    """Statistics for a single pipeline stage."""
    items_in: int = 0
    items_out: int = 0
    errors: int = 0
    duration_s: float = 0.0


@dataclass
class PipelineStats:
    """Overall pipeline run statistics."""
    files_discovered: int = 0
    files_parsed: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_indexed: int = 0
    flush_count: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    interrupted: bool = False

    @property
    def chunks_per_second(self) -> float:
        """Return the average chunk indexing rate."""
        return self.chunks_indexed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Files: {self.files_parsed}/{self.files_discovered} parsed",
            f"Chunks: {self.chunks_created} created -> {self.chunks_embedded} embedded -> {self.chunks_indexed} indexed",
            f"Batches: {self.flush_count}",
            f"Rate: {self.chunks_per_second:.0f} chunks/s",
            f"Duration: {self.elapsed_s:.1f}s",
        ]
        if self.files_unchanged:
            lines.append(f"Unchanged: {self.files_unchanged} (skipped)")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        if self.interrupted:
            lines.append("(interrupted by user)")
        return " | ".join(lines)


# ── Main pipeline ────────────────────────────────────────────────────────────

class IngestionPipeline:
    """Producer-consumer pipeline: parse (threads) -> buffer -> embed (GPU) -> index.

    Buffer accumulates chunks until embed_batch_size, then flushes.
    Ctrl+C loses at most one batch.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        index_store: IndexStore | None = None,
        *,
        parse_workers: int = 4,
        embed_batch_size: int = 32,
        max_chunk_chars: int = 1500,
        chunk_overlap: int = 100,
        incremental: bool = True,
        job_manager: JobManager | None = None,
    ):
        self.embedder = embedder
        self.index_store = index_store
        self.parse_workers = parse_workers
        self.embed_batch_size = embed_batch_size
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap
        self.incremental = incremental
        self.job_manager = job_manager

        # Buffer state
        self._buffer: list[PipelineChunk] = []
        self._stats = PipelineStats()
        self._start_time: float = 0.0

        # Interrupt handling
        self._interrupted = False
        self._original_sigint = None

    def run(
        self,
        root: str | Path,
        *,
        manifest_path: Path | None = None,
        progress_callback: Callable[[PipelineStats], None] | None = None,
    ) -> PipelineStats:
        """Run the full pipeline on a directory.

        Args:
            root: Directory to ingest.
            manifest_path: Path to store file hashes for incremental mode.
            progress_callback: Called after each flush with current stats.

        Returns:
            PipelineStats with final metrics.
        """
        root = Path(root).resolve()
        self._stats = PipelineStats()
        self._start_time = time.time()
        self._interrupted = False

        # Job tracking
        job_id = None
        if self.job_manager:
            job_id = str(uuid.uuid4())[:8]
            self.job_manager.start_job(job_id, f"Ingest {root.name}")

        # Manifest for incremental
        if manifest_path is None:
            manifest_path = root / ".mullet" / "pipeline_manifest.json"
        manifest = FileManifest.load(manifest_path) if self.incremental else FileManifest()

        # Discover files
        files = list(discover_files(root))
        self._stats.files_discovered = len(files)
        logger.info("Pipeline: discovered %d files in %s", len(files), root)

        self._install_signal_handler()

        try:
            self._parse_stage(files, root, manifest, progress_callback)
            self._flush_buffer()  # Flush remaining
        finally:
            self._restore_signal_handler()

        # Save manifest
        if self.incremental:
            manifest.save(manifest_path)

        self._stats.elapsed_s = time.time() - self._start_time

        # Complete job
        if self.job_manager and job_id:
            status = "completed" if not self._stats.interrupted else "cancelled"
            self.job_manager.complete_job(job_id, status=status, progress=100)

        logger.info("Pipeline complete: %s", self._stats.summary())
        return self._stats

    def run_chunks(
        self,
        chunks: list[PipelineChunk],
        *,
        progress_callback: Callable[[PipelineStats], None] | None = None,
    ) -> PipelineStats:
        """Run embed+index on pre-parsed chunks (e.g. from code extraction)."""
        self._stats = PipelineStats()
        self._start_time = time.time()
        self._stats.chunks_created = len(chunks)

        self._install_signal_handler()
        try:
            for chunk in chunks:
                self._buffer.append(chunk)
                if len(self._buffer) >= self.embed_batch_size:
                    self._flush_buffer()
                    if progress_callback:
                        progress_callback(self._stats)
                if self._interrupted:
                    break
            self._flush_buffer()
        finally:
            self._restore_signal_handler()

        self._stats.elapsed_s = time.time() - self._start_time
        return self._stats

    # ── Parse stage ──────────────────────────────────────────────────────────

    def _parse_stage(
        self,
        files: list[Path],
        root: Path,
        manifest: FileManifest,
        progress_callback: Callable | None,
    ):
        """Parse files in thread pool, buffering chunks."""
        with ThreadPoolExecutor(max_workers=self.parse_workers) as executor:
            future_to_path = {
                executor.submit(self._parse_one_file, fpath, root, manifest): fpath
                for fpath in files
            }

            for future in as_completed(future_to_path):
                if self._interrupted:
                    for f in future_to_path:
                        f.cancel()
                    self._stats.interrupted = True
                    break

                fpath = future_to_path[future]
                try:
                    chunks, status = future.result()
                    if status == "unchanged":
                        self._stats.files_unchanged += 1
                    elif status == "skipped":
                        self._stats.files_skipped += 1
                    elif status == "parsed":
                        self._stats.files_parsed += 1
                        self._stats.chunks_created += len(chunks)
                        for chunk in chunks:
                            self._buffer.append(chunk)
                            if len(self._buffer) >= self.embed_batch_size:
                                self._flush_buffer()
                                if progress_callback:
                                    progress_callback(self._stats)
                except Exception as e:
                    self._stats.errors.append(f"{fpath.name}: {e}")
                    logger.error("Parse error for %s: %s", fpath.name, e)

    def _parse_one_file(
        self, fpath: Path, root: Path, manifest: FileManifest,
    ) -> tuple[list[PipelineChunk], str]:
        """Parse a single file into chunks. Returns (chunks, status)."""
        try:
            content = fpath.read_text(errors="replace")
        except (OSError, UnicodeDecodeError) as e:
            logger.debug("Skip %s: %s", fpath.name, e)
            return [], "skipped"

        rel_path = str(fpath.relative_to(root))

        # Incremental: skip unchanged
        if self.incremental and not manifest.is_changed(rel_path, content):
            return [], "unchanged"

        # Chunk
        raw_chunks = chunk_text(content, max_chars=self.max_chunk_chars, overlap=self.chunk_overlap)
        if not raw_chunks:
            return [], "skipped"

        pipeline_chunks = [
            PipelineChunk(
                id=PipelineChunk.make_id(rel_path, i),
                content=text,
                metadata={
                    "source": rel_path,
                    "chunk_index": i,
                    "language": fpath.suffix.lstrip('.'),
                },
            )
            for i, text in enumerate(raw_chunks)
        ]

        # Record in manifest
        manifest.record(rel_path, content)

        return pipeline_chunks, "parsed"

    # ── Embed + Index stage ──────────────────────────────────────────────────

    def _flush_buffer(self):
        """Embed buffered chunks and push to index store."""
        if not self._buffer:
            return

        count = len(self._buffer)
        logger.debug("Flushing %d chunks (batch %d)", count, self._stats.flush_count + 1)

        # Dedup by ID
        seen: set[str] = set()
        deduped: list[PipelineChunk] = []
        for chunk in self._buffer:
            if chunk.id not in seen:
                seen.add(chunk.id)
                deduped.append(chunk)
        self._buffer = deduped
        count = len(self._buffer)

        texts = [c.content for c in self._buffer]

        # Embed
        if self.embedder:
            try:
                embeddings = self.embedder.embed_batch(texts)
                self._stats.chunks_embedded += count
            except Exception as e:
                logger.error("Embedding failed: %s", e)
                self._stats.errors.append(f"embed: {e}")
                self._buffer.clear()
                return
        else:
            # No embedder — pass empty vectors (dry-run or text-only indexing)
            embeddings = [[] for _ in texts]
            self._stats.chunks_embedded += count

        # Index
        if self.index_store:
            try:
                self.index_store.upsert(
                    ids=[c.id for c in self._buffer],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[c.metadata for c in self._buffer],
                )
                self._stats.chunks_indexed += count
            except Exception as e:
                logger.error("Indexing failed: %s", e)
                self._stats.errors.append(f"index: {e}")
        else:
            # No store — count as indexed (dry-run)
            self._stats.chunks_indexed += count

        self._stats.flush_count += 1
        self._buffer.clear()

        elapsed = time.time() - self._start_time
        rate = self._stats.chunks_indexed / elapsed if elapsed > 0 else 0
        logger.info(
            "Indexed %d chunks (total: %d, rate: %.0f/s)",
            count, self._stats.chunks_indexed, rate,
        )

    # ── Signal handling ──────────────────────────────────────────────────────

    def _install_signal_handler(self):
        """Install SIGINT handler for graceful shutdown."""
        if threading.current_thread() is not threading.main_thread():
            return
        self._original_sigint = signal.getsignal(signal.SIGINT)

        def _handler(signum, frame):
            if self._interrupted:
                logger.warning("Force exit requested")
                if self._original_sigint and callable(self._original_sigint):
                    self._original_sigint(signum, frame)
                return
            logger.info("Interrupt received — flushing buffer before exit...")
            self._interrupted = True

        signal.signal(signal.SIGINT, _handler)

    def _restore_signal_handler(self):
        """Restore original SIGINT handler."""
        if threading.current_thread() is not threading.main_thread():
            return
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None
