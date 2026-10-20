"""Document ingestion pipeline for the knowledge engine.

Handles file discovery, content extraction, chunking, and indexing.
Uses the vendored knowledge engine components for search/embedding.

Simplified from Trove core/ingest.py + core/pipeline.py.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# File extensions we can ingest
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift", ".kt",
    ".md", ".txt", ".rst", ".html", ".xml", ".json", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".conf", ".env", ".sh", ".bash", ".zsh",
    ".sql", ".graphql", ".proto", ".dockerfile", ".makefile",
    ".css", ".scss", ".less",
}

# Directories to skip
SKIP_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "target", "vendor", ".cargo", ".mullet",
}

# Max file size (500KB)
MAX_FILE_SIZE = 500_000

# Minimum chunk quality
MIN_CHUNK_CHARS = 50
MIN_ALPHA_RATIO = 0.3


@dataclass
class FileManifest:
    """Tracks which files have been ingested with their content hashes."""
    entries: dict[str, str] = field(default_factory=dict)  # path -> sha256

    def is_changed(self, path: str, content: str) -> bool:
        """Check if a file has changed since last ingestion."""
        new_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.entries.get(path) != new_hash

    def record(self, path: str, content: str):
        """Record a file's content hash."""
        self.entries[path] = hashlib.sha256(content.encode()).hexdigest()

    def save(self, manifest_path: Path):
        """Save manifest to disk."""
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(self.entries, indent=2))

    @classmethod
    def load(cls, manifest_path: Path) -> "FileManifest":
        """Load manifest from disk."""
        if manifest_path.exists():
            try:
                entries = json.loads(manifest_path.read_text())
                return cls(entries=entries)
            except (json.JSONDecodeError, OSError):
                pass
        return cls()


@dataclass
class IngestResult:
    """Result of an ingestion run."""
    files_scanned: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


def discover_files(
    root: str | Path,
    extensions: set[str] | None = None,
    skip_dirs: set[str] | None = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> Iterator[Path]:
    """Discover ingestible files under a root directory.

    Yields absolute Path objects for text files matching extensions.
    """
    root = Path(root).resolve()
    extensions = extensions or TEXT_EXTENSIONS
    skip_dirs = skip_dirs or SKIP_DIRS

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        for filename in filenames:
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()

            # Check extension
            if ext not in extensions and filepath.name.lower() not in {"makefile", "dockerfile"}:
                continue

            # Check size
            try:
                if filepath.stat().st_size > max_file_size:
                    continue
            except OSError as _exc:
                logger.warning("ingest: failed to stat file %s: %s", filepath, _exc)
                continue

            yield filepath


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 100) -> list[str]:
    """Simple chunking by paragraph boundaries with overlap.

    For production use, the vendored knowledge engine has more
    sophisticated chunking (semantic, recursive). This is a
    lightweight default for the ingestion CLI.
    """
    if not text or len(text) < MIN_CHUNK_CHARS:
        return []

    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    if not paragraphs:
        return []

    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                # Hard split for very long paragraphs
                for i in range(0, len(para), max_chars - overlap):
                    chunk = para[i:i + max_chars]
                    if validate_chunk(chunk):
                        chunks.append(chunk)
                current = ""
            else:
                current = para

    if current and validate_chunk(current):
        chunks.append(current)

    return chunks


def validate_chunk(text: str) -> bool:
    """Check if a chunk meets minimum quality standards."""
    if len(text) < MIN_CHUNK_CHARS:
        return False

    # Check alphanumeric ratio (reject binary garbage)
    alpha_count = sum(1 for c in text if c.isalnum())
    if alpha_count / max(len(text), 1) < MIN_ALPHA_RATIO:
        return False

    return True


def ingest_directory(
    root: str | Path,
    client_name: str = "",
    max_chars: int = 1500,
    incremental: bool = True,
) -> IngestResult:
    """Ingest all text files from a directory.

    Args:
        root: Directory to scan
        client_name: Name for the knowledge client
        max_chars: Maximum chars per chunk
        incremental: If True, skip unchanged files

    Returns:
        IngestResult with statistics
    """
    root = Path(root).resolve()
    start = time.time()
    result = IngestResult()

    # Load or create manifest for incremental ingestion
    manifest_path = root / ".mullet" / "ingest_manifest.json"
    manifest = FileManifest.load(manifest_path) if incremental else FileManifest()

    all_chunks: list[dict] = []

    for filepath in discover_files(root):
        result.files_scanned += 1

        try:
            content = filepath.read_text(errors="replace")
        except (OSError, UnicodeDecodeError) as e:
            result.errors.append(f"{filepath}: {e}")
            result.files_skipped += 1
            continue

        rel_path = str(filepath.relative_to(root))

        # Skip unchanged files
        if incremental and not manifest.is_changed(rel_path, content):
            result.files_unchanged += 1
            continue

        # Chunk the content
        chunks = chunk_text(content, max_chars=max_chars)
        if not chunks:
            result.files_skipped += 1
            continue

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "source": rel_path,
                "chunk_index": i,
                "chunk_id": f"{rel_path}:{i}",
            })

        manifest.record(rel_path, content)
        result.files_ingested += 1

    result.chunks_created = len(all_chunks)
    result.duration_s = time.time() - start

    # Save manifest
    if incremental:
        manifest.save(manifest_path)

    logger.info(
        "Ingestion complete: %d files scanned, %d ingested, %d chunks in %.1fs",
        result.files_scanned, result.files_ingested, result.chunks_created, result.duration_s,
    )

    return result
