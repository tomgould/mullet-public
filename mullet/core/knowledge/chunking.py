"""Semantic chunking — sentence-boundary and code-aware text splitting.

Provides chunking strategies that are smarter than simple paragraph splitting:
- SentenceChunker: splits on sentence boundaries
- CodeChunker: splits on function/class boundaries for source code
- RecursiveChunker: tries multiple separators, falling back to smaller units

All chunkers produce enriched ChunkMetadata alongside the text.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    source: str
    chunk_index: int
    char_start: int = 0
    char_end: int = 0
    heading_path: str = ""  # e.g. "## Setup > ### Installation"
    language: str = ""


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    metadata: ChunkMetadata


class Chunker(Protocol):
    """Interface for text chunking strategies."""
    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split text into chunks."""
        ...


# ── Sentence-boundary chunker ────────────────────────────────────────────────

# Sentence-ending patterns
_SENTENCE_END = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'  # Period/exclamation/question + space + capital
    r'|(?<=\n)\n+'              # Double newline (paragraph break)
)

# Heading detection for heading_path tracking
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


class SentenceChunker:
    """Split text on sentence boundaries.

    Accumulates sentences until max_chars, then creates a chunk.
    Respects paragraph boundaries as natural split points.
    """

    def __init__(self, max_chars: int = 1500, overlap_chars: int = 100, min_chars: int = 50):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split text into sentence-bounded chunks."""
        if not text or len(text.strip()) < self.min_chars:
            return []

        sentences = _SENTENCE_END.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:  # pragma: no cover — line 68 guard makes this unreachable
            return []

        chunks: list[Chunk] = []
        current = ""
        char_pos = 0

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self.max_chars:
                current = f"{current} {sentence}".strip() if current else sentence
            else:
                if current and len(current) >= self.min_chars:
                    heading = self._extract_heading_path(current)
                    chunks.append(Chunk(
                        text=current,
                        metadata=ChunkMetadata(
                            source=source,
                            chunk_index=len(chunks),
                            char_start=char_pos,
                            char_end=char_pos + len(current),
                            heading_path=heading,
                        ),
                    ))
                    char_pos += len(current) - self.overlap_chars

                # Start new chunk with overlap
                if self.overlap_chars > 0 and current:
                    overlap = current[-self.overlap_chars:]
                    current = f"{overlap} {sentence}".strip()
                else:
                    current = sentence

        # Last chunk
        if current and len(current) >= self.min_chars:
            chunks.append(Chunk(
                text=current,
                metadata=ChunkMetadata(
                    source=source,
                    chunk_index=len(chunks),
                    char_start=char_pos,
                    char_end=char_pos + len(current),
                    heading_path=self._extract_heading_path(current),
                ),
            ))

        return chunks

    def _extract_heading_path(self, text: str) -> str:
        """Extract markdown heading hierarchy from chunk text."""
        headings = _HEADING_RE.findall(text)
        if not headings:
            return ""
        return " > ".join(h[1].strip() for h in headings[:3])


# ── Code-aware chunker ───────────────────────────────────────────────────────

# Code block boundaries
_CODE_BOUNDARY_PY = re.compile(r'^(?:def |class |async def )', re.MULTILINE)
_CODE_BOUNDARY_TS = re.compile(r'^(?:export |function |class |interface |const \w+ = )', re.MULTILINE)
_CODE_BOUNDARY_GO = re.compile(r'^(?:func |type )', re.MULTILINE)


class CodeChunker:
    """Split source code on function/class boundaries.

    Tries to keep each function/class as a single chunk.
    Falls back to line-based splitting for very long functions.
    """

    def __init__(self, max_chars: int = 3000, min_chars: int = 50):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def chunk(self, text: str, source: str = "", language: str = "") -> list[Chunk]:
        """Split source code on function or class boundaries."""
        if not text or len(text.strip()) < self.min_chars:
            return []

        if not language:
            language = self._detect_language(source)

        boundary = self._get_boundary(language)
        if boundary is None:
            # Fall back to line-based splitting
            return self._chunk_by_lines(text, source, language)

        # Find all boundaries
        matches = list(boundary.finditer(text))
        if not matches:
            return self._chunk_by_lines(text, source, language)

        chunks: list[Chunk] = []
        positions = [m.start() for m in matches]
        positions.append(len(text))

        for i in range(len(positions) - 1):
            start = positions[i]
            end = positions[i + 1]
            segment = text[start:end].rstrip()

            if len(segment) < self.min_chars:
                continue

            if len(segment) > self.max_chars:
                # Split very long functions by lines
                sub_chunks = self._chunk_by_lines(segment, source, language, offset=start)
                for sc in sub_chunks:
                    sc.metadata.chunk_index = len(chunks)
                    chunks.append(sc)
            else:
                chunks.append(Chunk(
                    text=segment,
                    metadata=ChunkMetadata(
                        source=source,
                        chunk_index=len(chunks),
                        char_start=start,
                        char_end=end,
                        language=language,
                    ),
                ))

        return chunks

    def _chunk_by_lines(self, text: str, source: str, language: str, offset: int = 0) -> list[Chunk]:
        """Fall back to line-based chunking."""
        lines = text.split('\n')
        chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_len = 0

        for line in lines:
            if current_len + len(line) + 1 > self.max_chars and current_lines:
                chunk_text = '\n'.join(current_lines)
                if len(chunk_text) >= self.min_chars:
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            source=source,
                            chunk_index=len(chunks),
                            char_start=offset,
                            char_end=offset + len(chunk_text),
                            language=language,
                        ),
                    ))
                    offset += len(chunk_text)
                current_lines = []
                current_len = 0

            current_lines.append(line)
            current_len += len(line) + 1

        if current_lines:
            chunk_text = '\n'.join(current_lines)
            if len(chunk_text) >= self.min_chars:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        source=source,
                        chunk_index=len(chunks),
                        char_start=offset,
                        char_end=offset + len(chunk_text),
                        language=language,
                    ),
                ))

        return chunks

    def _get_boundary(self, language: str):
        """Get the code boundary regex for a language."""
        return {
            'python': _CODE_BOUNDARY_PY,
            'typescript': _CODE_BOUNDARY_TS,
            'javascript': _CODE_BOUNDARY_TS,
            'go': _CODE_BOUNDARY_GO,
        }.get(language)

    def _detect_language(self, source: str) -> str:
        """Detect language from file extension."""
        ext = Path(source).suffix.lower() if source else ""
        return {
            '.py': 'python', '.ts': 'typescript', '.tsx': 'typescript',
            '.js': 'javascript', '.jsx': 'javascript', '.go': 'go',
        }.get(ext, "")


# ── Recursive chunker ────────────────────────────────────────────────────────

class RecursiveChunker:
    """Try multiple separators, falling back to smaller units.

    Tries: paragraph → sentence → line → character splits.
    """

    def __init__(self, max_chars: int = 1500, min_chars: int = 50):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self._separators = [
            '\n\n',  # Paragraph
            '. ',    # Sentence
            '\n',    # Line
            ' ',     # Word
        ]

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split text recursively using progressively finer separators."""
        if not text or len(text.strip()) < self.min_chars:
            return []

        raw_chunks = self._split_recursive(text.strip(), 0)
        return [
            Chunk(
                text=c,
                metadata=ChunkMetadata(source=source, chunk_index=i),
            )
            for i, c in enumerate(raw_chunks)
            if len(c) >= self.min_chars
        ]

    def _split_recursive(self, text: str, sep_idx: int) -> list[str]:
        """Recursively split text using progressively finer separators."""
        if len(text) <= self.max_chars:
            return [text]

        if sep_idx >= len(self._separators):
            # Last resort: hard split
            return [text[i:i + self.max_chars] for i in range(0, len(text), self.max_chars)]

        sep = self._separators[sep_idx]
        parts = text.split(sep)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.max_chars:
                    chunks.extend(self._split_recursive(part, sep_idx + 1))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks


def enrich_chunks(chunks: list[Chunk], source_path: str) -> list[Chunk]:
    """Enrich chunks with metadata from the source file path."""
    ext = Path(source_path).suffix.lower() if source_path else ""
    lang = {
        '.py': 'python', '.ts': 'typescript', '.tsx': 'typescript',
        '.js': 'javascript', '.go': 'go', '.php': 'php',
    }.get(ext, "")

    for chunk in chunks:
        chunk.metadata.source = source_path
        if not chunk.metadata.language:
            chunk.metadata.language = lang

    return chunks
