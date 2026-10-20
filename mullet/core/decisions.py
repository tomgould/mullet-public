"""Knowledge-graph seed: extract decision/rationale/rejection sentences from prose artifacts.

Scans docs/*.md, .mullet/progress.txt, and git log, writing matches to
.mullet/decisions.jsonl.  No LLM calls — pure regex/keyword extraction.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, re.Pattern[str]] = {
    "decision": re.compile(
        r"\b(decided(?:\s+to)?|chose|selected|picked|will\s+use|we\s+use|going\s+with)\b.{5,180}",
        re.IGNORECASE,
    ),
    "rejection": re.compile(
        r"\b(rejected|ruled\s+out|dropped|skipped|not\s+going\s+with|decided\s+not\s+to)\b.{5,180}",
        re.IGNORECASE,
    ),
    "rationale": re.compile(
        r"\b(because|reason(?:ing)?|rationale|so\s+that|in\s+order\s+to|to\s+avoid|to\s+preserve)\b.{5,180}",
        re.IGNORECASE,
    ),
}

# Priority: decision > rationale > rejection (when multiple match)
_KIND_PRIORITY = ["decision", "rationale", "rejection"]

_MARKDOWN_ONLY = re.compile(r"^[\#\-\*\|\s]+$")

# ---------------------------------------------------------------------------
# Post-match false-positive filter
# ---------------------------------------------------------------------------

# Minimum alpha-word count before a candidate is accepted
MIN_WORD_COUNT = 5

# Pytest / test-runner output (e.g. "3584 passed, 2 skipped, 0 failed")
_PYTEST_OUTPUT = re.compile(r"\b\d+\s+(passed|skipped|failed|deselected)\b", re.IGNORECASE)

# Lines that are purely ASCII table/separator content (no real words)
_TABLE_SEPARATOR = re.compile(r"^\s*[|+\-]")

# Inline code span — backtick-flanked text
_INLINE_CODE = re.compile(r"`[^`]+`")


def _is_false_positive(line: str, matched_keyword: str) -> bool:
    """Return True if *line* should be rejected as a false positive.

    Checks (in order):
    1. Pytest/CI output containing digit+status word.
    2. Table/separator-only lines (no real prose).
    3. Trigger keyword sits inside an inline code span.
    4. >60 % of characters are non-alphanumeric (data/table rows).
    5. Fewer than MIN_WORD_COUNT alpha-only words.
    """
    if _PYTEST_OUTPUT.search(line):
        return True

    if _TABLE_SEPARATOR.match(line) and not re.search(r"[A-Za-z]{4,}", line):
        return True

    # Check if the keyword appears exclusively inside inline-code spans
    keyword_lower = matched_keyword.lower()
    line_without_code = _INLINE_CODE.sub("", line)
    if keyword_lower not in line_without_code.lower():
        return True

    # >60% non-alphanumeric
    non_alnum = sum(1 for c in line if not c.isalnum() and not c.isspace())
    total = len(line)
    if total > 0 and (non_alnum / total) > 0.60:
        return True

    # Too few alpha-only words (skip check on long lines — they clearly have prose content)
    if len(line) < 80:
        alpha_words = [w for w in re.split(r"\W+", line) if w.isalpha()]
        if len(alpha_words) < MIN_WORD_COUNT:
            return True

    return False


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """A single extracted decision, rationale, or rejection sentence."""

    id: str
    source_path: str
    source_type: str
    line_number: int | None
    timestamp: str | None
    excerpt: str
    kind: str
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert the decision to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Decision:
        """Create a Decision instance from a dictionary."""
        return cls(**d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(source_path: str, excerpt: str) -> str:
    return hashlib.sha1((source_path + "|" + excerpt).encode()).hexdigest()[:16]


def _classify_line(line: str) -> tuple[str, list[str]] | None:
    """Return (kind, keywords) if line matches, else None.

    Priority: decision > rationale > rejection.
    """
    trimmed = line.strip()
    if len(trimmed) < 15 or len(trimmed) > 400:
        return None
    if _MARKDOWN_ONLY.match(trimmed):
        return None

    matched_kinds: list[tuple[str, str]] = []  # (kind, keyword)
    for kind in _KIND_PRIORITY:
        pat = _PATTERNS[kind]
        m = pat.search(trimmed)
        if m:
            matched_kinds.append((kind, m.group(1)))

    if not matched_kinds:
        return None

    # Highest-priority kind wins; collect all matched keywords
    best_kind = matched_kinds[0][0]
    keywords = [kw for _, kw in matched_kinds]

    # Post-match false-positive filter (use the first/primary keyword)
    if _is_false_positive(trimmed, matched_kinds[0][1]):
        return None

    return best_kind, keywords


def _extract_from_lines(
    lines: list[str],
    source_path: str,
    source_type: str,
    timestamp: str | None = None,
) -> list[Decision]:
    """Extract decisions from a list of text lines, skipping code blocks."""
    results: list[Decision] = []
    in_code_block = False

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()

        # Track triple-backtick fences
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        result = _classify_line(stripped)
        if result is None:
            continue

        kind, keywords = result
        excerpt = stripped[:400]
        did = _make_id(source_path, excerpt)
        results.append(
            Decision(
                id=did,
                source_path=source_path,
                source_type=source_type,
                line_number=lineno,
                timestamp=timestamp,
                excerpt=excerpt,
                kind=kind,
                keywords=keywords,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def decisions_path(base: Optional[Path] = None) -> Path:
    """Return path to .mullet/decisions.jsonl under *base* (or cwd)."""
    root = base or Path.cwd()
    return root / ".mullet" / "decisions.jsonl"


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------

def scan_docs(docs_dir: Optional[Path] = None) -> list[Decision]:
    """Scan all .md files under *docs_dir* (defaults to <cwd>/docs)."""
    root = Path.cwd()
    if docs_dir is not None:
        target = docs_dir
        # Use docs_dir's parent as the relative-path anchor when an explicit
        # directory is given (e.g. in tests using tmp_path).
        root = docs_dir.parent
    else:
        target = root / "docs"
    if not target.is_dir():
        return []

    results: list[Decision] = []
    for md_file in sorted(target.rglob("*.md")):
        rel = str(md_file.relative_to(root))
        name = md_file.name.lower()
        if "roadmap" in name:
            src_type = "roadmap"
        else:
            src_type = "docs"

        try:
            lines = md_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        results.extend(_extract_from_lines(lines, rel, src_type))

    return results


def scan_progress(progress_file: Optional[Path] = None) -> list[Decision]:
    """Scan .mullet/progress.txt."""
    root = Path.cwd()
    target = progress_file or (root / ".mullet" / "progress.txt")
    if not target.is_file():
        return []

    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    rel = str(target.relative_to(root)) if target.is_relative_to(root) else str(target)
    return _extract_from_lines(lines, rel, "progress")


def scan_git_log(limit: int = 200, repo_dir: Optional[Path] = None) -> list[Decision]:
    """Scan recent git commit messages for decision sentences.

    Returns empty list if the directory is not a git repo or git is unavailable.
    """
    cwd = repo_dir or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "log", f"-n{limit}", "--format=%H|%cI|%s%n%b"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    decisions: list[Decision] = []
    current_hash: str = ""
    current_date: str | None = None
    body_lines: list[str] = []

    def _flush() -> None:
        if current_hash and body_lines:
            extracted = _extract_from_lines(
                body_lines,
                source_path=f"git:{current_hash[:12]}",
                source_type="commit",
                timestamp=current_date,
            )
            decisions.extend(extracted)

    for raw_line in result.stdout.splitlines():
        # Header line: HASH|ISO_DATE|subject
        if "|" in raw_line and len(raw_line) > 50 and raw_line[:40].replace("-", "").isalnum():
            # Looks like a new commit header
            parts = raw_line.split("|", 2)
            if len(parts) == 3:
                _flush()
                current_hash = parts[0]
                current_date = parts[1] or None
                body_lines = [parts[2]] if parts[2].strip() else []
                continue
        body_lines.append(raw_line)

    _flush()
    return decisions


def scan_all(base: Optional[Path] = None, git_limit: int = 200) -> list[Decision]:
    """Run all scanners and return combined list (deduped by id)."""
    root = base or Path.cwd()
    all_decisions: list[Decision] = []
    all_decisions.extend(scan_docs(root / "docs"))
    all_decisions.extend(scan_progress(root / ".mullet" / "progress.txt"))
    all_decisions.extend(scan_git_log(limit=git_limit, repo_dir=root))

    # Dedup by id (keep first seen)
    seen: set[str] = set()
    unique: list[Decision] = []
    for d in all_decisions:
        if d.id not in seen:
            seen.add(d.id)
            unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist(decisions: list[Decision], base: Optional[Path] = None) -> int:
    """Write decisions to .mullet/decisions.jsonl, deduping by id.

    Overwrites existing file atomically.  Returns count of records written.
    """
    path = decisions_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing, dedup against new
    existing: dict[str, Decision] = {}
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    d = Decision.from_dict(json.loads(line))
                    existing[d.id] = d
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    # New decisions take priority over existing
    merged: dict[str, Decision] = {**existing}
    for d in decisions:
        merged[d.id] = d

    # Write atomically via tmp file
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            for d in merged.values():
                f.write(json.dumps(d.to_dict()) + "\n")
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    return len(merged)


def load(base: Optional[Path] = None) -> list[Decision]:
    """Load all decisions from .mullet/decisions.jsonl."""
    path = decisions_path(base)
    if not path.exists():
        return []

    results: list[Decision] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                results.append(Decision.from_dict(json.loads(line)))
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return results


def search(
    query: str,
    limit: int = 20,
    kind: Optional[str] = None,
    base: Optional[Path] = None,
) -> list[Decision]:
    """Substring search on excerpt, newest first (None timestamps go last).

    Parameters
    ----------
    query:  case-insensitive substring to match against excerpt.
    limit:  max results to return.
    kind:   if set, filter to only this kind ("decision", "rejection", "rationale").
    base:   project root; defaults to cwd.
    """
    all_d = load(base)
    q = query.lower()

    def _matches(d: Decision) -> bool:
        if kind and d.kind != kind:
            return False
        return q in d.excerpt.lower()

    filtered = [d for d in all_d if _matches(d)]

    # Sort: valid timestamps descending, None last
    def _sort_key(d: Decision) -> tuple[int, str]:
        if d.timestamp:
            return (0, d.timestamp)
        return (1, "")

    filtered.sort(key=_sort_key, reverse=False)
    # reverse descending for timestamped, None-last
    timestamped = [d for d in filtered if d.timestamp]
    no_ts = [d for d in filtered if not d.timestamp]
    timestamped.sort(key=lambda d: d.timestamp or "", reverse=True)
    return (timestamped + no_ts)[:limit]
