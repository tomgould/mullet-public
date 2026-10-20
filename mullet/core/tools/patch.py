"""Patch parser — apply unified diffs to files.

Parses unified diff format and applies changes with fuzzy line matching.
Handles file creation, modification, and deletion.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Unified diff header patterns
_DIFF_HEADER = re.compile(r'^diff --git a/(.+) b/(.+)$')
_FILE_OLD = re.compile(r'^--- (?:a/)?(.+)$')
_FILE_NEW = re.compile(r'^\+\+\+ (?:b/)?(.+)$')
_HUNK_HEADER = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


@dataclass
class HunkResult:
    """Result of applying a single hunk."""
    applied: bool
    offset: int = 0  # Line offset used for fuzzy matching
    reason: str = ""


@dataclass
class PatchResult:
    """Result of applying a complete patch."""
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    hunks_applied: int = 0
    hunks_rejected: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when no hunks were rejected and no errors were collected."""
        return self.hunks_rejected == 0 and not self.errors


@dataclass
class Hunk:
    """A single hunk from a unified diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]  # Each line starts with ' ', '+', or '-'


def parse_unified_diff(patch_text: str) -> list[tuple[str, str, list[Hunk]]]:
    """Parse unified diff text into (old_file, new_file, hunks) tuples."""
    files: list[tuple[str, str, list[Hunk]]] = []
    lines = patch_text.splitlines()
    i = 0

    while i < len(lines):
        # Look for file headers
        old_file = ""
        new_file = ""

        # Skip to --- header
        while i < len(lines):
            m = _FILE_OLD.match(lines[i])
            if m:
                old_file = m.group(1)
                break
            i += 1

        if i >= len(lines):
            break
        i += 1

        # +++ header
        if i < len(lines):
            m = _FILE_NEW.match(lines[i])
            if m:
                new_file = m.group(1)
            i += 1

        if not new_file:
            continue

        # Parse hunks
        hunks: list[Hunk] = []
        while i < len(lines):
            m = _HUNK_HEADER.match(lines[i])
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) else 1
                i += 1

                hunk_lines: list[str] = []
                while i < len(lines) and not _HUNK_HEADER.match(lines[i]) and not _FILE_OLD.match(lines[i]):
                    if lines[i] and lines[i][0] in (' ', '+', '-'):
                        hunk_lines.append(lines[i])
                    elif lines[i] == '\\ No newline at end of file':
                        pass  # Skip
                    else:
                        break
                    i += 1

                hunks.append(Hunk(
                    old_start=old_start, old_count=old_count,
                    new_start=new_start, new_count=new_count,
                    lines=hunk_lines,
                ))
            else:
                break

        if hunks:
            files.append((old_file, new_file, hunks))

    return files


def apply_hunk(file_lines: list[str], hunk: Hunk, offset_tolerance: int = 3) -> HunkResult:
    """Apply a single hunk to file lines with fuzzy matching.

    Returns HunkResult indicating success/failure.
    Modifies file_lines in place.
    """
    target_line = hunk.old_start - 1  # Convert to 0-indexed

    # Try exact position first, then search with offset
    for offset in range(offset_tolerance + 1):
        for direction in (0, 1, -1) if offset == 0 else (1, -1):
            pos = target_line + (offset * direction)
            if pos < 0 or pos > len(file_lines):
                continue

            if _hunk_matches(file_lines, hunk, pos):
                _apply_hunk_at(file_lines, hunk, pos)
                return HunkResult(applied=True, offset=offset * direction)

    return HunkResult(applied=False, reason="Context mismatch — no matching location found")


def _hunk_matches(file_lines: list[str], hunk: Hunk, pos: int) -> bool:
    """Check if a hunk's context/remove lines match at the given position."""
    file_idx = pos
    for line in hunk.lines:
        if line.startswith(' ') or line.startswith('-'):
            expected = line[1:]
            if file_idx >= len(file_lines):
                return False
            if file_lines[file_idx].rstrip('\n') != expected.rstrip('\n'):
                return False
            file_idx += 1
    return True


def _apply_hunk_at(file_lines: list[str], hunk: Hunk, pos: int):
    """Apply a hunk at a specific position. Modifies file_lines in place."""
    new_lines: list[str] = []
    file_idx = pos

    for line in hunk.lines:
        if line.startswith(' '):
            # Context — keep
            new_lines.append(file_lines[file_idx])
            file_idx += 1
        elif line.startswith('-'):
            # Remove — skip
            file_idx += 1
        elif line.startswith('+'):
            # Add
            new_lines.append(line[1:] + '\n')

    # Replace the old lines with new lines
    old_end = file_idx
    file_lines[pos:old_end] = new_lines


class PatchParser:
    """Apply unified diffs to files."""

    def __init__(self, offset_tolerance: int = 3):
        self.offset_tolerance = offset_tolerance

    def apply_patch(self, patch_text: str, base_dir: str | Path) -> PatchResult:
        """Apply a unified diff patch to files under base_dir."""
        base = Path(base_dir)
        result = PatchResult()

        parsed = parse_unified_diff(patch_text)
        if not parsed:
            result.errors.append("No valid diffs found in patch text")
            return result

        for old_file, new_file, hunks in parsed:
            # New file creation (check before path traversal — old_file is /dev/null)
            if old_file == "/dev/null" or old_file == "dev/null":
                file_path = base / new_file
                # Security: block path traversal outside base_dir
                if not file_path.resolve().is_relative_to(base.resolve()):
                    result.errors.append(f"Path traversal blocked: {new_file}")
                    result.hunks_rejected += len(hunks)
                    continue
                content = "".join(
                    ln[1:] + '\n'
                    for h in hunks
                    for ln in h.lines
                    if ln.startswith('+')
                )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
                result.files_created.append(new_file)
                result.hunks_applied += len(hunks)
                continue

            # File deletion (/dev/null as new_file is legitimate in unified diffs)
            if new_file == "/dev/null" or new_file == "dev/null":
                old_path = base / old_file
                # Security: block path traversal on deletion target
                if not old_path.resolve().is_relative_to(base.resolve()):
                    result.errors.append(f"Path traversal blocked: {old_file}")
                    result.hunks_rejected += len(hunks)
                    continue
                if old_path.exists():
                    old_path.unlink()
                    result.files_deleted.append(old_file)
                    result.hunks_applied += len(hunks)
                continue

            file_path = base / new_file

            # Security: block path traversal outside base_dir
            if not file_path.resolve().is_relative_to(base.resolve()):
                result.errors.append(f"Path traversal blocked: {new_file}")
                result.hunks_rejected += len(hunks)
                continue

            # File modification
            if not file_path.exists():
                result.errors.append(f"File not found: {new_file}")
                result.hunks_rejected += len(hunks)
                continue

            file_lines = file_path.read_text().splitlines(keepends=True)

            for hunk in hunks:
                hunk_result = apply_hunk(file_lines, hunk, self.offset_tolerance)
                if hunk_result.applied:
                    result.hunks_applied += 1
                else:
                    result.hunks_rejected += 1
                    result.errors.append(f"{new_file}: {hunk_result.reason}")

            file_path.write_text("".join(file_lines))
            if new_file not in result.files_modified:
                result.files_modified.append(new_file)

        return result


PATCH_SCHEMA = {
    "name": "patch",
    "description": "Apply a unified diff patch to files. Supports file creation, modification, and deletion with fuzzy line matching.",
    "parameters": {
        "type": "object",
        "properties": {
            "patch_text": {
                "type": "string",
                "description": "The unified diff text to apply.",
            },
            "base_dir": {
                "type": "string",
                "description": "Base directory to apply the patch in. Defaults to current working directory.",
            },
            "offset_tolerance": {
                "type": "integer",
                "description": "Number of lines to search above/below for fuzzy matching. Default 3.",
            },
        },
        "required": ["patch_text"],
    },
}


def patch_handler(**kwargs) -> str:
    """Handle patch tool calls."""
    import json

    patch_text = kwargs.get("patch_text", "")
    base_dir = kwargs.get("base_dir", os.getcwd())
    offset_tolerance = kwargs.get("offset_tolerance", 3)

    if not patch_text.strip():
        return json.dumps({"error": "patch_text is required"})

    parser = PatchParser(offset_tolerance=offset_tolerance)
    result = parser.apply_patch(patch_text, base_dir)

    return json.dumps({
        "success": result.success,
        "files_modified": result.files_modified,
        "files_created": result.files_created,
        "files_deleted": result.files_deleted,
        "hunks_applied": result.hunks_applied,
        "hunks_rejected": result.hunks_rejected,
        "errors": result.errors,
    })


def register(registry):
    """Register patch tool into the tool registry."""
    registry.register(
        name="patch",
        func=patch_handler,
        schema=PATCH_SCHEMA,
        toolset="file",
    )
