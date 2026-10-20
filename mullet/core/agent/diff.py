"""Track and display file changes made during agent execution.

Captures file state before and after agent runs, then shows
a human-readable diff of what changed.
"""

import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """A single file change."""
    path: str
    change_type: str  # "created" | "modified" | "deleted"
    old_hash: str = ""
    new_hash: str = ""
    size_before: int = 0
    size_after: int = 0


class ChangeTracker:
    """Track file changes during agent execution.

    Usage:
        tracker = ChangeTracker(watch_dirs=["."])
        tracker.snapshot_before()
        # ... agent runs and modifies files ...
        changes = tracker.snapshot_after()
        print(tracker.format_changes())
    """

    def __init__(self, watch_dirs: list[str] | None = None, extensions: list[str] | None = None):
        self._watch_dirs = watch_dirs or ["."]
        self._extensions = extensions  # None = all files
        self._before: dict[str, tuple[str, int]] = {}  # path → (hash, size)
        self._after: dict[str, tuple[str, int]] = {}

    def snapshot_before(self):
        """Capture file state before agent execution."""
        self._before = self._scan()

    def snapshot_after(self) -> list[FileChange]:
        """Capture file state after execution and compute changes."""
        self._after = self._scan()
        return self.get_changes()

    def get_changes(self) -> list[FileChange]:
        """Compute the diff between before and after snapshots."""
        changes = []

        all_paths = set(self._before.keys()) | set(self._after.keys())

        for path in sorted(all_paths):
            before = self._before.get(path)
            after = self._after.get(path)

            if before and not after:
                changes.append(FileChange(
                    path=path, change_type="deleted",
                    old_hash=before[0], size_before=before[1],
                ))
            elif not before and after:
                changes.append(FileChange(
                    path=path, change_type="created",
                    new_hash=after[0], size_after=after[1],
                ))
            elif before and after and before[0] != after[0]:
                changes.append(FileChange(
                    path=path, change_type="modified",
                    old_hash=before[0], new_hash=after[0],
                    size_before=before[1], size_after=after[1],
                ))

        return changes

    def format_changes(self) -> str:
        """Format changes as human-readable output."""
        changes = self.get_changes()
        if not changes:
            return "No files changed."

        lines = [f"\n\033[1mFiles changed ({len(changes)}):\033[0m\n"]

        for c in changes:
            if c.change_type == "created":
                lines.append(f"  \033[32m+ {c.path}\033[0m ({_fmt_size(c.size_after)})")
            elif c.change_type == "deleted":
                lines.append(f"  \033[31m- {c.path}\033[0m")
            elif c.change_type == "modified":
                delta = c.size_after - c.size_before
                sign = "+" if delta >= 0 else ""
                lines.append(f"  \033[33m~ {c.path}\033[0m ({sign}{delta} bytes)")

        return "\n".join(lines)

    def _scan(self) -> dict[str, tuple[str, int]]:
        """Scan all watched directories and hash files."""
        result = {}
        for watch_dir in self._watch_dirs:
            base = Path(watch_dir)
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                if any(part.startswith(".") for part in p.parts):
                    continue  # Skip hidden dirs
                if "__pycache__" in str(p):
                    continue
                if self._extensions and p.suffix not in self._extensions:
                    continue
                try:
                    content = p.read_bytes()
                    h = hashlib.md5(content).hexdigest()
                    result[str(p)] = (h, len(content))
                except (PermissionError, OSError) as e:
                    logger.warning("diff: failed to scan file %s: %s", p, e)
        return result


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"
