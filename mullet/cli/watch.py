"""File watcher mode — re-run tasks on file changes.

Watches the working directory for file changes and re-executes
a task when source files are modified. Useful for TDD workflows:
  mullet watch "run tests and fix failures"
"""

import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# File extensions to watch
_WATCH_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java",
    ".rb", ".php", ".c", ".cpp", ".h", ".yaml", ".yml", ".json",
    ".toml", ".md", ".html", ".css", ".sql", ".sh",
}

# Directories to ignore
_IGNORE_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".mullet", ".claude",
}

# Minimum interval between runs (seconds)
MIN_INTERVAL = 3


def _scan_files(directory: str = ".") -> dict[str, float]:
    """Scan directory and return {path: mtime} for watched files."""
    files = {}
    for root, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
        for f in filenames:
            if Path(f).suffix in _WATCH_EXTENSIONS:
                path = os.path.join(root, f)
                try:
                    files[path] = os.path.getmtime(path)
                except OSError as _exc:
                    logger.warning("watch: Failed to get file modification time for %s: %s", path, _exc)
                    pass
    return files


def _detect_changes(before: dict[str, float], after: dict[str, float]) -> list[str]:
    """Find files that changed between two scans."""
    changes = []

    # Modified or new files
    for path, mtime in after.items():
        if path not in before or before[path] != mtime:
            changes.append(path)

    # Deleted files
    for path in before:
        if path not in after:
            changes.append(f"{path} (deleted)")

    return changes


def run_watch(task: str, args, interval: float = 1.0):
    """Watch for file changes and re-run the task.

    Args:
        task: The task to run on each change.
        args: CLI arguments.
        interval: Poll interval in seconds.
    """
    from mullet.core.agent.project_detect import detect_project

    project = detect_project()
    print("\033[1mMullet Watch Mode\033[0m")
    print(f"  Task: {task[:60]}{'...' if len(task) > 60 else ''}")
    if project.language != "unknown":
        print(f"  Project: {project.language}" + (f" ({project.framework})" if project.framework else ""))
    print(f"  Watching: {', '.join(sorted(_WATCH_EXTENSIONS)[:8])}...")
    print("  Press Ctrl+C to stop.\n")

    last_run = 0
    baseline = _scan_files()
    run_count = 0

    try:
        while True:
            time.sleep(interval)
            current = _scan_files()
            changes = _detect_changes(baseline, current)

            if changes and (time.time() - last_run) >= MIN_INTERVAL:
                run_count += 1
                changed_names = [os.path.basename(c) for c in changes[:3]]
                print(f"\033[33m--- Change detected ({', '.join(changed_names)}) — run #{run_count} ---\033[0m")

                # Import here to avoid circular imports
                from mullet.cli.main import run_task
                try:
                    run_task(task, args)
                except SystemExit:
                    pass  # run_task calls sys.exit
                except Exception as e:
                    print(f"\033[31mError: {e}\033[0m", file=sys.stderr)

                print("\033[90m--- Watching for changes... ---\033[0m\n")
                last_run = time.time()
                baseline = _scan_files()  # Re-scan after run
            else:
                baseline = current

    except KeyboardInterrupt:
        print(f"\n\033[1mWatch stopped.\033[0m {run_count} runs completed.")
