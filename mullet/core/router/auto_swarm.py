"""Auto-swarm routing heuristic.

Decides whether a ``mullet_run`` task should be fanned out to a swarm
instead of dispatched as a single provider call.

The heuristic is intentionally conservative: false-negatives (missing a
swarm opportunity) are cheaper than false-positives (unnecessary overhead).
"""

from __future__ import annotations

import os
from typing import Sequence


# --- tuneable thresholds -------------------------------------------------

_MIN_FILES_FOR_SWARM = 2          # ≥N files → consider swarm
_MIN_PROMPT_LEN_FOR_SWARM = 3_000  # chars
_CRITICAL_TIER = "CRITICAL"
_DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc", ".html"}
_MIN_DOC_FILES_FOR_SWARM = 2


def _is_doc(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in _DOC_EXTENSIONS


def _language_count(files: Sequence[str]) -> int:
    """Count distinct programming-language extensions in a file list."""
    exts = {os.path.splitext(p.lower())[1] for p in files}
    # Exclude doc/data extensions — only code languages
    code_exts = exts - _DOC_EXTENSIONS - {"", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
    return len(code_exts)


_FILE_PATH_RE = None  # lazy-compiled on first use


def _detect_file_mentions_in_task(task: str) -> int:
    """Count file-path-like tokens in a task description.

    Catches the case where a caller writes 'review files A.py, B.py, C.py'
    in the task text instead of passing files=[...]. When this count is
    high but the files param is empty, auto-swarm is still the right call
    — the task genuinely spans multiple files; we just don't have their
    content.
    """
    import re
    global _FILE_PATH_RE
    if _FILE_PATH_RE is None:
        # Matches paths like "foo.py", "src/foo.py", "a/b/c.tsx", etc.
        # Requires at least one slash OR a known code/doc extension to avoid
        # matching words with punctuation.
        _FILE_PATH_RE = re.compile(
            r"(?:[\w\-./]+/)?[\w\-]+\.(py|js|ts|tsx|jsx|go|rs|rb|java|c|cpp|h|hpp|"
            r"md|rst|txt|yaml|yml|toml|json|sh|php)\b"
        )
    return len(_FILE_PATH_RE.findall(task))


def _should_swarm(
    task: str,
    files: Sequence[str],
    prompt_len: int | None = None,
    tier: str | None = None,
) -> tuple[bool, str]:
    """Return ``(should_swarm, reason)`` for the given task parameters.

    Signals checked (any one is sufficient):
    1. Five or more files in scope  →  each file gets its own sub-task
    2. CRITICAL complexity tier  →  problem is complex enough to benefit from parallelism
    3. Prompt (task + estimated file sizes) exceeds 8 000 chars
    4. Multiple language extensions  →  cross-language work benefits from specialist sub-calls
    5. Three or more documentation files  →  doc-set review/generation scales horizontally
    6. Task mentions 5+ file-path-like tokens (caller forgot files=[...])

    Returns ``(False, "")`` when none of the signals fire.
    """
    file_count = len(files)

    # Effective prompt length: caller's value, OR derive from task length
    # plus approximate file sizes. File size is a proxy for the context
    # the delegate will actually see when files are packed into context.
    if prompt_len is None:
        prompt_len = len(task)
        # Add approximate file sizes so a "review 2 huge files" call still
        # trips the prompt_len signal even though file_count < 5.
        for f in files:
            try:
                prompt_len += os.path.getsize(f)
            except OSError:
                continue

    if tier and tier.upper() == _CRITICAL_TIER:
        return True, f"tier={tier}"

    if prompt_len >= _MIN_PROMPT_LEN_FOR_SWARM:
        return True, f"prompt_len={prompt_len} ≥ {_MIN_PROMPT_LEN_FOR_SWARM}"

    if file_count >= 2 and _language_count(files) >= 2:
        return True, f"multi-language ({_language_count(files)} extensions across {file_count} files)"

    doc_files = [f for f in files if _is_doc(f)]
    if len(doc_files) >= _MIN_DOC_FILES_FOR_SWARM:
        return True, f"doc_file_count={len(doc_files)} ≥ {_MIN_DOC_FILES_FOR_SWARM}"

    if file_count >= _MIN_FILES_FOR_SWARM:
        return True, f"file_count={file_count} ≥ {_MIN_FILES_FOR_SWARM}"

    # Caller-forgot-files detection: if task text mentions many file paths
    # but files=[] was passed, treat it as a swarm-worthy multi-file task.
    if file_count == 0:
        mentioned = _detect_file_mentions_in_task(task)
        if mentioned >= _MIN_FILES_FOR_SWARM:
            return True, f"task mentions {mentioned} file paths (caller did not pass files=[...])"

    return False, ""
