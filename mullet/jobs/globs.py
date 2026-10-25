"""Repo-relative path patterns for node scopes and protected paths.

Patterns are POSIX globs matched against repo-relative paths:

- `**` spans any number of directories, and `dir/**` also matches `dir` itself;
- `*`, `?` and `[...]` stay within one path segment;
- a pattern without a `/` matches the name at any depth, as in .gitignore;
- a leading `/` anchors a pattern to the repo root, and a trailing `/`
  means the directory and everything under it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache


def _translate(pattern: str) -> str:
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == n:
            out.append("(?:/.*)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            end = pattern.find("]", i + 2)
            if end == -1:
                out.append(re.escape("["))
                i += 1
                continue
            body = pattern[i + 1:end]
            negate = body.startswith("!")
            if negate:
                body = body[1:]
            body = body.replace("\\", "\\\\").replace("^", "\\^")
            # A negated class must still never match a path separator.
            out.append(f"[^/{body}]" if negate else f"[{body}]")
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return "".join(out)


@lru_cache(maxsize=1024)
def compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile one pattern. Raises ValueError for an empty pattern."""
    p = pattern.strip()
    if not p:
        raise ValueError("empty path pattern")
    if p.endswith("/"):
        p = p.rstrip("/") + "/**"
    if p.startswith("/"):
        p = p.lstrip("/")
    elif "/" not in p:
        p = "**/" + p
    return re.compile(r"\A" + _translate(p) + r"\Z")


def matches(path: str, pattern: str) -> bool:
    """True if the repo-relative POSIX `path` matches `pattern`."""
    return compile_pattern(pattern).match(path.lstrip("/")) is not None


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    """True if `path` matches at least one of `patterns`."""
    return any(matches(path, p) for p in patterns)
