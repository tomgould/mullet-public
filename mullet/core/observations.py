"""Append-only observation capture for PostToolUse events.

Records every tool call (Read, Write, Edit, Bash, etc.) to a per-session JSONL
file under .mullet/observations/.  Designed to be fail-open — errors are
swallowed so a broken capture never blocks real work.
"""
import fcntl
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


_WHITESPACE_RE = re.compile(r"\s+")
_INPUT_LIMIT = 400
_OUTPUT_LIMIT = 1200
_LINE_HARD_LIMIT = 2000


@dataclass
class Observation:
    """A single captured tool-use event."""

    timestamp: str          # ISO 8601 UTC
    session_id: str
    tool_name: str
    tool_input_preview: str   # truncated to ~400 chars
    tool_output_preview: str  # truncated to ~1200 chars
    cwd: str


def observations_dir(base: Optional[Path] = None) -> Path:
    """Return (and create) the observations directory.

    Defaults to ``<base or cwd>/.mullet/observations/``.
    """
    root = Path(base) if base else Path.cwd()
    d = root / ".mullet" / "observations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compact(text: str) -> str:
    """Collapse runs of whitespace to a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _make_input_preview(tool_input: dict) -> str:
    try:
        raw = json.dumps(tool_input, ensure_ascii=False)
    except Exception:
        raw = str(tool_input)
    return _compact(raw[:_INPUT_LIMIT])


def _make_output_preview(tool_output: str) -> str:
    return _compact(str(tool_output)[:_OUTPUT_LIMIT])


def _fit_line(obs_dict: dict) -> str:
    """JSON-serialise the observation, trimming output_preview until it fits."""
    line = json.dumps(obs_dict, ensure_ascii=False)
    if len(line) <= _LINE_HARD_LIMIT:
        return line
    # Trim output_preview incrementally
    preview = obs_dict.get("tool_output_preview", "")
    while len(line) > _LINE_HARD_LIMIT and preview:
        drop = max(1, len(line) - _LINE_HARD_LIMIT)
        preview = preview[: max(0, len(preview) - drop)]
        obs_dict = {**obs_dict, "tool_output_preview": preview}
        line = json.dumps(obs_dict, ensure_ascii=False)
    return line


def record(
    tool_name: str,
    tool_input: dict,
    tool_output: str,
    session_id: str,
    cwd: Optional[str] = None,
    base: Optional[Path] = None,
) -> Observation:
    """Append one observation to ``<observations_dir>/<session_id>.jsonl``.

    Returns the captured :class:`Observation`.  Never raises — errors are
    swallowed so callers can always continue.
    """
    obs = Observation(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        session_id=session_id,
        tool_name=tool_name,
        tool_input_preview=_make_input_preview(tool_input),
        tool_output_preview=_make_output_preview(tool_output),
        cwd=cwd or str(Path.cwd()),
    )
    try:
        d = observations_dir(base)
        path = d / f"{session_id}.jsonl"
        line = _fit_line(asdict(obs))
        with open(path, "a", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX)
                fh.write(line + "\n")
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception:
        pass  # fail open
    return obs


def _parse_line(line: str) -> Optional[Observation]:
    try:
        d = json.loads(line)
        return Observation(**d)
    except Exception:
        return None


def iter_session(
    session_id: str, base: Optional[Path] = None
) -> Iterator[Observation]:
    """Yield all observations for *session_id*, oldest first."""
    path = observations_dir(base) / f"{session_id}.jsonl"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                obs = _parse_line(line)
                if obs:
                    yield obs


def iter_recent(
    limit: int = 50, base: Optional[Path] = None
) -> Iterator[Observation]:
    """Yield up to *limit* observations across all sessions, newest first."""
    d = observations_dir(base)
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    collected: list[Observation] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        obs = _parse_line(line)
                        if obs:
                            collected.append(obs)
        except Exception:
            continue
    # Sort newest first by timestamp string (ISO 8601 sorts lexicographically)
    collected.sort(key=lambda o: o.timestamp, reverse=True)
    yield from collected[:limit]


def search(
    query: str, limit: int = 20, base: Optional[Path] = None
) -> list[Observation]:
    """Return up to *limit* observations whose previews match *query* (case-insensitive), newest first."""
    q = query.lower()
    results: list[Observation] = []
    for obs in iter_recent(limit=10_000, base=base):
        if q in obs.tool_input_preview.lower() or q in obs.tool_output_preview.lower():
            results.append(obs)
            if len(results) >= limit:
                break
    return results
