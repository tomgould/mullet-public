"""Transcript parser for Claude Code JSONL session transcripts.

Pure Python port of the key parts of claude-hud's transcript.ts.
Returns a :class:`TranscriptSnapshot` from a JSONL file path, with
on-disk caching keyed on (mtime_ns, size).

The public API is :func:`parse_transcript`.  All errors are swallowed
so that the HUD never crashes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Represents a single tool invocation in a transcript."""
    id: str
    name: str
    status: str  # 'running' | 'completed' | 'error'
    start_time: datetime
    end_time: Optional[datetime] = None
    target: Optional[str] = None  # brief target: file_path, pattern, command[:30]
    # provider/model taken from the enclosing assistant message's `model` field
    # so every in-flight tool pill can show which LLM spawned it.
    provider: Optional[str] = None
    model: Optional[str] = None


@dataclass
class AgentCall:
    """Represents a subagent invocation in a transcript."""
    id: str
    agent_type: str  # 'general-purpose', 'Explore', etc. from subagent_type
    model: Optional[str]
    description: Optional[str]
    status: str  # 'running' | 'completed'
    start_time: datetime
    end_time: Optional[datetime] = None
    # provider inferred from `model` prefix (claude-*→anthropic, kimi-*→kimi, …)
    provider: Optional[str] = None


@dataclass
class TodoItem:
    """Represents a todo item extracted from a transcript."""
    content: str
    status: str  # 'pending' | 'in_progress' | 'completed'


@dataclass
class TranscriptSnapshot:
    """Snapshot of running and completed tools, agents, and todos."""
    running_tools: list[ToolCall] = field(default_factory=list)
    completed_tools: list[ToolCall] = field(default_factory=list)  # last 20
    agents: list[AgentCall] = field(default_factory=list)  # last 10
    todos: list[TodoItem] = field(default_factory=list)
    session_started_at: Optional[datetime] = None
    last_assistant_response_at: Optional[datetime] = None
    session_tokens_in: int = 0
    session_tokens_out: int = 0
    session_cache_creation_tokens: int = 0
    session_cache_read_tokens: int = 0
    last_compact_boundary_at: Optional[datetime] = None
    last_compact_post_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _default_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".claude", ".mullet-hud", "transcript-cache")


def _cache_path(real_path: str, cache_dir: str) -> str:
    digest = hashlib.sha256(real_path.encode()).hexdigest()
    return os.path.join(cache_dir, f"{digest}.json")


def _dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _tool_to_dict(t: ToolCall) -> dict:
    return {
        "id": t.id,
        "name": t.name,
        "status": t.status,
        "start_time": _dt_to_iso(t.start_time),
        "end_time": _dt_to_iso(t.end_time),
        "target": t.target,
        "provider": t.provider,
        "model": t.model,
    }


def _dict_to_tool(d: dict) -> ToolCall:
    return ToolCall(
        id=d["id"],
        name=d["name"],
        status=d["status"],
        start_time=_iso_to_dt(d.get("start_time")) or datetime.now(timezone.utc),
        end_time=_iso_to_dt(d.get("end_time")),
        target=d.get("target"),
        provider=d.get("provider"),
        model=d.get("model"),
    )


def _agent_to_dict(a: AgentCall) -> dict:
    return {
        "id": a.id,
        "agent_type": a.agent_type,
        "model": a.model,
        "description": a.description,
        "status": a.status,
        "start_time": _dt_to_iso(a.start_time),
        "end_time": _dt_to_iso(a.end_time),
        "provider": a.provider,
    }


def _dict_to_agent(d: dict) -> AgentCall:
    return AgentCall(
        id=d["id"],
        agent_type=d.get("agent_type", "agent"),
        model=d.get("model"),
        description=d.get("description"),
        status=d["status"],
        start_time=_iso_to_dt(d.get("start_time")) or datetime.now(timezone.utc),
        end_time=_iso_to_dt(d.get("end_time")),
        provider=d.get("provider"),
    )


def _snapshot_to_dict(snap: TranscriptSnapshot) -> dict:
    return {
        "running_tools": [_tool_to_dict(t) for t in snap.running_tools],
        "completed_tools": [_tool_to_dict(t) for t in snap.completed_tools],
        "agents": [_agent_to_dict(a) for a in snap.agents],
        "todos": [{"content": t.content, "status": t.status} for t in snap.todos],
        "session_started_at": _dt_to_iso(snap.session_started_at),
        "last_assistant_response_at": _dt_to_iso(snap.last_assistant_response_at),
        "session_tokens_in": snap.session_tokens_in,
        "session_tokens_out": snap.session_tokens_out,
        "session_cache_creation_tokens": snap.session_cache_creation_tokens,
        "session_cache_read_tokens": snap.session_cache_read_tokens,
        "last_compact_boundary_at": _dt_to_iso(snap.last_compact_boundary_at),
        "last_compact_post_tokens": snap.last_compact_post_tokens,
    }


def _dict_to_snapshot(d: dict) -> TranscriptSnapshot:
    return TranscriptSnapshot(
        running_tools=[_dict_to_tool(t) for t in d.get("running_tools", [])],
        completed_tools=[_dict_to_tool(t) for t in d.get("completed_tools", [])],
        agents=[_dict_to_agent(a) for a in d.get("agents", [])],
        todos=[TodoItem(content=t["content"], status=t["status"]) for t in d.get("todos", [])],
        session_started_at=_iso_to_dt(d.get("session_started_at")),
        last_assistant_response_at=_iso_to_dt(d.get("last_assistant_response_at")),
        session_tokens_in=d.get("session_tokens_in", 0),
        session_tokens_out=d.get("session_tokens_out", 0),
        session_cache_creation_tokens=d.get("session_cache_creation_tokens", 0),
        session_cache_read_tokens=d.get("session_cache_read_tokens", 0),
        last_compact_boundary_at=_iso_to_dt(d.get("last_compact_boundary_at")),
        last_compact_post_tokens=d.get("last_compact_post_tokens"),
    )


def _read_cache(real_path: str, mtime_ns: int, size: int, cache_dir: str) -> Optional[TranscriptSnapshot]:
    try:
        cp = _cache_path(real_path, cache_dir)
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        if (
            data.get("version") != _CACHE_VERSION
            or data.get("transcript_path") != real_path
            or data.get("mtime_ns") != mtime_ns
            or data.get("size") != size
        ):
            return None
        return _dict_to_snapshot(data["snapshot"])
    except Exception:
        return None


def _write_cache(real_path: str, mtime_ns: int, size: int, snap: TranscriptSnapshot, cache_dir: str) -> None:
    try:
        cp = _cache_path(real_path, cache_dir)
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        payload = {
            "version": _CACHE_VERSION,
            "transcript_path": real_path,
            "mtime_ns": mtime_ns,
            "size": size,
            "snapshot": _snapshot_to_dict(snap),
        }
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass  # cache failures are non-fatal


# ---------------------------------------------------------------------------
# Target extraction (port of extractTarget from transcript.ts lines 458-475)
# ---------------------------------------------------------------------------

def _extract_target(tool_name: str, inp: Optional[dict]) -> Optional[str]:
    if not inp:
        return None
    if tool_name in ("Read", "Write", "Edit"):
        return inp.get("file_path") or inp.get("path")
    if tool_name in ("Glob", "Grep"):
        return inp.get("pattern")
    if tool_name == "Bash":
        cmd = inp.get("command", "")
        if cmd:
            return cmd[:30] + ("..." if len(cmd) > 30 else "")
    return None


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------

def _infer_provider_from_model(model_id: Optional[str]) -> Optional[str]:
    """Map a model id to a provider tag for HUD badges.

    Returns None for missing input so callers can skip rendering the badge.
    Prefix order matters — moonshot-* must beat kimi-* only if we ever add a
    rule that conflicts, but today both map to 'kimi'. Local LLMs (mistral,
    qwen, phi, gemma, deepseek distills, llama) fall through to 'local'.
    """
    if not model_id:
        return None
    m = model_id.lower()
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("kimi-") or m.startswith("moonshot-"):
        return "kimi"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith(("gpt-", "o1-", "o3-", "openai-")):
        return "openai"
    # Local LLMs served via LM Studio / Ollama — no cloud provider
    if m.startswith((
        "mistral-", "qwen", "phi-", "gemma-", "llama-",
        "deepseek-", "codestral", "nemo",
    )):
        return "local"
    return None


# ---------------------------------------------------------------------------
# Line-level processing (port of processEntry from transcript.ts)
# ---------------------------------------------------------------------------

def _normalise_token_count(val) -> int:
    try:
        n = int(val)
        return max(0, n)
    except (TypeError, ValueError):
        return 0


def _process_entry(
    entry: dict,
    tool_map: dict,   # id -> ToolCall (insertion-order)
    agent_map: dict,  # id -> AgentCall
    latest_todos: list,
    result: TranscriptSnapshot,
) -> None:
    ts_str = entry.get("timestamp")
    ts: datetime
    try:
        ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
    except (ValueError, TypeError):
        ts = datetime.now(timezone.utc)

    valid_ts = ts_str is not None

    if result.session_started_at is None and valid_ts:
        result.session_started_at = ts

    if entry.get("type") == "assistant" and valid_ts:
        result.last_assistant_response_at = ts

    msg = entry.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, list):
        return

    # Every tool_use / Task block inside this assistant message was emitted by
    # the same model — the message's `model` field is authoritative for per-
    # row attribution. A mid-session model switch shows up as the next
    # assistant message carrying a different model id.
    msg_model: Optional[str] = msg.get("model") if isinstance(msg.get("model"), str) else None
    msg_provider: Optional[str] = _infer_provider_from_model(msg_model)

    for block in content:
        if not isinstance(block, dict):
            continue

        btype = block.get("type")
        bname = block.get("name")
        bid = block.get("id")

        if btype == "tool_use" and bid and bname:
            inp = block.get("input") or {}

            if bname in ("Task", "Agent"):
                # Prefer the explicit `model` on the Task input if present —
                # that's the subagent's chosen model. Otherwise inherit from
                # the enclosing assistant message.
                agent_model = inp.get("model") or msg_model
                agent = AgentCall(
                    id=bid,
                    agent_type=(inp.get("subagent_type") or "agent"),
                    model=agent_model,
                    description=inp.get("description"),
                    status="running",
                    start_time=ts,
                    provider=_infer_provider_from_model(agent_model),
                )
                agent_map[bid] = agent

            elif bname == "TodoWrite":
                raw_todos = inp.get("todos")
                if isinstance(raw_todos, list):
                    latest_todos.clear()
                    for item in raw_todos:
                        if isinstance(item, dict):
                            latest_todos.append(TodoItem(
                                content=str(item.get("content", "")),
                                status=str(item.get("status", "pending")),
                            ))

            else:
                tool = ToolCall(
                    id=bid,
                    name=bname,
                    status="running",
                    start_time=ts,
                    target=_extract_target(bname, inp),
                    provider=msg_provider,
                    model=msg_model,
                )
                tool_map[bid] = tool

        elif btype == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if not tool_use_id:
                continue

            tool = tool_map.get(tool_use_id)
            if tool is not None:
                tool.status = "error" if block.get("is_error") else "completed"
                tool.end_time = ts

            agent = agent_map.get(tool_use_id)
            if agent is not None:
                agent.status = "completed"
                agent.end_time = ts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_transcript(
    transcript_path: str,
    cache_dir: Optional[str] = None,
) -> TranscriptSnapshot:
    """Parse a Claude Code JSONL transcript and return a snapshot.

    Caches on (mtime_ns, size) keyed on sha256(realpath). Cache default:
    ~/.claude/.mullet-hud/transcript-cache/<sha256>.json.  Malformed lines
    are skipped silently.  Returns an empty snapshot on any unrecoverable
    error (missing file, permission denied) — the HUD must never crash.
    """
    empty = TranscriptSnapshot()
    if not transcript_path:
        return empty

    # Resolve to canonical path
    try:
        real_path = os.path.realpath(transcript_path)
    except Exception:
        return empty

    # Stat the file
    try:
        st = os.stat(real_path)
    except OSError:
        return empty

    if not os.path.isfile(real_path):
        return empty

    mtime_ns = st.st_mtime_ns
    size = st.st_size
    resolved_cache_dir = cache_dir or _default_cache_dir()

    # Check cache
    cached = _read_cache(real_path, mtime_ns, size, resolved_cache_dir)
    if cached is not None:
        return cached

    # Parse JSONL
    tool_map: dict = {}
    agent_map: dict = {}
    latest_todos: list[TodoItem] = []
    result = TranscriptSnapshot()

    session_tokens_in = 0
    session_tokens_out = 0
    session_cache_creation = 0
    session_cache_read = 0

    parsed_cleanly = False
    try:
        with open(real_path, encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        continue

                    # Accumulate token usage from assistant entries
                    if entry.get("type") == "assistant":
                        usage = (entry.get("message") or {}).get("usage") or {}
                        session_tokens_in += _normalise_token_count(usage.get("input_tokens", 0))
                        session_tokens_out += _normalise_token_count(usage.get("output_tokens", 0))
                        session_cache_creation += _normalise_token_count(usage.get("cache_creation_input_tokens", 0))
                        session_cache_read += _normalise_token_count(usage.get("cache_read_input_tokens", 0))

                    # compact_boundary
                    if entry.get("type") == "system" and entry.get("subtype") == "compact_boundary":
                        ts_str = entry.get("timestamp")
                        if ts_str:
                            try:
                                ts = datetime.fromisoformat(ts_str)
                                if result.last_compact_boundary_at is None or ts > result.last_compact_boundary_at:
                                    result.last_compact_boundary_at = ts
                                    meta = entry.get("compactMetadata") or {}
                                    post = meta.get("postTokens")
                                    if isinstance(post, (int, float)) and post >= 0:
                                        result.last_compact_post_tokens = int(post)
                            except (ValueError, TypeError):
                                pass

                    _process_entry(entry, tool_map, agent_map, latest_todos, result)

                except (json.JSONDecodeError, Exception):
                    # Skip malformed lines — mirror the TS try/catch on line 264-298
                    continue

        parsed_cleanly = True
    except (OSError, Exception) as exc:
        logger.debug("hud_transcript: error reading %s: %s", real_path, exc)
        # Return partial results

    # Build final snapshot — keep last 20 tools (insertion-order), last 10 agents
    all_tools = list(tool_map.values())[-20:]
    result.running_tools = [t for t in all_tools if t.status == "running"]
    result.completed_tools = [t for t in all_tools if t.status != "running"]
    result.agents = list(agent_map.values())[-10:]
    result.todos = list(latest_todos)
    result.session_tokens_in = session_tokens_in
    result.session_tokens_out = session_tokens_out
    result.session_cache_creation_tokens = session_cache_creation
    result.session_cache_read_tokens = session_cache_read

    if parsed_cleanly:
        _write_cache(real_path, mtime_ns, size, result, resolved_cache_dir)

    return result
