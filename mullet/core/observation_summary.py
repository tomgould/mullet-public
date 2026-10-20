"""LLM-powered summarisation of raw observation captures.

Converts raw tool-use events from :mod:`mullet.core.observations` into
structured :class:`SessionSummary` objects via a cheap LLM call.  Fail-open
throughout — no exception ever escapes public functions.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from mullet.core.observations import iter_recent, iter_session, Observation

logger = logging.getLogger("mullet.core.observation_summary")

_MAX_PROMPT_CHARS = 16_000
_FILE_TOOLS = {"Read", "Write", "Edit", "Glob"}
_FILE_PATH_RE = re.compile(r'["\']?(/[^\s"\'>,]+)["\']?')


@dataclass
class SessionSummary:
    """Summary of a single developer tool session."""
    session_id: str
    started_at: str
    ended_at: str
    tool_counts: dict[str, int]
    files_touched: list[str]
    summary: str
    bullet_points: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fact collection (no LLM)
# ---------------------------------------------------------------------------

def collect_session_facts(
    session_id: str,
    base: Optional[Path] = None,
    max_observations: int = 200,
) -> dict:
    """Return a stats dict from raw observations — no LLM call."""
    obs_list = list(iter_session(session_id, base=base))[:max_observations]
    return _facts_from_obs(obs_list, session_id)


def _facts_from_obs(obs_list: list[Observation], session_id: str) -> dict:
    if not obs_list:
        return {
            "session_id": session_id,
            "started_at": "",
            "ended_at": "",
            "tool_counts": {},
            "files_touched": [],
            "excerpts": [],
        }

    timestamps = [o.timestamp for o in obs_list]
    started_at = min(timestamps)
    ended_at = max(timestamps)

    tool_counts: dict[str, int] = {}
    files_seen: dict[str, None] = {}  # ordered-set via dict

    for obs in obs_list:
        tool_counts[obs.tool_name] = tool_counts.get(obs.tool_name, 0) + 1
        if obs.tool_name in _FILE_TOOLS:
            for m in _FILE_PATH_RE.finditer(obs.tool_input_preview):
                p = m.group(1)
                if len(p) > 3:
                    files_seen[p] = None

    excerpts = [
        f"[{o.timestamp[11:16]}] {o.tool_name}: {o.tool_input_preview[:120]}"
        for o in obs_list
    ]

    return {
        "session_id": session_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "tool_counts": tool_counts,
        "files_touched": list(files_seen),
        "excerpts": excerpts,
    }


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def _resolve_provider(provider: str):
    """Return an LLM provider object or None if nothing is available."""
    want_local = provider in ("local", "auto")
    want_haiku = provider in ("haiku", "auto")

    if want_local:
        try:
            from mullet.core.providers.local_detection import detect_local_provider
            p = detect_local_provider()
            if p is not None:
                return p
        except Exception as exc:
            logger.debug("local provider unavailable: %s", exc)

    if want_haiku:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                from mullet.core.providers.anthropic import AnthropicProvider
                return AnthropicProvider(api_key=api_key, model="claude-haiku-4-5")
            except Exception as exc:
                logger.debug("AnthropicProvider init failed: %s", exc)

    logger.warning(
        "observation_summary: no provider available for provider=%r (tried: %s)",
        provider,
        "local+haiku" if provider == "auto" else provider,
    )
    return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are summarising a developer's recent tool activity. "
    "Given these observations, produce a JSON object with keys "
    "`summary` (3-6 sentences), `bullet_points` (3-8 bullets), "
    "`open_questions` (3-8 items). Only output JSON."
)


def _build_prompt(facts: dict) -> str:
    header = (
        f"Session: {facts['session_id']}\n"
        f"Period: {facts['started_at']} → {facts['ended_at']}\n"
        f"Tool counts: {json.dumps(facts['tool_counts'])}\n"
        f"Files touched: {', '.join(facts['files_touched'][:30]) or 'none'}\n\n"
        "Recent activity excerpts:\n"
    )
    lines = "\n".join(facts.get("excerpts", []))
    full = header + lines
    if len(full) > _MAX_PROMPT_CHARS:
        # Keep first 25% and last 75% — middle is usually least useful
        keep_front = _MAX_PROMPT_CHARS // 4
        keep_tail = _MAX_PROMPT_CHARS - keep_front - 60
        full = full[:keep_front] + "\n... [truncated] ...\n" + full[-keep_tail:]
    return full


# ---------------------------------------------------------------------------
# LLM call + JSON parse
# ---------------------------------------------------------------------------

def _call_llm(provider, prompt: str) -> dict:
    """Call the provider and parse JSON; fall back gracefully on failure."""
    messages = [
        {"role": "user", "content": _SYSTEM_PROMPT + "\n\n" + prompt},
    ]
    try:
        resp = provider.create(messages=messages, max_tokens=1024, temperature=0.3)
        raw = resp.content.strip()
    except Exception as exc:
        logger.warning("observation_summary: LLM call failed: %s", exc)
        return {"summary": "", "bullet_points": [], "open_questions": []}

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")

    try:
        parsed = json.loads(raw)
        return {
            "summary": str(parsed.get("summary", "")),
            "bullet_points": [str(b) for b in parsed.get("bullet_points", [])],
            "open_questions": [str(q) for q in parsed.get("open_questions", [])],
        }
    except (json.JSONDecodeError, TypeError):
        logger.warning("observation_summary: JSON parse failed; using raw text")
        return {
            "summary": raw[:500],
            "bullet_points": [],
            "open_questions": [],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarise_session(
    session_id: str,
    *,
    provider: str = "auto",
    max_observations: int = 200,
    base: Optional[Path] = None,
) -> Optional[SessionSummary]:
    """Summarise a single session via LLM.  Returns None if no data or unavailable."""
    try:
        facts = collect_session_facts(session_id, base=base, max_observations=max_observations)
        if not facts["started_at"]:
            logger.info("observation_summary: no observations for session %s", session_id)
            return None

        llm = _resolve_provider(provider)
        if llm is None:
            return None

        result = _call_llm(llm, _build_prompt(facts))
        return SessionSummary(
            session_id=session_id,
            started_at=facts["started_at"],
            ended_at=facts["ended_at"],
            tool_counts=facts["tool_counts"],
            files_touched=facts["files_touched"],
            summary=result["summary"],
            bullet_points=result["bullet_points"],
            open_questions=result["open_questions"],
        )
    except Exception as exc:
        logger.error("observation_summary: unexpected error: %s", exc)
        return None


def summarise_window(
    minutes: int = 60,
    *,
    provider: str = "auto",
    base: Optional[Path] = None,
) -> Optional[SessionSummary]:
    """Summarise all observations within the last *minutes* minutes."""
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

        all_obs = list(iter_recent(limit=10_000, base=base))
        window_obs = [o for o in all_obs if o.timestamp >= cutoff_str]

        if not window_obs:
            logger.info("observation_summary: no observations in last %d minutes", minutes)
            return None

        window_obs.sort(key=lambda o: o.timestamp)
        session_id = f"window-{minutes}m"
        facts = _facts_from_obs(window_obs, session_id)

        llm = _resolve_provider(provider)
        if llm is None:
            return None

        result = _call_llm(llm, _build_prompt(facts))
        return SessionSummary(
            session_id=session_id,
            started_at=facts["started_at"],
            ended_at=facts["ended_at"],
            tool_counts=facts["tool_counts"],
            files_touched=facts["files_touched"],
            summary=result["summary"],
            bullet_points=result["bullet_points"],
            open_questions=result["open_questions"],
        )
    except Exception as exc:
        logger.error("observation_summary: unexpected error in summarise_window: %s", exc)
        return None
