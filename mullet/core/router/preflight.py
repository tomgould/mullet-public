"""Pre-flight prompt analyzer for Mullet delegation routing."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

from mullet.core.paths import cache_dir
from mullet.core.router.complexity import (
    Tier, is_simple_message, extract_signals, score_complexity,
    score_to_tier, is_reasoning_task,
)
from mullet.core.providers.tokenize import estimate_tokens

logger = logging.getLogger(__name__)

_CACHE_TTL_DAYS = 7
# The in-memory cache, loaded from disk on first use; tests set it directly.
_CACHE: dict[str, PreflightResult] | None = None
# Tests point this at a temporary file; None means preflight.jsonl in the
# cache directory.
_CACHE_PATH: Path | None = None


def _cache_path() -> Path:
    return _CACHE_PATH or cache_dir() / "preflight.jsonl"


@dataclass(frozen=True)
class PreflightResult:
    """Pre-routing analysis: tokens, complexity tier, reasoning need, confidence + provenance."""

    input_tokens: int
    expected_output_tokens: int
    complexity_tier: Tier
    needs_reasoning: bool
    confidence: float
    analyzer: Literal["local_llm", "gemini_flash", "heuristic", "fast_path"]
    latency_ms: int
    cached: bool


def _load_cache() -> dict[str, PreflightResult]:
    """Read the disk cache, dropping stale entries."""
    cache: dict[str, PreflightResult] = {}
    path = _cache_path()
    if not path.exists():
        return cache
    cutoff = time.time() - (_CACHE_TTL_DAYS * 24 * 3600)
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("written_at", 0) < cutoff:
                        continue
                    result = PreflightResult(
                        input_tokens=entry["result"]["input_tokens"],
                        expected_output_tokens=entry["result"]["expected_output_tokens"],
                        complexity_tier=Tier(entry["result"]["complexity_tier"]),
                        needs_reasoning=entry["result"]["needs_reasoning"],
                        confidence=entry["result"]["confidence"],
                        analyzer=entry["result"]["analyzer"],
                        latency_ms=entry["result"]["latency_ms"],
                        cached=True,
                    )
                    cache[entry["key"]] = result
                except Exception:
                    continue
    except Exception as exc:
        logger.debug("Failed to load preflight cache: %s", exc)
    return cache


def _memory_cache() -> dict[str, PreflightResult]:
    """The in-memory cache, read from disk the first time it is needed."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_cache()
    return _CACHE


def _save_to_cache(key: str, result: PreflightResult) -> None:
    """Append *result* to the cache atomically; rewrite whole file if stale."""
    cache = _memory_cache()
    cache[key] = result
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "result": {
            "input_tokens": result.input_tokens,
            "expected_output_tokens": result.expected_output_tokens,
            "complexity_tier": result.complexity_tier.value,
            "needs_reasoning": result.needs_reasoning,
            "confidence": result.confidence,
            "analyzer": result.analyzer,
            "latency_ms": result.latency_ms,
        },
        "written_at": int(time.time()),
    }
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        os.replace(tmp, path)
    except Exception as exc:
        logger.debug("Failed to append to preflight cache: %s", exc)
        return

    # Rewrite whole file to evict stale entries if anything is too old
    cutoff = time.time() - (_CACHE_TTL_DAYS * 24 * 3600)
    if any(
        (getattr(r, "_cached_written_at", None) or 0) < cutoff
        for r in cache.values()
    ):
        try:
            fresh_lines = []
            now = int(time.time())
            for k, r in cache.items():
                rec = {
                    "key": k,
                    "result": {
                        "input_tokens": r.input_tokens,
                        "expected_output_tokens": r.expected_output_tokens,
                        "complexity_tier": r.complexity_tier.value,
                        "needs_reasoning": r.needs_reasoning,
                        "confidence": r.confidence,
                        "analyzer": r.analyzer,
                        "latency_ms": r.latency_ms,
                    },
                    "written_at": now,
                }
                fresh_lines.append(json.dumps(rec, separators=(",", ":")))
            with tmp.open("w", encoding="utf-8") as f:
                f.write("\n".join(fresh_lines) + "\n")
            os.replace(tmp, path)
        except Exception as exc:
            logger.debug("Failed to rewrite preflight cache: %s", exc)


def _cache_key(task: str, files: Sequence[str], agent_type: str | None) -> str:
    raw = task + "|" + ",".join(sorted(files)) + "|" + (agent_type or "")
    return hashlib.sha256(raw.encode()).hexdigest()


_META_PROMPT = """Analyze this engineering task. Reply with ONLY a JSON object, no prose, no markdown:
{{"expected_output_tokens": <integer 50-32000>, "complexity": "LOW|STANDARD|ADVANCED|CRITICAL", "needs_reasoning": <true|false>, "confidence": <0.0-1.0>}}

Task:
<<<{task}>>>"""


def _parse_meta_json(text: str) -> dict | None:
    """Extract and validate JSON from a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    # expected_output_tokens
    try:
        data["expected_output_tokens"] = max(50, min(32000, int(data.get("expected_output_tokens", 512))))
    except (TypeError, ValueError):
        data["expected_output_tokens"] = 512

    # complexity
    comp = str(data.get("complexity", "STANDARD")).upper()
    if comp not in {"LOW", "STANDARD", "ADVANCED", "CRITICAL"}:
        comp = "STANDARD"
    data["complexity"] = comp

    # needs_reasoning
    nr = data.get("needs_reasoning", False)
    if isinstance(nr, str):
        nr = nr.lower() in {"true", "yes", "1"}
    else:
        nr = bool(nr)
    data["needs_reasoning"] = nr

    # confidence
    try:
        data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        data["confidence"] = 0.5

    return data


def analyze_prompt(
    task: str,
    *,
    files: Sequence[str] = (),
    agent_type: str | None = None,
    speed_mode: str = "balanced",
) -> PreflightResult:
    """Analyze a task and return preflight routing metadata."""
    input_tokens = estimate_tokens(task)
    cache_key = _cache_key(task, files, agent_type)
    memory_cache = _memory_cache()
    if cache_key in memory_cache:
        cached = memory_cache[cache_key]
        return PreflightResult(
            input_tokens=cached.input_tokens,
            expected_output_tokens=cached.expected_output_tokens,
            complexity_tier=cached.complexity_tier,
            needs_reasoning=cached.needs_reasoning,
            confidence=cached.confidence,
            analyzer=cached.analyzer,
            latency_ms=cached.latency_ms,
            cached=True,
        )

    if is_simple_message(task):
        result = PreflightResult(
            input_tokens=input_tokens,
            expected_output_tokens=512,
            complexity_tier=Tier.LOW,
            needs_reasoning=False,
            confidence=1.0,
            analyzer="fast_path",
            latency_ms=0,
            cached=False,
        )
        _save_to_cache(cache_key, result)
        return result

    result = _try_local_llm(task)
    if result is None and speed_mode != "economy":
        result = _try_gemini_flash(task)
    if result is None:
        result = _heuristic(task, input_tokens)

    _save_to_cache(cache_key, result)
    return result


def _try_local_llm(task: str) -> PreflightResult | None:
    try:
        from mullet.core.providers.local_detection import detect_local_provider
        provider = detect_local_provider()
        if provider is None:
            return None
        start = time.monotonic()
        # Thinking off: a local reasoning model would otherwise spend the whole
        # 200-token budget thinking and return empty content.
        response = provider.create(
            messages=[{"role": "user", "content": _META_PROMPT.format(task=task[:8000])}],
            max_tokens=200,
            temperature=0.0,
            reasoning_effort="none",
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        data = _parse_meta_json(response.content)
        if data is None:
            return None
        return PreflightResult(
            input_tokens=estimate_tokens(task),
            expected_output_tokens=data["expected_output_tokens"],
            complexity_tier=Tier(data["complexity"]),
            needs_reasoning=data["needs_reasoning"],
            confidence=data["confidence"],
            analyzer="local_llm",
            latency_ms=latency_ms,
            cached=False,
        )
    except Exception as exc:
        logger.debug("Local LLM preflight failed: %s", exc)
        return None


def _try_gemini_flash(task: str) -> PreflightResult | None:
    try:
        from mullet.core.providers.gemini import GeminiProvider
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        provider = GeminiProvider(api_key=api_key)
        start = time.monotonic()
        response = provider.create(
            messages=[{"role": "user", "content": _META_PROMPT.format(task=task[:8000])}],
            max_tokens=200,
            temperature=0.0,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        data = _parse_meta_json(response.content)
        if data is None:
            return None
        return PreflightResult(
            input_tokens=estimate_tokens(task),
            expected_output_tokens=data["expected_output_tokens"],
            complexity_tier=Tier(data["complexity"]),
            needs_reasoning=data["needs_reasoning"],
            confidence=data["confidence"],
            analyzer="gemini_flash",
            latency_ms=latency_ms,
            cached=False,
        )
    except Exception as exc:
        logger.debug("Gemini flash preflight failed: %s", exc)
        return None


def _heuristic(task: str, input_tokens: int) -> PreflightResult:
    signals = extract_signals(task)
    score = score_complexity(signals)
    tier = score_to_tier(score)
    return PreflightResult(
        input_tokens=input_tokens,
        expected_output_tokens=max(256, min(4096, input_tokens // 8)),
        complexity_tier=tier,
        needs_reasoning=is_reasoning_task(task),
        confidence=0.5,
        analyzer="heuristic",
        latency_ms=0,
        cached=False,
    )
