"""Semantic task-signature helpers for routing-history re-rank.

Provides lightweight embedding + cosine similarity so that
``routing_priors.pick_better_provider_semantic`` can match paraphrased
tasks against existing history entries.

Backend resolution (tried in order):
  1. sentence-transformers ``all-MiniLM-L6-v2`` (384-d), cached per-process.
  2. Character-trigram hash vector (256-d) — always available, deterministic.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence-transformers model — loaded at most once per process
# ---------------------------------------------------------------------------

_ST_MODEL = None          # SentenceTransformer instance once loaded
_ST_AVAILABLE: bool | None = None  # None = not yet probed


def _get_st_model():
    """Return cached SentenceTransformer model or None if unavailable."""
    global _ST_MODEL, _ST_AVAILABLE
    if _ST_AVAILABLE is False:
        return None
    if _ST_MODEL is not None:
        return _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _ST_AVAILABLE = True
        logger.debug("semantic_sig: sentence-transformers loaded (384-d)")
        return _ST_MODEL
    except Exception:
        _ST_AVAILABLE = False
        logger.debug("semantic_sig: sentence-transformers unavailable — using trigram fallback")
        return None


# ---------------------------------------------------------------------------
# Trigram hash vector (256-d)
# ---------------------------------------------------------------------------

def _trigram_embed(text: str) -> list[float]:
    """Return a 256-d trigram hash vector, L2-normalised.

    For each 3-character window in the lowercased text, map to one of 256
    buckets using the first byte of its MD5 digest, accumulate counts, then
    divide by the L2 norm to produce a unit vector.
    """
    counts = [0.0] * 256
    t = text.lower()
    for i in range(len(t) - 2):
        trigram = t[i : i + 3]
        bucket = hashlib.md5(trigram.encode()).digest()[0] % 256
        counts[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in counts))
    if norm > 0.0:
        return [x / norm for x in counts]
    # All-zero vector (text shorter than 3 chars): return as-is
    return counts


# ---------------------------------------------------------------------------
# Public: embed
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float] | None:
    """Return a fixed-size embedding vector for *text*, or None on total failure.

    Tries sentence-transformers (384-d) first; falls back to trigram (256-d).
    The trigram path should never fail, so None is only returned on a truly
    unexpected error.
    """
    model = _get_st_model()
    if model is not None:
        try:
            vec = model.encode(text, normalize_embeddings=True)
            # Force plain Python floats — sentence-transformers returns numpy
            # ndarray whose elements are numpy.float32 and some callers expect
            # built-in floats (JSON serialisation, strict isinstance checks).
            return [float(x) for x in vec]
        except Exception as exc:
            logger.debug("semantic_sig: ST encode failed, falling back: %s", exc)

    # Trigram fallback — always available
    try:
        return _trigram_embed(text)
    except Exception as exc:
        logger.warning("semantic_sig: trigram embed failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public: cosine
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, clamped to [0, 1].

    Returns 0.0 on empty or length-mismatched input without raising.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

@dataclass
class _EmbedCache:
    """In-process cache of (signature → (task_preview, vector)) pairs."""

    mtime: float
    entries: dict[str, tuple[str, list[float]]] = field(default_factory=dict)


_embed_cache: _EmbedCache | None = None


def _build_cache(path: Path) -> _EmbedCache:
    """Read history file, embed each distinct task_preview, return cache."""
    mtime = path.stat().st_mtime
    entries: dict[str, tuple[str, list[float]]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                sig = d.get("task_signature", "")
                preview = d.get("task_preview", "")
                if sig and preview and sig not in entries:
                    vec = embed(preview)
                    if vec is not None:
                        entries[sig] = (preview, vec)
            except Exception:
                continue
    except OSError as exc:
        logger.warning("semantic_sig: failed to build embed cache: %s", exc)
    return _EmbedCache(mtime=mtime, entries=entries)


# ---------------------------------------------------------------------------
# Public: nearest_signatures
# ---------------------------------------------------------------------------

def nearest_signatures(
    task: str,
    threshold: float = 0.75,
    limit: int = 5,
    base: Optional[Path] = None,
) -> list[tuple[str, float]]:
    """Return *(signature, similarity)* pairs from routing history.

    Searches the ``routing_history.jsonl`` file for entries whose
    ``task_preview`` has cosine similarity >= *threshold* with *task*.

    Results are sorted by similarity descending; among ties the order
    follows the cache-load order (newest entry in the file wins since
    the file is append-only and we preserve insertion order).

    The embedding cache is invalidated whenever the history file's mtime
    changes (i.e. a new record was appended).

    Returns ``[]`` on any error — never raises.
    """
    global _embed_cache
    try:
        from mullet.core.routing_history import history_path

        path = history_path(base)
        if not path.exists():
            return []

        # Refresh cache if file has changed
        current_mtime = path.stat().st_mtime
        if _embed_cache is None or _embed_cache.mtime != current_mtime:
            _embed_cache = _build_cache(path)

        task_vec = embed(task)
        if task_vec is None:
            return []

        scored: list[tuple[str, float]] = []
        for sig, (_preview, vec) in _embed_cache.entries.items():
            sim = cosine(task_vec, vec)
            if sim >= threshold:
                scored.append((sig, sim))

        # Stable sort: highest similarity first; insertion order preserved on ties
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    except Exception as exc:
        logger.debug("semantic_sig: nearest_signatures error (non-fatal): %s", exc)
        return []
