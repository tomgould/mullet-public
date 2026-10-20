"""Smart routing learning — improve routing from past outcomes.

Tracks routing decisions and their outcomes (success/failure, token usage,
tool calls) to make better decisions over time. Uses a simple scoring
system: tasks that succeeded with fewer tokens at a lower tier are evidence
that the tier is sufficient for similar tasks.

This is v1 — keyword-based similarity. Future versions could use embeddings.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

LEARNING_DIR = Path.home() / ".mullet" / "routing_history"
MAX_HISTORY = 500


@dataclass
class RoutingOutcome:
    """Record of a routing decision and its outcome."""
    task_keywords: list[str]
    tier_used: str
    provider_used: str
    success: bool
    tool_calls: int = 0
    tokens_used: int = 0
    timestamp: float = field(default_factory=time.time)


def extract_keywords(task: str) -> list[str]:
    """Extract key terms from a task for similarity matching."""
    import re
    # Normalize and split
    words = re.findall(r"[a-z]+", task.lower())
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "and", "or", "not", "this", "that", "it", "to", "of", "in",
        "for", "with", "on", "at", "from", "by", "as", "do", "does",
        "can", "will", "would", "should", "could", "just", "my", "me",
        "i", "you", "we", "they", "he", "she", "all", "some",
    }
    return [w for w in words if w not in stop_words and len(w) > 2]


def record_outcome(task: str, tier: str, provider: str,
                    success: bool, tool_calls: int = 0, tokens: int = 0):
    """Record a routing outcome for future learning."""
    keywords = extract_keywords(task)
    if not keywords:
        return

    outcome = RoutingOutcome(
        task_keywords=keywords[:10],  # Keep top 10
        tier_used=tier,
        provider_used=provider,
        success=success,
        tool_calls=tool_calls,
        tokens_used=tokens,
    )

    try:
        LEARNING_DIR.mkdir(parents=True, exist_ok=True)
        path = LEARNING_DIR / "outcomes.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps({
                "keywords": outcome.task_keywords,
                "tier": outcome.tier_used,
                "provider": outcome.provider_used,
                "success": outcome.success,
                "tool_calls": outcome.tool_calls,
                "tokens": outcome.tokens_used,
                "ts": outcome.timestamp,
            }) + "\n")

        # Trim to max size
        _trim_history(path)
    except Exception as e:
        logger.debug("Failed to record routing outcome: %s", e)


def suggest_tier(task: str) -> str | None:
    """Suggest a routing tier based on past outcomes for similar tasks.

    Returns the tier that worked best for similar tasks, or None
    if there's not enough history.
    """
    keywords = extract_keywords(task)
    if not keywords:
        return None

    history = _load_history()
    if not history:
        return None

    # Find similar tasks (keyword overlap)
    keyword_set = set(keywords)
    candidates = []
    for entry in history:
        entry_keywords = set(entry.get("keywords", []))
        overlap = len(keyword_set & entry_keywords)
        if overlap >= max(2, len(keyword_set) // 3):
            candidates.append((overlap, entry))

    if len(candidates) < 3:
        return None  # Not enough evidence

    # Score each tier: prefer successful outcomes at lower tiers
    tier_scores: dict[str, list[float]] = {}
    tier_order = {"LOW": 0, "STANDARD": 1, "ADVANCED": 2}

    for overlap, entry in candidates:
        tier = entry.get("tier", "STANDARD")
        if tier not in tier_scores:
            tier_scores[tier] = []

        # Score: higher for success, bonus for lower tier
        score = overlap * 10
        if entry.get("success"):
            score += 50
            score += (2 - tier_order.get(tier, 1)) * 20  # Bonus for cheaper tier
        else:
            score -= 30

        tier_scores[tier].append(score)

    if not tier_scores:  # pragma: no cover — candidates >= 3 guarantees loop always populates tier_scores
        return None

    # Pick tier with highest average score
    best_tier_name, best_tier_scores = max(tier_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))
    if not best_tier_scores:  # pragma: no cover — loop always appends at least one score per tier key
        return None
    avg_score = sum(best_tier_scores) / len(best_tier_scores)

    # Only suggest if confidence is reasonable
    if avg_score > 30 and len(best_tier_scores) >= 2:
        logger.info("Routing learning suggests %s (avg score %.0f from %d matches)",
                     best_tier_name, avg_score, len(best_tier_scores))
        return best_tier_name

    return None


def _load_history() -> list[dict]:
    """Load routing history from disk."""
    path = LEARNING_DIR / "outcomes.jsonl"
    if not path.exists():
        return []

    entries = []
    try:
        for line in path.read_text().strip().split("\n"):
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping corrupt routing history line: %s", e)
                    continue
    except Exception as e:
        logger.warning("Failed to load routing history: %s", e)
        return []
    return entries


def _trim_history(path: Path):
    """Keep only the most recent MAX_HISTORY entries."""
    import types
    fcntl: types.ModuleType | None = None
    try:
        import fcntl
    except ImportError:
        pass

    try:
        with open(path, "r+") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                lines = f.read().strip().split("\n")
                if len(lines) > MAX_HISTORY:
                    f.seek(0)
                    f.write("\n".join(lines[-MAX_HISTORY:]) + "\n")
                    f.truncate()
            finally:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.warning("Failed to trim routing history: %s", e)
