"""Reranker — re-score search results for improved relevance.

Provides a simple keyword-overlap reranker (no external dependencies)
and a cross-encoder reranker (requires sentence-transformers).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol


class Reranker(Protocol):
    """Interface for reranking search results."""

    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """Re-score and re-order results by relevance to query."""
        ...


@dataclass
class RerankResult:
    """A search result with a reranking score."""
    content: str
    score: float
    metadata: dict


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer — lowercase, strip punctuation."""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


class SimpleReranker:
    """Keyword-overlap reranker — no external dependencies.

    Scores results using TF-IDF-inspired keyword overlap:
    - Term frequency of query words in each result
    - Bonus for exact phrase matches
    - Penalty for very short results
    """

    def __init__(self, phrase_boost: float = 2.0):
        self.phrase_boost = phrase_boost

    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """Re-score results by keyword relevance to query."""
        if not results or not query:
            return results[:top_k]

        query_tokens = _tokenize(query)
        if not query_tokens:
            return results[:top_k]

        query_lower = query.lower()
        query_tf = Counter(query_tokens)

        # Score each result
        scored: list[tuple[float, int, dict]] = []
        for i, result in enumerate(results):
            content = result.get("content", "")
            score = self._score(content, query_tokens, query_tf, query_lower)
            scored.append((score, i, result))

        # Sort by score descending, then by original index for ties
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Return top_k, adding rerank_score to each
        output = []
        for score, _, result in scored[:top_k]:
            enriched = dict(result)
            enriched["rerank_score"] = round(score, 4)
            output.append(enriched)

        return output

    def _score(
        self,
        content: str,
        query_tokens: list[str],
        query_tf: Counter,
        query_lower: str,
    ) -> float:
        """Score a single result against the query."""
        content_tokens = _tokenize(content)
        if not content_tokens:
            return 0.0

        content_tf = Counter(content_tokens)
        content_len = len(content_tokens)

        # Term overlap score (sum of min TF ratios)
        overlap = 0.0
        matched_terms = 0
        for token in query_tf:
            if token in content_tf:
                # TF component: how frequent is this query term in the result?
                tf = content_tf[token] / content_len
                overlap += tf * query_tf[token]
                matched_terms += 1

        # Coverage: what fraction of unique query terms matched?
        unique_query = len(set(query_tokens))
        coverage = matched_terms / unique_query if unique_query > 0 else 0.0

        # Exact phrase bonus
        phrase_bonus = 0.0
        if len(query_tokens) > 1 and query_lower in content.lower():
            phrase_bonus = self.phrase_boost

        # Length normalization (slight penalty for very short results)
        length_factor = min(1.0, math.log(content_len + 1) / 5.0)

        return (overlap + coverage + phrase_bonus) * length_factor


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    Requires: pip install sentence-transformers
    Uses a cross-encoder model to score (query, document) pairs.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers required for CrossEncoderReranker. "
                    "Run: pip install sentence-transformers"
                ) from exc
        return self._model

    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """Re-score using cross-encoder model."""
        if not results or not query:
            return results[:top_k]

        pairs = [(query, r.get("content", "")) for r in results]
        scores = self.model.predict(pairs)

        scored = list(zip(scores, range(len(results)), results))
        scored.sort(key=lambda x: (-x[0], x[1]))

        output = []
        for score, _, result in scored[:top_k]:
            enriched = dict(result)
            enriched["rerank_score"] = round(float(score), 4)
            output.append(enriched)

        return output


def get_reranker(model_name: str | None = None) -> SimpleReranker | CrossEncoderReranker:
    """Factory: get a reranker. None = SimpleReranker, string = CrossEncoder."""
    if model_name is None:
        return SimpleReranker()
    return CrossEncoderReranker(model_name)
