"""RAG retrieval quality measurement framework.

Computes retrieval metrics (MRR, Hit@k, Precision@k, nDCG) and
aggregated reports for search result quality assessment.

Ported from Trove core/evaluation.py.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case for retrieval quality."""
    query: str
    expected_ids: list[str]
    expected_answer: str = ""

    # Tell pytest not to collect this as a test class.
    __test__ = False


@dataclass
class RetrievalResult:
    """Result for a single query's retrieval quality."""
    query: str
    mrr: float = 0.0
    hit_at_k: dict[int, bool] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    ndcg: float = 0.0
    retrieved_ids: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Aggregated quality report across multiple queries."""
    results: list[RetrievalResult] = field(default_factory=list)
    mean_mrr: float = 0.0
    mean_ndcg: float = 0.0
    hit_rates: dict[int, float] = field(default_factory=dict)
    mean_precision: dict[int, float] = field(default_factory=dict)
    total_queries: int = 0

    def summary(self) -> str:
        """Render the report as a multi-line human-readable string."""
        lines = [
            f"RAG Quality Report ({self.total_queries} queries)",
            f"  MRR:      {self.mean_mrr:.3f}",
            f"  nDCG:     {self.mean_ndcg:.3f}",
        ]
        for k in sorted(self.hit_rates):
            lines.append(f"  Hit@{k}:   {self.hit_rates[k]:.1%}")
        for k in sorted(self.mean_precision):
            lines.append(f"  P@{k}:     {self.mean_precision[k]:.3f}")
        return "\n".join(lines)


def load_test_cases(path: str | Path) -> list[TestCase]:
    """Load test cases from a JSON file.

    Format: [{"query": "...", "expected_ids": ["id1", "id2"]}]
    """
    data = json.loads(Path(path).read_text())
    return [TestCase(**tc) for tc in data]


def measure_retrieval(
    results: list[dict],
    expected_ids: list[str],
    k_values: list[int] | None = None,
) -> RetrievalResult:
    """Measure retrieval quality for a single query.

    Args:
        results: Search results with "chunk_id" or "id" field
        expected_ids: Ground truth relevant chunk IDs
        k_values: Values of k for Hit@k and P@k (default: [1, 3, 5, 10])
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    expected = set(expected_ids)
    retrieved = []
    for r in results:
        rid = r.get("chunk_id") or r.get("id") or r.get("_id", "")
        retrieved.append(rid)

    rv = RetrievalResult(query="", retrieved_ids=retrieved)

    # MRR (Mean Reciprocal Rank)
    for i, rid in enumerate(retrieved):
        if rid in expected:
            rv.mrr = 1.0 / (i + 1)
            break

    # Hit@k and Precision@k
    for k in k_values:
        top_k = set(retrieved[:k])
        hits = len(top_k & expected)
        rv.hit_at_k[k] = hits > 0
        rv.precision_at_k[k] = hits / min(k, len(expected)) if expected else 0.0

    # nDCG
    rv.ndcg = _compute_ndcg(retrieved, expected)

    return rv


def aggregate_results(results: list[RetrievalResult]) -> QualityReport:
    """Aggregate individual results into a report."""
    if not results:
        return QualityReport()

    report = QualityReport(results=results, total_queries=len(results))
    report.mean_mrr = sum(r.mrr for r in results) / len(results)
    report.mean_ndcg = sum(r.ndcg for r in results) / len(results)

    all_k: set[int] = set()
    for r in results:
        all_k.update(r.hit_at_k.keys())
        all_k.update(r.precision_at_k.keys())

    for k in all_k:
        hits = sum(1 for r in results if r.hit_at_k.get(k, False))
        report.hit_rates[k] = hits / len(results)
        precisions = [r.precision_at_k.get(k, 0.0) for r in results]
        report.mean_precision[k] = sum(precisions) / len(precisions)

    return report


def _compute_ndcg(retrieved: list[str], expected: set[str], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain."""
    dcg = 0.0
    for i, rid in enumerate(retrieved[:k]):
        if rid in expected:
            dcg += 1.0 / math.log2(i + 2)

    ideal_dcg = 0.0
    for i in range(min(len(expected), k)):
        ideal_dcg += 1.0 / math.log2(i + 2)

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
