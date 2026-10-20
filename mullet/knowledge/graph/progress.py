"""
Progress tracking for knowledge graph building.
"""

import time
from dataclasses import dataclass, field


@dataclass
class GraphBuildProgress:
    """Track progress of graph extraction."""

    total_chunks: int
    processed_chunks: int = 0
    entities_found: int = 0
    relationships_found: int = 0
    failed_chunks: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def percent(self) -> float:
        """Return the completion percentage."""
        if self.total_chunks == 0:
            return 100.0
        return (self.processed_chunks / self.total_chunks) * 100.0

    @property
    def eta_seconds(self) -> float | None:
        """Return the estimated time remaining in seconds."""
        if self.processed_chunks == 0:
            return None
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return None
        rate = self.processed_chunks / elapsed
        remaining = self.total_chunks - self.processed_chunks
        return remaining / rate if rate > 0 else None

    def format_eta(self) -> str:
        """Return a human-readable ETA string."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."
        minutes, seconds = divmod(int(eta), 60)
        if minutes > 0:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"
