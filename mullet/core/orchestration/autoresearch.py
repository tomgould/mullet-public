"""Automated research pipeline — iterative research with scoring and refinement."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResearchFinding:
    """A single piece of research content with relevance scoring."""
    content: str
    source: str
    relevance: float = 1.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ResearchIteration:
    """One cycle of research comprising a query and its findings."""
    iteration: int
    query: str
    findings: list[ResearchFinding] = field(default_factory=list)
    summary: str = ""
    score: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ResearchPipeline:
    """Manages an iterative research process with scoring and refinement."""

    def __init__(self, question: str, max_iterations: int = 5, min_score: float = 0.8):
        self.question = question
        self.max_iterations = max_iterations
        self.min_score = min_score
        self.iterations: list[ResearchIteration] = []
        self.status: str = "pending"

    def add_iteration(self, query: str, findings: list[ResearchFinding],
                      summary: str, score: float) -> ResearchIteration:
        """Append a new research iteration and update pipeline status."""
        it = ResearchIteration(
            iteration=len(self.iterations) + 1,
            query=query,
            findings=findings,
            summary=summary,
            score=score,
        )
        self.iterations.append(it)
        self.status = "running"
        if not self.should_continue():
            self.status = "completed"
        return it

    def should_continue(self) -> bool:
        """Check whether another research iteration is needed."""
        if not self.iterations:
            return True
        if len(self.iterations) >= self.max_iterations:
            return False
        return self.iterations[-1].score < self.min_score

    def best_iteration(self) -> ResearchIteration | None:
        """Return the iteration with the highest score."""
        if not self.iterations:
            return None
        return max(self.iterations, key=lambda i: i.score)

    def get_refined_query(self) -> str:
        """Generate a follow-up query based on the last iteration."""
        if not self.iterations:
            return self.question
        last = self.iterations[-1]
        return f"{self.question} (refine: previous score {last.score:.1f}, gaps in: {last.summary[:100]})"

    def to_dict(self) -> dict:
        """Serialize the pipeline to a dictionary."""
        return {
            "question": self.question,
            "max_iterations": self.max_iterations,
            "min_score": self.min_score,
            "status": self.status,
            "iterations": [
                {
                    **asdict(it),
                    "findings": [asdict(f) for f in it.findings],
                }
                for it in self.iterations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchPipeline":
        """Deserialize a pipeline from a dictionary."""
        p = cls(
            question=data["question"],
            max_iterations=data.get("max_iterations", 5),
            min_score=data.get("min_score", 0.8),
        )
        p.status = data.get("status", "pending")
        for it_data in data.get("iterations", []):
            findings = [ResearchFinding(**f) for f in it_data.get("findings", [])]
            it = ResearchIteration(
                iteration=it_data["iteration"],
                query=it_data["query"],
                findings=findings,
                summary=it_data.get("summary", ""),
                score=it_data.get("score", 0.0),
                timestamp=it_data.get("timestamp", ""),
            )
            p.iterations.append(it)
        return p

    def save(self, path: str) -> None:
        """Persist the pipeline to a JSON file atomically."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(p)

    @classmethod
    def load(cls, path: str) -> "ResearchPipeline":
        """Restore a ResearchPipeline from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
