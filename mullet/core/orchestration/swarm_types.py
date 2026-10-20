"""Shared types for Kimi Agent Swarm components."""
from dataclasses import dataclass, field
from mullet.core.agent.types import TokenUsage


@dataclass
class SwarmTask:
    """A single unit of work dispatched to one Kimi call."""
    id: str
    instruction: str
    context_refs: list[str] = field(default_factory=list)
    expected_output_shape: str = "markdown report"
    tier: str = "STANDARD"
    apply: bool = False
    target_dir: str = ""


@dataclass
class SwarmContext:
    """Typed context passed to Decomposer.decompose()."""
    files: list[str] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    output_shape: str = "markdown report"
    synthesis_prompt: str | None = None
    concurrency: int = 20
    apply: bool = False
    target_dir: str = ""

    def __post_init__(self):
        if not isinstance(self.files, list):
            raise TypeError(f"SwarmContext.files must be list[str], got {type(self.files).__name__}")
        for i, x in enumerate(self.files):
            if not isinstance(x, str):
                raise TypeError(f"SwarmContext.files[{i}] must be str, got {type(x).__name__}")
        if not isinstance(self.items, list):
            raise TypeError(f"SwarmContext.items must be list[str], got {type(self.items).__name__}")
        for i, x in enumerate(self.items):
            if not isinstance(x, str):
                raise TypeError(f"SwarmContext.items[{i}] must be str, got {type(x).__name__}")


@dataclass
class TaskResult:
    """Result from a single dispatched SwarmTask."""
    task_id: str
    success: bool
    response: str
    error: str | None
    tokens: TokenUsage = field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0
    files_changed: list[str] = field(default_factory=list)
    provider: str = ""


@dataclass
class SwarmResult:
    """Final result of a complete swarm run."""
    goal: str
    tasks: list[SwarmTask] = field(default_factory=list)
    results: list[TaskResult] = field(default_factory=list)
    synthesis: str = ""
    fell_back: bool = False
    partial_failure: bool = False
    synthesis_fallback: bool = False
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    wall_clock_seconds: float = 0.0
    timestamp: str = ""
    estimated_input_tokens: int = 0    # dry-run estimate before dispatch
    estimated_cost_usd: float = 0.0    # dry-run cost projection