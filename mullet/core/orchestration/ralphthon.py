"""Ralphthon — marathon autonomous mode with PRD, stories, tasks, hardening.

Lifecycle: interview → execution → hardening → complete
Terminates after N consecutive clean hardening waves with no new issues.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RalphthonTask:
    """Represents a single task in a ralphthon story."""
    id: str
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, done, skipped, failed
    retries: int = 0
    notes: str = ""


@dataclass
class RalphthonStory:
    """Groups related tasks into a user story with acceptance criteria."""
    id: str
    title: str
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    priority: str = "medium"  # critical, high, medium, low
    tasks: list[RalphthonTask] = field(default_factory=list)

    @property
    def progress(self) -> float:
        """Return completion ratio of tasks in this story."""
        if not self.tasks:
            return 0.0
        done = sum(1 for t in self.tasks if t.status in ("done", "skipped"))
        return done / len(self.tasks)


@dataclass
class HardeningTask:
    """Represents a hardening task for a specific quality wave."""
    id: str
    title: str
    description: str
    category: str = "quality"  # edge_case, test, quality, security, performance
    status: str = "pending"
    wave: int = 1
    retries: int = 0
    notes: str = ""


@dataclass
class RalphthonConfig:
    """Configuration parameters for ralphthon execution."""
    max_waves: int = 10
    clean_waves_for_termination: int = 3
    max_retries: int = 3


@dataclass
class RalphthonPRD:
    """Product requirements document for a ralphthon session."""
    project: str
    description: str
    stories: list[RalphthonStory] = field(default_factory=list)
    hardening: list[HardeningTask] = field(default_factory=list)
    config: RalphthonConfig = field(default_factory=RalphthonConfig)

    def to_dict(self) -> dict:
        """Serialize the PRD to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RalphthonPRD":
        """Deserialize a PRD from a dictionary."""
        config = RalphthonConfig(**data.get("config", {}))
        stories = []
        for s in data.get("stories", []):
            tasks = [RalphthonTask(**t) for t in s.get("tasks", [])]
            stories.append(RalphthonStory(
                id=s["id"], title=s["title"], description=s["description"],
                acceptance_criteria=s.get("acceptance_criteria", []),
                priority=s.get("priority", "medium"), tasks=tasks,
            ))
        hardening = [HardeningTask(**h) for h in data.get("hardening", [])]
        return cls(
            project=data["project"], description=data["description"],
            stories=stories, hardening=hardening, config=config,
        )


class RalphthonState:
    """Tracks the state of a ralphthon session."""

    def __init__(self, prd: RalphthonPRD):
        self.prd = prd
        self.phase: str = "execution"  # interview, execution, hardening, complete, failed
        self.current_wave: int = 0
        self.consecutive_clean_waves: int = 0
        self.tasks_completed: int = 0
        self.tasks_skipped: int = 0
        self.started_at: str = datetime.now().isoformat()

    def get_next_task(self) -> RalphthonTask | HardeningTask | None:
        """Get the next pending task based on current phase."""
        if self.phase == "execution":
            for story in self.prd.stories:
                for task in story.tasks:
                    if task.status == "pending":
                        return task
            # All execution tasks done — transition to hardening
            self.phase = "hardening"
            self.current_wave = 1

        if self.phase == "hardening":
            for ht in self.prd.hardening:
                if ht.status == "pending" and ht.wave == self.current_wave:
                    return ht
            return None

        return None

    def complete_task(self, task_id: str, success: bool = True, notes: str = "") -> None:
        """Mark a task as done or retry/skip it on failure."""
        task = self._find_task(task_id)
        if not task:
            return
        if success:
            task.status = "done"
            task.notes = notes
            self.tasks_completed += 1
        else:
            task.retries += 1
            if task.retries >= self.prd.config.max_retries:
                task.status = "skipped"
                task.notes = f"Skipped after {task.retries} retries: {notes}"
                self.tasks_skipped += 1
            else:
                task.status = "pending"
                task.notes = notes

    def start_hardening_wave(self, new_tasks: list[HardeningTask]) -> int:
        """Start a new hardening wave. Returns wave number."""
        self.current_wave += 1
        for ht in new_tasks:
            ht.wave = self.current_wave
            self.prd.hardening.append(ht)
        if not new_tasks:
            self.consecutive_clean_waves += 1
        else:
            self.consecutive_clean_waves = 0
        if self.consecutive_clean_waves >= self.prd.config.clean_waves_for_termination:
            self.phase = "complete"
        if self.current_wave > self.prd.config.max_waves:
            self.phase = "complete"
        return self.current_wave

    def _find_task(self, task_id: str) -> RalphthonTask | HardeningTask | None:
        for story in self.prd.stories:
            for task in story.tasks:
                if task.id == task_id:
                    return task
        for ht in self.prd.hardening:
            if ht.id == task_id:
                return ht
        return None

    @property
    def is_complete(self) -> bool:
        """Return whether the session has finished or failed."""
        return self.phase in ("complete", "failed")

    def to_dict(self) -> dict:
        """Serialize the current state to a dictionary."""
        return {
            "prd": self.prd.to_dict(),
            "phase": self.phase,
            "current_wave": self.current_wave,
            "consecutive_clean_waves": self.consecutive_clean_waves,
            "tasks_completed": self.tasks_completed,
            "tasks_skipped": self.tasks_skipped,
            "started_at": self.started_at,
        }

    def save(self, path: str) -> None:
        """Persist the state to a JSON file atomically."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(p)

    @classmethod
    def load(cls, path: str) -> "RalphthonState":
        """Restore a RalphthonState from a JSON file."""
        data = json.loads(Path(path).read_text())
        prd = RalphthonPRD.from_dict(data["prd"])
        state = cls(prd)
        state.phase = data["phase"]
        state.current_wave = data["current_wave"]
        state.consecutive_clean_waves = data["consecutive_clean_waves"]
        state.tasks_completed = data["tasks_completed"]
        state.tasks_skipped = data["tasks_skipped"]
        state.started_at = data["started_at"]
        return state
