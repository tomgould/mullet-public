"""Task scheduler — simple interval-based scheduling for recurring agent tasks."""

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_INTERVAL_RE = re.compile(r'^(\d+)\s*([smhd])$')
_UNIT_SECONDS = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}


def parse_interval(schedule: str) -> int:
    """Parse interval string to seconds: '5m' -> 300, '1h' -> 3600."""
    m = _INTERVAL_RE.match(schedule.strip().lower())
    if not m:
        raise ValueError(f"Invalid interval: {schedule!r} (use e.g. '5m', '1h', '30s', '1d')")
    return int(m.group(1)) * _UNIT_SECONDS[m.group(2)]


@dataclass
class ScheduledTask:
    """Represents a scheduled task with its execution details."""
    name: str
    schedule: str
    command: str
    enabled: bool = True
    created_at: str = ""
    last_run: str | None = None
    next_run: str | None = None
    run_count: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.next_run:
            self._compute_next_run()

    def _compute_next_run(self):
        try:
            interval = parse_interval(self.schedule)
            if self.last_run:
                base = datetime.fromisoformat(self.last_run)
            else:
                base = datetime.fromisoformat(self.created_at)
            self.next_run = (base + timedelta(seconds=interval)).isoformat()
        except ValueError:
            self.next_run = None


class TaskScheduler:
    """Persistent interval-based task scheduler."""

    def __init__(self, config_path: str = ".mullet/scheduler.json"):
        self._path = Path(config_path)
        self._tasks: dict[str, ScheduledTask] = {}
        self._load()

    def add(self, name: str, schedule: str, command: str) -> ScheduledTask:
        """Register a new task; raises ``ValueError`` if a task of that name already exists."""
        parse_interval(schedule)  # validate
        if name in self._tasks:
            raise ValueError(f"Task '{name}' already exists")
        task = ScheduledTask(name=name, schedule=schedule, command=command)
        self._tasks[name] = task
        self._save()
        return task

    def remove(self, name: str) -> bool:
        """Delete the named task. Returns True if a task was removed."""
        if name in self._tasks:
            del self._tasks[name]
            self._save()
            return True
        return False

    def pause(self, name: str) -> bool:
        """Disable a task without removing it. Returns True if the task existed."""
        task = self._tasks.get(name)
        if task:
            task.enabled = False
            self._save()
            return True
        return False

    def resume(self, name: str) -> bool:
        """Re-enable a paused task and recompute its next run. Returns True if found."""
        task = self._tasks.get(name)
        if task:
            task.enabled = True
            task._compute_next_run()
            self._save()
            return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        """Return all registered tasks in insertion order."""
        return list(self._tasks.values())

    def get_due_tasks(self) -> list[ScheduledTask]:
        """Return every enabled task whose ``next_run`` is at or before now."""
        now = datetime.now()
        due = []
        for task in self._tasks.values():
            if not task.enabled or not task.next_run:
                continue
            if datetime.fromisoformat(task.next_run) <= now:
                due.append(task)
        return due

    def mark_run(self, name: str) -> None:
        """Record that ``name`` just ran: bumps ``run_count`` and reschedules next run."""
        task = self._tasks.get(name)
        if not task:
            return
        task.last_run = datetime.now().isoformat()
        task.run_count += 1
        task._compute_next_run()
        self._save()

    def _save(self):
        """Save scheduler state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(task) for name, task in self._tasks.items()}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._path)

    def _load(self):
        """Load scheduler state from disk."""
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for name, d in data.items():
                self._tasks[name] = ScheduledTask(**d)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load scheduler state: %s", e)
