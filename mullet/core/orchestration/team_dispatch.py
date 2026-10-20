"""Manages dispatch rate limiting and cooldowns for team-based agent orchestration."""
import time

class TeamDispatchManager:
    """Manages dispatch rate limiting and cooldowns for team-based agent orchestration."""
    def __init__(
        self,
        issue_cooldown: float = 900.0,  # 15 minutes
        trigger_cooldown: float = 30.0,  # 30 seconds
        max_retries: int = 3,
    ):
        self.issue_cooldown = issue_cooldown
        self.trigger_cooldown = trigger_cooldown
        self.max_retries = max_retries

        self.last_dispatch_time: float | None = None
        self.issue_dispatch_counts: dict[str, int] = {}
        self.issue_last_dispatch_time: dict[str, float] = {}

    def can_dispatch(self, issue_id: str) -> bool:
        """Return True if the issue can be dispatched given current cooldowns."""
        current_time = time.time()

        # Check trigger cooldown
        if self.last_dispatch_time and (current_time - self.last_dispatch_time) < self.trigger_cooldown:
            return False

        # Check issue cooldown
        if issue_id in self.issue_last_dispatch_time and \
           (current_time - self.issue_last_dispatch_time[issue_id]) < self.issue_cooldown:
            return False

        # Check max retries
        if self.issue_dispatch_counts.get(issue_id, 0) >= self.max_retries:
            return False

        return True

    def record_dispatch(self, issue_id: str):
        """Record a dispatch for the given issue and update cooldown state."""
        current_time = time.time()
        self.last_dispatch_time = current_time
        self.issue_dispatch_counts[issue_id] = self.issue_dispatch_counts.get(issue_id, 0) + 1
        self.issue_last_dispatch_time[issue_id] = current_time