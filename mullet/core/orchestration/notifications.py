"""Manages system notifications and message logging across the orchestration layer."""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class Notification:
    """Represents a single system notification."""
    message: str
    level: str = 'info'
    timestamp: str = ''
    source: str = 'system'

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class NotificationManager:
    """Store and manage system notifications."""

    def __init__(self):
        self._notifications: List[Notification] = []

    def notify(self, message: str, level: str = 'info', source: str = 'system'):
        """Add a new notification to the notification list."""
        notification = Notification(message=message, level=level, source=source)
        self._notifications.append(notification)

    def get_recent(self, count: int = 10) -> List[Notification]:
        """Get the most recent notifications up to the specified count."""
        return self._notifications[-count:]

    def clear(self):
        """Clear all notifications from the list."""
        self._notifications = []

    @property
    def unread_count(self) -> int:
        """Return the total number of stored notifications."""
        return len(self._notifications)
