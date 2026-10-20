"""Tests for NotificationManager."""
from mullet.core.orchestration.notifications import NotificationManager, Notification


def test_notify_and_get():
    nm = NotificationManager()
    nm.notify("hello")
    assert nm.unread_count == 1
    recent = nm.get_recent()
    assert len(recent) == 1
    assert recent[0].message == "hello"
    assert recent[0].level == "info"


def test_clear():
    nm = NotificationManager()
    nm.notify("a")
    nm.notify("b")
    assert nm.unread_count == 2
    nm.clear()
    assert nm.unread_count == 0


def test_get_recent_limit():
    nm = NotificationManager()
    for i in range(20):
        nm.notify(f"msg{i}")
    assert len(nm.get_recent(5)) == 5


def test_notification_timestamp():
    n = Notification(message="test")
    assert n.timestamp != ""
    assert n.source == "system"
