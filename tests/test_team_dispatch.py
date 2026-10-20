import pytest
from unittest.mock import patch

from mullet.core.orchestration.team_dispatch import TeamDispatchManager

@pytest.fixture
def manager():
    return TeamDispatchManager(
        issue_cooldown=900.0,
        trigger_cooldown=30.0,
        max_retries=3,
    )

def test_initial_state(manager):
    assert manager.last_dispatch_time is None
    assert manager.issue_dispatch_counts == {}
    assert manager.issue_last_dispatch_time == {}

def test_can_dispatch_initially(manager):
    assert manager.can_dispatch("issue-1") is True

def test_record_dispatch(manager):
    with patch("mullet.core.orchestration.team_dispatch.time") as mock_time:
        mock_time.time.return_value = 1672560000.0  # Corresponds to datetime(2023, 1, 1, 10, 0, 0)
        manager.record_dispatch("issue-1")

        assert manager.last_dispatch_time == 1672560000.0
        assert manager.issue_dispatch_counts == {"issue-1": 1}
        assert manager.issue_last_dispatch_time == {"issue-1": 1672560000.0}

def test_trigger_cooldown(manager):
    with patch("mullet.core.orchestration.team_dispatch.time") as mock_time:
        # First dispatch
        mock_time.time.return_value = 1672560000.0
        manager.record_dispatch("issue-1")

        # Within trigger cooldown
        mock_time.time.return_value = 1672560015.0  # 15 seconds later
        assert manager.can_dispatch("issue-2") is False

        # After trigger cooldown
        mock_time.time.return_value = 1672560031.0  # 31 seconds later
        assert manager.can_dispatch("issue-2") is True

def test_issue_cooldown(manager):
    with patch("mullet.core.orchestration.team_dispatch.time") as mock_time:
        # First dispatch for issue-1
        mock_time.time.return_value = 1672560000.0
        manager.record_dispatch("issue-1")

        # Dispatch for issue-2 (allowed)
        mock_time.time.return_value = 1672560031.0 # After trigger cooldown
        manager.record_dispatch("issue-2")

        # Try to dispatch issue-1 again within its cooldown
        mock_time.time.return_value = 1672560600.0 # 10 minutes after issue-1's last dispatch
        assert manager.can_dispatch("issue-1") is False

        # Try to dispatch issue-1 after its cooldown
        mock_time.time.return_value = 1672560901.0 # 15 minutes and 1 second after issue-1's last dispatch
        assert manager.can_dispatch("issue-1") is True

def test_max_retries(manager):
    with patch("mullet.core.orchestration.team_dispatch.time") as mock_time:
        mock_time.time.return_value = 1672560000.0

        # Dispatch 3 times (allowed)
        manager.record_dispatch("issue-1")
        mock_time.time.return_value += 961.0 # Advance time past cooldowns (16 min + 1 sec)
        manager.record_dispatch("issue-1")
        mock_time.time.return_value += 961.0
        manager.record_dispatch("issue-1")

        assert manager.issue_dispatch_counts["issue-1"] == 3
        assert manager.can_dispatch("issue-1") is False # Max retries reached

        mock_time.time.return_value += 961.0
        assert manager.can_dispatch("issue-1") is False # Still disallowed after cooldown

def test_multiple_issues(manager):
    with patch("mullet.core.orchestration.team_dispatch.time") as mock_time:
        mock_time.time.return_value = 1672560000.0

        # Dispatch issue-1
        manager.record_dispatch("issue-1")
        mock_time.time.return_value += 961.0

        # Dispatch issue-2
        manager.record_dispatch("issue-2")
        mock_time.time.return_value += 961.0

        # Dispatch issue-1 again
        manager.record_dispatch("issue-1")

        assert manager.issue_dispatch_counts["issue-1"] == 2
        assert manager.issue_dispatch_counts["issue-2"] == 1

        # Check cooldowns for both
        mock_time.time.return_value = 1672560000.0 + (961.0*2 + 300.0) # 5 min after last dispatch
        assert manager.can_dispatch("issue-1") is False
        assert manager.can_dispatch("issue-2") is True # Corrected assertion

        mock_time.time.return_value = 1672560000.0 + (961.0*2 + 960.0) # 16 min after last dispatch
        assert manager.can_dispatch("issue-1") is True
        assert manager.can_dispatch("issue-2") is True
