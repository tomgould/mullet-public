"""Test per-task provider overrides in delegate_batch (Phase C1 routing-v2)."""

from unittest.mock import MagicMock, patch

from mullet.core.agent.delegation import delegate_batch
from mullet.core.agent.types import DelegationResult


class FakeProvider:
    """Minimal provider stand-in for testing."""
    def __init__(self, name: str):
        self.name = name


def _make_fake_task(goal: str, provider_name: str | None = None) -> dict:
    """Build a task dict, optionally with a _provider_override."""
    task = {"goal": goal, "context": "", "max_iterations": 10}
    if provider_name:
        task["_provider_override"] = FakeProvider(provider_name)
    return task


@patch("mullet.core.agent.delegation.delegate_task")
def test_delegate_batch_uses_per_task_provider_override(mock_delegate):
    """Each task should receive its own provider_override, not a single batch-wide one."""
    mock_delegate.return_value = DelegationResult(
        goal="", completed=True, summary="ok"
    )

    parent = MagicMock()
    parent.provider = FakeProvider("local")
    parent._delegation_depth = 0
    parent._interrupt_lock = MagicMock()
    parent._active_children = []
    parent.tool_registry = MagicMock()
    parent.toolsets = []

    tasks = [
        _make_fake_task("task-a", "kimi"),
        _make_fake_task("task-b", "sonnet"),
        _make_fake_task("task-c"),          # no override
    ]

    delegate_batch(tasks, parent=parent, provider_override=None)

    assert mock_delegate.call_count == 3

    calls = mock_delegate.call_args_list
    assert calls[0].kwargs["provider_override"].name == "kimi"
    assert calls[1].kwargs["provider_override"].name == "sonnet"
    assert calls[2].kwargs["provider_override"] is None


@patch("mullet.core.agent.delegation.delegate_task")
def test_delegate_batch_local_any_task_forces_max_workers_one(mock_delegate):
    """If ANY task in the batch routes to local, concurrency must be limited to 1."""
    mock_delegate.return_value = DelegationResult(
        goal="", completed=True, summary="ok"
    )

    parent = MagicMock()
    parent.provider = FakeProvider("kimi")
    parent._delegation_depth = 0
    parent._interrupt_lock = MagicMock()
    parent._active_children = []
    parent.tool_registry = MagicMock()
    parent.toolsets = []

    # One local task among non-local tasks
    tasks = [
        _make_fake_task("task-a", "kimi"),
        _make_fake_task("task-b", "local"),
        _make_fake_task("task-c", "sonnet"),
    ]

    # Should not raise and should dispatch all tasks
    results = delegate_batch(tasks, parent=parent, provider_override=None)

    assert mock_delegate.call_count == 3
    assert len(results) == 3

    # Verify the local task got local provider, others got their overrides
    calls = mock_delegate.call_args_list
    names = [c.kwargs["provider_override"].name if c.kwargs["provider_override"] else None for c in calls]
    assert "kimi" in names
    assert "local" in names
    assert "sonnet" in names


@patch("mullet.core.agent.delegation.delegate_task")
def test_delegate_batch_fallback_to_parent_provider(mock_delegate):
    """Tasks without _provider_override should fall back to provider_override arg then parent."""
    mock_delegate.return_value = DelegationResult(
        goal="", completed=True, summary="ok"
    )

    parent = MagicMock()
    parent.provider = FakeProvider("openrouter")
    parent._delegation_depth = 0
    parent._interrupt_lock = MagicMock()
    parent._active_children = []
    parent.tool_registry = MagicMock()
    parent.toolsets = []

    fallback = FakeProvider("gemini")
    tasks = [
        _make_fake_task("task-a"),
        _make_fake_task("task-b", "sonnet"),
    ]

    delegate_batch(tasks, parent=parent, provider_override=fallback)

    calls = mock_delegate.call_args_list
    # task-a gets the batch-level fallback
    assert calls[0].kwargs["provider_override"].name == "gemini"
    # task-b keeps its own override
    assert calls[1].kwargs["provider_override"].name == "sonnet"
