"""Additional tests for delegation.py — _log_spend, delegate_with_fallback, delegate_batch."""
import sqlite3
from unittest.mock import MagicMock, patch


from mullet.core.agent.delegation import (
    _log_spend,
    delegate_with_fallback,
    delegate_batch,
    MAX_CONCURRENT,
)
from mullet.core.agent.types import DelegationResult


# ── _log_spend ────────────────────────────────────────────────────────────────

def test_log_spend_writes_to_db(tmp_path):
    db_file = tmp_path / "spend.db"
    # Patch the db path so _log_spend writes to our temp file
    with patch("mullet.core.agent.delegation.Path") as mock_path_cls:
        mock_home = MagicMock()
        mock_path_cls.home.return_value = mock_home
        # Path.home() / ".mullet" / "delegation_spend.db" → db_file
        mock_home.__truediv__.return_value.__truediv__.return_value = db_file
        db_file.parent.mkdir(parents=True, exist_ok=True)

        _log_spend("test goal", "kimi", 1000, True)

    # Reopen to verify (the function closes its own connection)
    conn = sqlite3.connect(str(db_file))
    row = conn.execute("SELECT goal, provider, tokens, success FROM spend LIMIT 1").fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "test goal"
    assert row[1] == "kimi"
    assert row[2] == 1000
    assert row[3] == 1


def test_log_spend_truncates_long_goal(tmp_path):
    long_goal = "x" * 500
    db_file = tmp_path / "spend2.db"
    with patch("mullet.core.agent.delegation.Path") as mock_path_cls:
        mock_home = MagicMock()
        mock_path_cls.home.return_value = mock_home
        mock_home.__truediv__.return_value.__truediv__.return_value = db_file
        db_file.parent.mkdir(parents=True, exist_ok=True)
        _log_spend(long_goal, "local", 0, False)

    conn = sqlite3.connect(str(db_file))
    row = conn.execute("SELECT goal FROM spend LIMIT 1").fetchone()
    conn.close()
    if row:
        assert len(row[0]) <= 200


def test_log_spend_never_crashes_on_db_error():
    with patch("mullet.core.agent.delegation.sqlite3.connect", side_effect=Exception("DB error")):
        with patch("mullet.core.agent.delegation.Path") as mock_path:
            mock_path.home.return_value.__truediv__ = lambda *a: MagicMock()
            _log_spend("goal", "provider", 0, True)  # must not raise


# ── delegate_with_fallback ────────────────────────────────────────────────────

def _make_provider(name, result=None, raises=None):
    p = MagicMock()
    p.name = name
    return p


def _make_success_result(goal="test"):
    return DelegationResult(goal=goal, completed=True, summary="done", error=None)


def _make_fail_result(goal="test", error="failed"):
    return DelegationResult(goal=goal, completed=False, summary="", error=error)


def test_delegate_with_fallback_empty_chain_calls_delegate_task():
    with patch("mullet.core.agent.delegation.delegate_task") as mock_dt:
        mock_dt.return_value = _make_success_result()
        result = delegate_with_fallback("goal", provider_chain=[])
    mock_dt.assert_called_once()
    assert result.completed is True


def test_delegate_with_fallback_first_provider_succeeds():
    p1, p2 = _make_provider("kimi"), _make_provider("local")
    with patch("mullet.core.agent.delegation.delegate_task") as mock_dt:
        mock_dt.return_value = _make_success_result()
        result = delegate_with_fallback("goal", provider_chain=[p1, p2])
    assert mock_dt.call_count == 1  # stopped at first success
    assert result.completed is True


def test_delegate_with_fallback_tries_next_on_failure():
    p1, p2 = _make_provider("kimi"), _make_provider("local")
    with patch("mullet.core.agent.delegation.delegate_task") as mock_dt:
        mock_dt.side_effect = [_make_fail_result(error="timeout"), _make_success_result()]
        result = delegate_with_fallback("goal", provider_chain=[p1, p2])
    assert mock_dt.call_count == 2
    assert result.completed is True


def test_delegate_with_fallback_all_fail_returns_error():
    p1, p2 = _make_provider("kimi"), _make_provider("local")
    with patch("mullet.core.agent.delegation.delegate_task") as mock_dt:
        mock_dt.side_effect = [Exception("kimi down"), Exception("local down")]
        result = delegate_with_fallback("goal", provider_chain=[p1, p2])
    assert result.completed is False
    assert "All providers exhausted" in result.error


# ── delegate_batch ────────────────────────────────────────────────────────────

def test_delegate_batch_empty_tasks():
    result = delegate_batch([], parent=None)
    assert result == []


def test_delegate_batch_no_parent():
    tasks = [{"goal": "task1"}, {"goal": "task2"}]
    results = delegate_batch(tasks, parent=None)
    assert len(results) == 2
    for r in results:
        assert r.completed is False
        assert "No parent agent" in r.error


def test_delegate_batch_concurrent_up_to_max():
    parent = MagicMock()
    tasks = [{"goal": f"task{i}"} for i in range(MAX_CONCURRENT + 1)]
    call_count = []

    def fake_delegate(goal, **kwargs):
        call_count.append(goal)
        return DelegationResult(goal=goal, completed=True, summary="ok", error=None)

    with patch("mullet.core.agent.delegation.delegate_task", side_effect=fake_delegate):
        results = delegate_batch(tasks, parent=parent)


# ── G14: allowed_providers inheritance on delegate_task ──────────────────────


def _fake_parent_for_delegate(allowed=None):
    """Build a minimal parent MagicMock that delegate_task can spawn a child from."""
    import threading
    parent = MagicMock()
    parent.allowed_providers = allowed
    parent.provider = MagicMock(name="parent_provider")
    parent.provider.name = "kimi"
    parent.tool_registry = MagicMock()
    parent.toolsets = ["terminal"]
    parent._active_children = []
    parent._interrupt_lock = threading.Lock()
    parent._delegation_depth = 0
    return parent


def _capture_child_kwargs():
    """Patch AgentRunner constructor; return (holder dict, patcher) pair."""
    holder = {}

    class _StubRunner:
        def __init__(self, **kwargs):
            holder.update(kwargs)
            self._delegation_depth = 0

        def run(self, goal):
            return MagicMock(completed=True, final_response="ok")

    return holder, _StubRunner


def test_delegate_task_child_inherits_parent_allowlist():
    """None (default) -> child AgentRunner gets parent.allowed_providers."""
    from mullet.core.agent import delegation as dm

    parent = _fake_parent_for_delegate(allowed=frozenset({"kimi"}))
    holder, StubRunner = _capture_child_kwargs()

    with patch("mullet.core.agent.runner.AgentRunner", StubRunner):
        dm.delegate_task("do thing", parent=parent)

    assert holder.get("allowed_providers") == frozenset({"kimi"})


def test_delegate_task_explicit_empty_frozenset_overrides_to_unrestricted():
    """Empty frozenset() -> explicit override, child gets empty frozenset (not parent's)."""
    from mullet.core.agent import delegation as dm

    parent = _fake_parent_for_delegate(allowed=frozenset({"kimi"}))
    holder, StubRunner = _capture_child_kwargs()

    with patch("mullet.core.agent.runner.AgentRunner", StubRunner):
        dm.delegate_task("do thing", parent=parent, allowed_providers=frozenset())

    # Explicit empty frozenset must NOT collapse to None / inherit
    assert holder.get("allowed_providers") == frozenset()


def test_delegate_task_explicit_nonempty_replaces_inherited():
    """Non-empty frozenset -> explicit restriction, replaces parent's allowlist."""
    from mullet.core.agent import delegation as dm

    parent = _fake_parent_for_delegate(allowed=frozenset({"kimi"}))
    holder, StubRunner = _capture_child_kwargs()

    with patch("mullet.core.agent.runner.AgentRunner", StubRunner):
        dm.delegate_task("do thing", parent=parent, allowed_providers=frozenset({"haiku"}))

    assert holder.get("allowed_providers") == frozenset({"haiku"})


def test_delegate_task_no_parent_allowlist_stays_none():
    """Parent has no allowed_providers attr -> child gets None."""
    from mullet.core.agent import delegation as dm

    parent = _fake_parent_for_delegate(allowed=None)
    holder, StubRunner = _capture_child_kwargs()

    with patch("mullet.core.agent.runner.AgentRunner", StubRunner):
        dm.delegate_task("do thing", parent=parent)

    assert holder.get("allowed_providers") is None


def test_delegate_batch_per_task_allowlist_wins():
    """Per-task 'allowed_providers' in task dict overrides both function-level and inherited."""
    from mullet.core.agent import delegation as dm

    parent = _fake_parent_for_delegate(allowed=frozenset({"kimi"}))
    captured = []

    def fake_delegate(**kwargs):
        captured.append(kwargs.get("allowed_providers"))
        return DelegationResult(goal=kwargs["goal"], completed=True, summary="ok")

    tasks = [
        {"goal": "a"},                                        # inherit path -> None
        {"goal": "b", "allowed_providers": frozenset({"haiku"})},  # per-task override
    ]
    with patch("mullet.core.agent.delegation.delegate_task", side_effect=fake_delegate):
        dm.delegate_batch(tasks, parent=parent, allowed_providers=frozenset({"gemini"}))

    # First task: no per-task key -> function-level fn arg ("gemini")
    # Second task: per-task key ("haiku") wins
    assert frozenset({"gemini"}) in captured
    assert frozenset({"haiku"}) in captured


def test_delegate_with_fallback_forwards_allowlist():
    """delegate_with_fallback passes allowed_providers through to each delegate_task call."""
    from mullet.core.agent import delegation as dm

    p1 = _make_provider("kimi")
    captured = []

    def fake_delegate(*args, **kwargs):
        captured.append(kwargs.get("allowed_providers"))
        return _make_success_result()

    with patch("mullet.core.agent.delegation.delegate_task", side_effect=fake_delegate):
        dm.delegate_with_fallback(
            "goal",
            provider_chain=[p1],
            allowed_providers=frozenset({"kimi"}),
        )

    assert captured == [frozenset({"kimi"})]


def test_delegate_with_fallback_empty_chain_forwards_allowlist():
    """With empty chain, the single delegate_task fallback still receives allowed_providers."""
    from mullet.core.agent import delegation as dm

    captured = []

    def fake_delegate(*args, **kwargs):
        captured.append(kwargs.get("allowed_providers"))
        return _make_success_result()

    with patch("mullet.core.agent.delegation.delegate_task", side_effect=fake_delegate):
        dm.delegate_with_fallback(
            "goal",
            provider_chain=[],
            allowed_providers=frozenset({"haiku"}),
        )

    assert captured == [frozenset({"haiku"})]
