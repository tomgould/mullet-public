"""Gap tests for mullet/core/agent/delegation.py — target ≥95% coverage."""

from unittest.mock import MagicMock, patch

from mullet.core.agent.delegation import (
    MAX_DEPTH,
    _build_child_system_prompt,
    delegate_batch,
    delegate_task,
    delegate_with_fallback,
)
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class _FakeProvider(Provider):
    def __init__(self, name: str):
        self.name = name
        self._calls = 0

    def create(self, messages, tools=None, **kwargs) -> LLMResponse:
        self._calls += 1
        return LLMResponse(
            content="done",
            tool_calls=None,
            finish_reason="stop",
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            model="fake-model",
            provider=self.name,
        )


def _parent_with_toolsets(toolsets: list[str]) -> AgentRunner:
    registry = ToolRegistry()
    registry.register(
        name="noop",
        func=lambda **_: "ok",
        schema={"description": "noop"},
        toolset="file",
    )
    return AgentRunner(
        provider=_FakeProvider(name="parent"),
        tool_registry=registry,
        toolsets=toolsets,
        system_prompt="parent",
        max_iterations=5,
    )


def _capture_runner_kwargs():
    captured: list[dict] = []

    def _factory(*args, **kwargs):
        captured.append(kwargs)
        fake = MagicMock()
        fake._delegation_depth = 0
        fake_result = MagicMock()
        fake_result.completed = True
        fake_result.final_response = "done"
        fake_result.tokens_used.total_tokens = 0
        fake.run.return_value = fake_result
        return fake

    return captured, patch("mullet.core.agent.runner.AgentRunner", side_effect=_factory)


# ---------------------------------------------------------------------------
# Line 128 — max depth reached
# ---------------------------------------------------------------------------

def test_max_depth_reached():
    parent = _parent_with_toolsets(["file"])
    result = delegate_task(goal="x", parent=parent, _depth=MAX_DEPTH)
    assert result.completed is False
    assert f"Max delegation depth ({MAX_DEPTH}) reached" in result.error


# ---------------------------------------------------------------------------
# Lines 140-141 — recommend_tier exception handled gracefully
# ---------------------------------------------------------------------------

def test_recommend_tier_exception_handled():
    parent = _parent_with_toolsets(["file"])
    with patch("mullet.core.router.complexity.recommend_tier", side_effect=Exception("tier boom")):
        captured, patcher = _capture_runner_kwargs()
        with patcher:
            result = delegate_task(goal="x", parent=parent)
    assert result.completed is True


# ---------------------------------------------------------------------------
# Lines 157-158 + 330-331 — profile detected in goal
# ---------------------------------------------------------------------------

def test_profile_in_goal_sets_child_toolsets():
    """If goal contains 'analyst', the analyst profile's default_toolsets are used."""
    parent = _parent_with_toolsets(["file", "terminal", "code"])
    captured, patcher = _capture_runner_kwargs()
    with patcher:
        delegate_task(goal="analyst: review the codebase", parent=parent)
    assert captured[0]["toolsets"] == ["terminal", "file", "web"]


def test_build_child_system_prompt_with_profile():
    prompt = _build_child_system_prompt(goal="analyst: review code", context="")
    assert "analyst agent" in prompt.lower()


# ---------------------------------------------------------------------------
# Lines 201-204 — exception during child.run()
# ---------------------------------------------------------------------------

def test_delegate_task_exception_in_child_run():
    parent = _parent_with_toolsets(["file"])

    def _factory(*args, **kwargs):
        fake = MagicMock()
        fake.run.side_effect = Exception("child boom")
        return fake

    with patch("mullet.core.agent.runner.AgentRunner", side_effect=_factory):
        result = delegate_task(goal="x", parent=parent)
    assert result.completed is False
    assert "child boom" in result.error


# ---------------------------------------------------------------------------
# Line 290 — local provider logs concurrency limit
# ---------------------------------------------------------------------------

def test_delegate_batch_local_provider_logs():
    parent = _parent_with_toolsets(["file"])
    parent.provider = _FakeProvider(name="local")

    def _factory(*args, **kwargs):
        fake = MagicMock()
        fake.run.return_value = MagicMock(
            completed=True, final_response="done", tokens_used=MagicMock(total_tokens=0)
        )
        return fake

    with patch("mullet.core.agent.runner.AgentRunner", side_effect=_factory):
        with patch("mullet.core.agent.delegation.logger") as mock_logger:
            delegate_batch([{"goal": "task1"}], parent=parent)
    mock_logger.info.assert_called_with("Local provider: limiting batch concurrency to 1")


# ---------------------------------------------------------------------------
# Lines 313-314 — future.exception path in delegate_batch
# ---------------------------------------------------------------------------

def test_delegate_batch_future_exception():
    parent = _parent_with_toolsets(["file"])

    mock_future = MagicMock()
    mock_future.result.side_effect = Exception("future boom")

    with patch("mullet.core.agent.delegation.ThreadPoolExecutor") as mock_executor_cls, \
         patch("mullet.core.agent.delegation.as_completed") as mock_as_completed:
        mock_executor = MagicMock()
        mock_executor_cls.return_value.__enter__.return_value = mock_executor
        mock_executor.submit.return_value = mock_future
        mock_as_completed.return_value = [mock_future]

        results = delegate_batch([{"goal": "task1"}], parent=parent)

    assert len(results) == 1
    assert results[0].completed is False
    assert "future boom" in results[0].error


# ---------------------------------------------------------------------------
# delegate_with_fallback — empty provider_chain falls back to delegate_task
# ---------------------------------------------------------------------------

def test_delegate_with_fallback_empty_chain():
    parent = _parent_with_toolsets(["file"])
    captured, patcher = _capture_runner_kwargs()
    with patcher:
        result = delegate_with_fallback(goal="x", parent=parent, provider_chain=[])
    assert result.completed is True


# ---------------------------------------------------------------------------
# delegate_batch — empty tasks returns empty list
# ---------------------------------------------------------------------------

def test_delegate_batch_empty_tasks():
    assert delegate_batch([], parent=None) == []


# ---------------------------------------------------------------------------
# Line 125 — parent is None in delegate_task
# ---------------------------------------------------------------------------

def test_delegate_task_no_parent():
    result = delegate_task(goal="x", parent=None)
    assert result.completed is False
    assert "No parent agent provided" in result.error


# ---------------------------------------------------------------------------
# Line 148 — allowed_providers explicitly provided (non-None)
# ---------------------------------------------------------------------------

def test_delegate_task_explicit_allowed_providers():
    parent = _parent_with_toolsets(["file"])
    captured, patcher = _capture_runner_kwargs()
    with patcher:
        delegate_task(goal="x", parent=parent, allowed_providers=frozenset({"kimi"}))
    assert captured[0]["allowed_providers"] == frozenset({"kimi"})


# ---------------------------------------------------------------------------
# Lines 169-176 — toolset stripping logs debug when list changes
# ---------------------------------------------------------------------------

def test_api_only_stripping_logs_debug():
    parent = _parent_with_toolsets(["file", "terminal", "code"])
    kimi = _FakeProvider(name="kimi")
    captured, patcher = _capture_runner_kwargs()
    with patcher:
        with patch("mullet.core.agent.delegation.logger") as mock_logger:
            delegate_task(goal="x", parent=parent, provider_override=kimi)
    assert mock_logger.debug.call_count >= 1


# ---------------------------------------------------------------------------
# Lines 236-253 — delegate_with_fallback all providers exhausted
# ---------------------------------------------------------------------------

def test_delegate_with_fallback_first_succeeds():
    parent = _parent_with_toolsets(["file"])
    provider_a = _FakeProvider(name="provider_a")
    provider_b = _FakeProvider(name="provider_b")

    call_count = 0

    def _factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        fake = MagicMock()
        fake.run.return_value = MagicMock(completed=True, final_response="ok", tokens_used=MagicMock(total_tokens=0))
        return fake

    with patch("mullet.core.agent.runner.AgentRunner", side_effect=_factory):
        result = delegate_with_fallback(goal="x", parent=parent, provider_chain=[provider_a, provider_b])

    assert result.completed is True
    assert call_count == 1  # only first provider was tried


def test_delegate_with_fallback_all_exhausted():
    parent = _parent_with_toolsets(["file"])
    provider_a = _FakeProvider(name="provider_a")
    provider_b = _FakeProvider(name="provider_b")

    call_count = 0

    def _factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        fake = MagicMock()
        fake.run.side_effect = Exception(f"fail {call_count}")
        return fake

    with patch("mullet.core.agent.runner.AgentRunner", side_effect=_factory):
        result = delegate_with_fallback(goal="x", parent=parent, provider_chain=[provider_a, provider_b])

    assert result.completed is False
    assert "All providers exhausted" in result.error
    assert "provider_a: fail 1" in result.error
    assert "provider_b: fail 2" in result.error


# ---------------------------------------------------------------------------
# Line 276 — parent is None in delegate_batch
# ---------------------------------------------------------------------------

def test_delegate_batch_no_parent():
    results = delegate_batch([{"goal": "task1"}], parent=None)
    assert len(results) == 1
    assert results[0].completed is False
    assert "No parent agent provided" in results[0].error


# ---------------------------------------------------------------------------
# Line 339 — _build_child_system_prompt with context
# ---------------------------------------------------------------------------

def test_build_child_system_prompt_with_context():
    prompt = _build_child_system_prompt(goal="do thing", context="extra info")
    assert "CONTEXT: extra info" in prompt


# ---------------------------------------------------------------------------
# Lines 95-96 — _log_spend exception path (best-effort, never crashes)
# ---------------------------------------------------------------------------

def test_log_spend_exception_best_effort():
    parent = _parent_with_toolsets(["file"])
    with patch("sqlite3.connect", side_effect=Exception("db boom")):
        captured, patcher = _capture_runner_kwargs()
        with patcher:
            result = delegate_task(goal="x", parent=parent)
    assert result.completed is True


# ---------------------------------------------------------------------------
# Lines 247-250 — delegate_with_fallback catches exception from delegate_task
# ---------------------------------------------------------------------------

def test_delegate_with_fallback_delegate_task_raises():
    """When delegate_task itself raises (e.g., AgentRunner construction fails),
    delegate_with_fallback should catch it and record the error."""
    parent = _parent_with_toolsets(["file"])
    provider_a = _FakeProvider(name="provider_a")

    with patch("mullet.core.agent.runner.AgentRunner", side_effect=Exception("constructor boom")):
        result = delegate_with_fallback(goal="x", parent=parent, provider_chain=[provider_a])

    assert result.completed is False
    assert "All providers exhausted" in result.error
    assert "provider_a: constructor boom" in result.error
