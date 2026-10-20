"""Tests for mullet.harness.coordinator.swarm."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mullet.harness.coordinator.swarm import Coordinator, SwarmSubtask, _extract_json_strings
from mullet.harness.coordinator.concurrency import ConcurrencyManager, ConcurrencyLimits

pytestmark = pytest.mark.harness


class _FakeEngine:
    """Records calls and returns fixed responses."""

    def __init__(self, responses: list[dict] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.responses = responses or []
        self._idx = 0

    async def arun(self, prompt: str, **kwargs: Any) -> dict:
        self.calls.append((prompt, kwargs))
        if self._idx < len(self.responses):
            resp = self.responses[self._idx]
            self._idx += 1
            return resp
        return {"response": "ok"}


class TestCoordinatorPlan:
    def test_plan_with_files_returns_one_subtask_per_file(self):
        coordinator = Coordinator()
        subtasks = asyncio.run(
            coordinator.plan(
                "refactor to Pydantic v2",
                files=["a.py", "b.py", "c.py"],
                n_agents=5,
            )
        )
        assert len(subtasks) == 3
        assert subtasks[0].id == "int-1"
        assert subtasks[0].files == ["a.py"]
        assert "a.py" in subtasks[0].description

    def test_plan_respects_n_agents_cap(self):
        coordinator = Coordinator()
        subtasks = asyncio.run(
            coordinator.plan(
                "refactor",
                files=["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"],
                n_agents=3,
            )
        )
        assert len(subtasks) == 3  # capped by n_agents


class TestCoordinatorExecute:
    def test_execute_runs_subtasks_in_parallel(self):
        call_order = []
        call_times = []
        import time

        class _SlowEngine:
            async def arun(self, prompt: str, **kwargs: Any) -> dict:
                call_order.append(prompt)
                call_times.append(time.monotonic())
                await asyncio.sleep(0.05)
                return {"response": f"done: {prompt[:20]}"}

        coordinator = Coordinator(engine_factory=_SlowEngine)
        subtasks = [
            SwarmSubtask(id="int-1", description="task one", use_worktree=False),
            SwarmSubtask(id="int-2", description="task two", use_worktree=False),
        ]
        result = asyncio.run(coordinator.execute(subtasks))
        assert len(result.successes) == 2
        assert set(result.successes) == {"int-1", "int-2"}
        # Parallel execution: total time < sum of individual sleeps
        assert len(call_times) == 2
        assert call_times[1] - call_times[0] < 0.09  # should overlap

    def test_concurrency_limits_respected(self):
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        class _CountingEngine:
            async def arun(self, prompt: str, **kwargs: Any) -> dict:
                nonlocal active, max_active
                async with lock:
                    active += 1
                    max_active = max(max_active, active)
                await asyncio.sleep(0.05)
                async with lock:
                    active -= 1
                return {"response": "ok"}

        concurrency = ConcurrencyManager(
            ConcurrencyLimits(default_concurrency=1, max_total_agents=1)
        )
        coordinator = Coordinator(
            engine_factory=_CountingEngine,
            concurrency=concurrency,
        )
        subtasks = [
            SwarmSubtask(id=f"int-{i}", description=f"task {i}", use_worktree=False)
            for i in range(3)
        ]
        result = asyncio.run(coordinator.execute(subtasks))
        assert len(result.successes) == 3
        # With limit=1, max concurrent should be 1
        assert max_active == 1

    def test_failed_subtask_reported_in_failures_list(self):
        class _FailingEngine:
            async def arun(self, prompt: str, **kwargs: Any) -> dict:
                raise RuntimeError("boom")

        coordinator = Coordinator(engine_factory=_FailingEngine)
        subtasks = [
            SwarmSubtask(id="int-1", description="task one", use_worktree=False),
        ]
        result = asyncio.run(coordinator.execute(subtasks))
        assert result.successes == []
        assert len(result.failures) == 1
        assert result.failures[0][0] == "int-1"
        assert "boom" in result.failures[0][1]


class TestExtractJsonStrings:
    def test_extracts_json_array(self):
        raw = 'Here is the plan:\n["task a", "task b", "task c"]\nDone.'
        assert _extract_json_strings(raw) == ["task a", "task b", "task c"]

    def test_fallback_line_split(self):
        raw = "- task a\n- task b\n- task c"
        result = _extract_json_strings(raw)
        assert "task a" in result
        assert "task b" in result

    def test_empty_returns_empty(self):
        assert _extract_json_strings("") == []
