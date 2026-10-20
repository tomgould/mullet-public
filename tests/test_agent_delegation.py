"""Tests for P0-2: API-only delegates get stripped toolsets.

Kimi/Gemini/Haiku-via-API/OpenRouter cannot dispatch filesystem, shell,
git, docker, browser, or subagent-delegation tools. The child runner's
toolset list must exclude those when the delegate is API-only; Anthropic
providers (sonnet/opus via claude_cli or first-party SDK) keep them.
"""

from unittest.mock import MagicMock, patch

from mullet.core.agent.delegation import (
    _API_ONLY_STRIPPED_TOOLSETS,
    delegate_task,
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
    """Return (captured_kwargs_list, patcher) — patcher patches AgentRunner
    inside ``mullet.core.agent.delegation`` so we can inspect what toolsets
    the child was constructed with, without actually running the agent loop.
    """
    captured: list[dict] = []

    def _factory(*args, **kwargs):
        captured.append(kwargs)
        fake = MagicMock()
        fake._delegation_depth = 0
        # run() returns something minimal that delegate_task consumes.
        fake_result = MagicMock()
        fake_result.completed = True
        fake_result.final_response = "done"
        fake_result.tokens_used.total_tokens = 0
        fake.run.return_value = fake_result
        return fake

    # Patch the import inside delegation.py: it does a local import of
    # AgentRunner in the function body, so we patch the source module.
    return captured, patch(
        "mullet.core.agent.runner.AgentRunner", side_effect=_factory
    )


def test_api_only_provider_gets_stripped_toolsets():
    full_toolsets = [
        "terminal", "file", "web", "code", "browser",
        "delegation", "session", "git", "testing", "vision", "docker",
    ]
    parent = _parent_with_toolsets(full_toolsets)
    kimi = _FakeProvider(name="kimi")

    captured, patcher = _capture_runner_kwargs()
    with patcher:
        result = delegate_task(goal="do something", parent=parent, provider_override=kimi)

    assert result.completed is True
    assert captured, "child AgentRunner was never constructed"
    child_kwargs = captured[0]
    child_toolsets = child_kwargs["toolsets"]
    # Every stripped toolset must be gone.
    for stripped in _API_ONLY_STRIPPED_TOOLSETS:
        assert stripped not in child_toolsets, (
            f"{stripped!r} should be stripped for kimi delegate, got {child_toolsets}"
        )
    # Non-local toolsets that ARE compatible with an API-only delegate survive.
    assert "web" in child_toolsets
    assert "code" in child_toolsets
    assert "session" in child_toolsets
    assert "vision" in child_toolsets


def test_gemini_provider_gets_stripped_toolsets():
    parent = _parent_with_toolsets(["file", "terminal", "code", "web"])
    gemini = _FakeProvider(name="gemini")

    captured, patcher = _capture_runner_kwargs()
    with patcher:
        delegate_task(goal="analyse", parent=parent, provider_override=gemini)

    child_toolsets = captured[0]["toolsets"]
    assert "file" not in child_toolsets
    assert "terminal" not in child_toolsets
    assert "code" in child_toolsets
    assert "web" in child_toolsets


def test_anthropic_provider_inherits_full_toolsets():
    # Sonnet/Opus via claude_cli or first-party SDK can dispatch MCP tools
    # server-side, so their child should inherit the parent toolset list
    # untouched.
    full_toolsets = [
        "terminal", "file", "web", "code", "browser",
        "delegation", "session", "git", "testing",
    ]
    parent = _parent_with_toolsets(full_toolsets)
    sonnet = _FakeProvider(name="sonnet")

    captured, patcher = _capture_runner_kwargs()
    with patcher:
        delegate_task(goal="do something", parent=parent, provider_override=sonnet)

    child_toolsets = captured[0]["toolsets"]
    assert child_toolsets == full_toolsets


def test_unknown_provider_inherits_full_toolsets():
    # Unclassified providers keep full toolsets to avoid accidentally
    # stripping capabilities from a new Anthropic-style provider.
    full_toolsets = ["file", "terminal", "code"]
    parent = _parent_with_toolsets(full_toolsets)
    unknown = _FakeProvider(name="brand-new-provider")

    captured, patcher = _capture_runner_kwargs()
    with patcher:
        delegate_task(goal="x", parent=parent, provider_override=unknown)

    assert captured[0]["toolsets"] == full_toolsets


def test_parent_provider_name_drives_stripping_when_no_override():
    # When no provider_override is given, the parent's provider is used.
    # If the parent itself is running on kimi, the child also inherits
    # kimi -> toolsets should be stripped.
    parent = _parent_with_toolsets(["file", "terminal", "code", "web"])
    parent.provider = _FakeProvider(name="kimi")

    captured, patcher = _capture_runner_kwargs()
    with patcher:
        delegate_task(goal="x", parent=parent)

    child_toolsets = captured[0]["toolsets"]
    assert "file" not in child_toolsets
    assert "terminal" not in child_toolsets
    assert "code" in child_toolsets
    assert "web" in child_toolsets


def test_explicit_toolsets_argument_also_stripped_for_api_only():
    # If the caller passes toolsets= directly, API-only stripping still runs —
    # otherwise the bug re-appears when a caller passes the full list.
    parent = _parent_with_toolsets(["test"])
    kimi = _FakeProvider(name="kimi")

    captured, patcher = _capture_runner_kwargs()
    with patcher:
        delegate_task(
            goal="x",
            toolsets=["file", "terminal", "code"],
            parent=parent,
            provider_override=kimi,
        )

    child_toolsets = captured[0]["toolsets"]
    assert "file" not in child_toolsets
    assert "terminal" not in child_toolsets
    assert "code" in child_toolsets
