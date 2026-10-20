"""Tests for the AgentRunner core loop."""

import json
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class SimpleProvider(Provider):
    """Provider that returns predefined responses for testing."""

    def __init__(self, responses: list[LLMResponse]):
        self.name = "test"
        self._responses = responses
        self._call_idx = 0

    def create(self, messages, tools=None, **kwargs) -> LLMResponse:
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return resp


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="add",
        func=lambda a, b, **_: json.dumps({"result": int(a) + int(b)}),
        schema={
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
        toolset="math",
    )
    return registry


def test_simple_response():
    """Agent with no tool calls returns immediately."""
    provider = SimpleProvider([
        LLMResponse(content="Hello!", usage=TokenUsage(total_tokens=10)),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"])

    result = agent.run("say hello")
    assert result.completed is True
    assert result.final_response == "Hello!"
    assert result.iterations_used == 1
    assert result.tool_calls_made == 0


def test_tool_call_loop():
    """Agent calls a tool, gets result, then responds."""
    provider = SimpleProvider([
        # First response: tool call
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'},
            }],
            usage=TokenUsage(total_tokens=20),
        ),
        # Second response: final answer after seeing tool result
        LLMResponse(
            content="The sum is 7.",
            usage=TokenUsage(total_tokens=15),
        ),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"])

    result = agent.run("what is 3 + 4?")
    assert result.completed is True
    assert "7" in result.final_response
    assert result.tool_calls_made == 1
    assert result.iterations_used == 2
    assert result.tokens_used.total_tokens == 35


def test_budget_exhaustion():
    """Agent stops when iteration budget runs out."""
    # Provider always requests another tool call — will hit budget limit
    tool_call_response = LLMResponse(
        content="",
        tool_calls=[{
            "id": "call_n",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1, "b": 1}'},
        }],
        usage=TokenUsage(total_tokens=5),
    )
    provider = SimpleProvider([tool_call_response] * 100)
    registry = _make_registry()
    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=["math"],
        max_iterations=3,
    )

    result = agent.run("keep adding")
    assert result.completed is False
    assert "budget" in result.final_response.lower()
    assert result.iterations_used == 3


def test_interrupt():
    """Agent stops when interrupted."""
    tool_call_response = LLMResponse(
        content="",
        tool_calls=[{
            "id": "call_n",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1, "b": 1}'},
        }],
        usage=TokenUsage(total_tokens=5),
    )
    provider = SimpleProvider([tool_call_response] * 100)
    registry = _make_registry()
    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=["math"],
        max_iterations=50,
    )

    # Interrupt before first iteration
    agent.interrupt("Stop now")
    result = agent.run("keep going forever")
    assert result.completed is False
    assert "Stop now" in result.final_response


def test_system_prompt():
    """System prompt is included in messages."""
    provider = SimpleProvider([
        LLMResponse(content="ok", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        system_prompt="You are a test agent.",
    )

    result = agent.run("hello")
    # The result preserves the full message array sent to the provider
    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == "You are a test agent."


def test_conversation_history():
    """Previous conversation history is preserved."""
    provider = SimpleProvider([
        LLMResponse(content="I remember!", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry)

    history = [
        {"role": "user", "content": "my name is Tom"},
        {"role": "assistant", "content": "Hi Tom!"},
    ]
    result = agent.run("what's my name?", conversation_history=history)
    assert result.completed is True
    # System + history (2 msgs) + new user msg = 3, then assistant response = 4
    assert len(result.messages) == 4


def test_on_tool_call_callback():
    """on_tool_call callback is invoked for each tool call."""
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }],
            usage=TokenUsage(),
        ),
        LLMResponse(content="3", usage=TokenUsage()),
    ])
    registry = _make_registry()

    tool_calls_seen = []
    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=["math"],
        on_tool_call=lambda name, preview: tool_calls_seen.append((name, preview)),
    )

    agent.run("add 1 and 2")
    assert len(tool_calls_seen) == 1
    assert tool_calls_seen[0][0] == "add"


def test_provider_fallback():
    """Agent falls back to secondary provider on failure."""
    class FailingProvider(Provider):
        name = "failing"
        def create(self, messages, **kwargs):
            raise ConnectionError("primary down")

    class BackupProvider(Provider):
        name = "backup"
        def create(self, messages, **kwargs):
            return LLMResponse(content="Backup worked!", usage=TokenUsage(total_tokens=10))

    registry = _make_registry()
    agent = AgentRunner(
        provider=FailingProvider(),
        tool_registry=registry,
        toolsets=["math"],
        fallback_providers=[BackupProvider()],
    )

    result = agent.run("hello")
    assert result.completed is True
    assert result.final_response == "Backup worked!"
    assert agent.provider.name == "backup"  # Switched to fallback


def test_provider_restoration_on_next_turn():
    """Primary provider is restored at start of each turn."""
    primary = SimpleProvider([LLMResponse(content="primary", usage=TokenUsage())])
    backup = SimpleProvider([LLMResponse(content="backup", usage=TokenUsage())])

    registry = _make_registry()
    agent = AgentRunner(
        provider=primary,
        tool_registry=registry,
        fallback_providers=[backup],
    )

    # Simulate a fallback
    agent.provider = backup
    agent._fallback_activated = True

    # On next run, primary should be restored
    agent.run("hello")
    assert agent._primary_provider.name == "test"  # primary was restored


def test_prefill_messages():
    """Prefill messages are injected between system prompt and history."""
    captured_messages = []

    class CapturingProvider(Provider):
        name = "capture"
        def create(self, messages, **kwargs):
            captured_messages.extend(messages)
            return LLMResponse(content="ok", usage=TokenUsage())

    registry = _make_registry()
    prefill = [
        {"role": "user", "content": "example input"},
        {"role": "assistant", "content": "example output"},
    ]
    agent = AgentRunner(
        provider=CapturingProvider(),
        tool_registry=registry,
        system_prompt="You are helpful.",
        prefill_messages=prefill,
    )
    agent.run("hello")

    # Messages should be: system, prefill[0], prefill[1], user
    assert captured_messages[0]["role"] == "system"
    assert captured_messages[1]["content"] == "example input"
    assert captured_messages[2]["content"] == "example output"
    assert captured_messages[3]["content"] == "hello"


def test_prefill_messages_empty():
    """Empty prefill doesn't break anything."""
    provider = SimpleProvider([LLMResponse(content="ok", usage=TokenUsage())])
    registry = _make_registry()
    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        prefill_messages=[],
    )
    result = agent.run("hello")
    assert result.completed is True


# ---------------------------------------------------------------------------
# G23: AgentRunner._try_fallback respects allowed_providers
# ---------------------------------------------------------------------------


class _NamedProvider(Provider):
    """Minimal Provider stub with a configurable .name (for fallback tests)."""

    def __init__(self, name: str):
        self.name = name

    def create(self, messages, tools=None, **kwargs):
        return LLMResponse(content="", usage=TokenUsage())


def test_try_fallback_skips_out_of_allowlist():
    """Fallbacks not in allowed_providers are skipped; returns False when none match."""
    primary = _NamedProvider("kimi")
    fallbacks = [_NamedProvider("haiku"), _NamedProvider("sonnet")]
    agent = AgentRunner(
        provider=primary,
        tool_registry=ToolRegistry(),
        toolsets=[],
        allowed_providers=frozenset({"kimi"}),
    )
    agent._fallback_providers = fallbacks

    assert agent._try_fallback(Exception("boom")) is False
    assert agent.provider is primary
    assert agent._fallback_activated is False


def test_try_fallback_uses_in_allowlist_fallback():
    """First in-allowlist fallback is activated; out-of-allowlist entries skipped."""
    primary = _NamedProvider("kimi")
    fallbacks = [_NamedProvider("gemini"), _NamedProvider("haiku")]
    agent = AgentRunner(
        provider=primary,
        tool_registry=ToolRegistry(),
        toolsets=[],
        allowed_providers=frozenset({"kimi", "haiku"}),
    )
    agent._fallback_providers = fallbacks

    assert agent._try_fallback(Exception("boom")) is True
    assert agent.provider.name == "haiku"
    assert agent._fallback_activated is True


def test_try_fallback_no_allowlist_uses_first_untried():
    """With allowed_providers=None, first untried fallback is activated (legacy behaviour)."""
    primary = _NamedProvider("kimi")
    fallbacks = [_NamedProvider("gemini"), _NamedProvider("haiku")]
    agent = AgentRunner(
        provider=primary,
        tool_registry=ToolRegistry(),
        toolsets=[],
        allowed_providers=None,
    )
    agent._fallback_providers = fallbacks

    assert agent._try_fallback(Exception("boom")) is True
    assert agent.provider.name == "gemini"


def test_clear_interrupt():
    """clear_interrupt resets flag so agent can run again."""
    from mullet.core.agent.runner import AgentRunner
    from tests.test_agent_runner import _make_registry
    simple_response = LLMResponse(
        content="done",
        tool_calls=[],
        usage=TokenUsage(total_tokens=3),
    )
    provider = SimpleProvider([simple_response])
    agent = AgentRunner(
        provider=provider,
        tool_registry=_make_registry(),
        toolsets=[],
    )
    agent.interrupt("stop")
    assert agent._interrupt_flag is True
    agent.clear_interrupt()
    assert agent._interrupt_flag is False
    assert agent._interrupt_message is None
