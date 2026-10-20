"""Tests for the OpenAICompatibleProvider HUD agent-tracking hook.

Verifies that any provider going through OpenAICompatibleProvider.create()
writes to .mullet/state/agents.json on entry and removes the entry on exit,
including on exceptions. This is how Kimi/local/Moonshot/OpenRouter calls
show up live on the HUD when they are invoked.
"""
import json
from unittest.mock import MagicMock

import pytest

from mullet.core.providers import openai_compat
from mullet.hud import agent_state


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect agent_state writes to a tmp file so tests don't touch real state."""
    state_file = tmp_path / "agents.json"
    monkeypatch.setattr(agent_state, "_state_path", lambda: state_file)
    yield state_file


def _fake_openai_response(content: str = "hi"):
    """Build a mock openai ChatCompletion-like object."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    response.model = "qwen3.5-9b"
    return response


def test_brief_task_summary_extracts_last_user_message():
    messages = [
        {"role": "system", "content": "you are a helper"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "an answer"},
        {"role": "user", "content": "second question"},
    ]
    assert openai_compat._brief_task_summary(messages) == "second question"


def test_brief_task_summary_handles_multipart_content():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "part one"},
            {"type": "image", "source": "..."},
            {"type": "text", "text": "part two"},
        ]},
    ]
    assert openai_compat._brief_task_summary(messages) == "part one part two"


def test_brief_task_summary_empty_fallback():
    assert openai_compat._brief_task_summary([]) == "LLM call"
    assert openai_compat._brief_task_summary([{"role": "system", "content": "x"}]) == "LLM call"


def test_brief_task_summary_truncates_to_80():
    long_msg = "x" * 200
    summary = openai_compat._brief_task_summary([{"role": "user", "content": long_msg}])
    assert len(summary) == 80


def test_hud_track_adds_and_removes_on_success(isolated_state):
    messages = [{"role": "user", "content": "summarise this file"}]
    captured: dict = {}

    def snapshot_during_call():
        try:
            captured["during"] = json.loads(isolated_state.read_text())
        except (OSError, json.JSONDecodeError):
            captured["during"] = None

    with openai_compat._hud_agent_track("local", "qwen3.5-9b", messages):
        snapshot_during_call()

    # During the call: the agent was registered.
    assert captured["during"] is not None
    assert captured["during"]["active_count"] == 1
    assert captured["during"]["agents"][0]["provider"] == "local"
    assert captured["during"]["agents"][0]["model"] == "qwen3.5-9b"
    assert captured["during"]["agents"][0]["description"] == "summarise this file"

    # After the context exits: the agent is cleared.
    final = json.loads(isolated_state.read_text())
    assert final["active_count"] == 0


def test_hud_track_removes_on_exception(isolated_state):
    messages = [{"role": "user", "content": "failing call"}]
    with pytest.raises(RuntimeError):
        with openai_compat._hud_agent_track("kimi", "kimi-k2.6", messages):
            raise RuntimeError("boom")

    final = json.loads(isolated_state.read_text())
    assert final["active_count"] == 0


def test_hud_track_concurrent_agents_coexist(isolated_state):
    """Two overlapping calls both appear on the HUD at the same time."""
    messages_a = [{"role": "user", "content": "task A"}]
    messages_b = [{"role": "user", "content": "task B"}]
    with openai_compat._hud_agent_track("kimi", "kimi-k2.6", messages_a):
        with openai_compat._hud_agent_track("local", "qwen3.5-9b", messages_b):
            data = json.loads(isolated_state.read_text())
            assert data["active_count"] == 2
            providers = sorted(a["provider"] for a in data["agents"])
            assert providers == ["kimi", "local"]

    final = json.loads(isolated_state.read_text())
    assert final["active_count"] == 0


def test_provider_create_triggers_hud_track(isolated_state, monkeypatch):
    """Full create() path: HUD state is populated during the call, cleared after."""
    provider = openai_compat.OpenAICompatibleProvider(
        base_url="http://localhost:1234/v1",
        api_key="no-key-required",
        model="qwen3.5-9b",
    )
    # Force name to 'local' (should already be 'local' from _infer_name)
    assert provider.name == "local"

    seen_during: list[dict] = []

    def fake_create(**kwargs):
        # Verify the HUD state WAS written during the in-flight call
        try:
            seen_during.append(json.loads(isolated_state.read_text()))
        except (OSError, json.JSONDecodeError):
            seen_during.append({})
        return _fake_openai_response()

    monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

    messages = [{"role": "user", "content": "probe"}]
    resp = provider.create(messages=messages, max_tokens=10, temperature=0.7, stream=False)
    assert resp.content == "hi"

    # Inside the call: HUD showed the local agent
    assert seen_during and seen_during[0]["active_count"] == 1
    assert seen_during[0]["agents"][0]["provider"] == "local"
    assert seen_during[0]["agents"][0]["model"] == "qwen3.5-9b"
    assert seen_during[0]["agents"][0]["description"] == "probe"

    # After the call: HUD is empty
    after = json.loads(isolated_state.read_text())
    assert after["active_count"] == 0
