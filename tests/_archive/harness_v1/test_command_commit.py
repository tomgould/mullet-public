"""Tests for mullet.harness.commands.commit."""

from __future__ import annotations

import pytest

from mullet.harness.commands.commit import handle_commit_command
from mullet.harness.hooks.registry import HookRegistry
from mullet.harness.hooks.types import MessagePayload

pytestmark = pytest.mark.harness


def test_commit_registers_message_hook() -> None:
    reg = HookRegistry()
    out = handle_commit_command("", registry=reg)
    assert reg.count("Message") == 1
    assert "/commit ready" in out


def test_commit_with_message_hint_includes_hint_in_system_message() -> None:
    reg = HookRegistry()
    handle_commit_command("fix the auth bug", registry=reg)

    payload = MessagePayload(
        message={"role": "user", "content": "do it"},
        session_id="s1",
        direction="user",
    )
    result = reg.fire("Message", payload)

    messages = getattr(result, "messages", None)
    assert isinstance(messages, list)
    assert messages, "messages list must not be empty"
    first = messages[0]
    assert first.get("role") == "system"
    content = first.get("content", "")
    assert "Message hint: fix the auth bug" in content


def test_commit_returns_ready_text() -> None:
    reg = HookRegistry()
    out = handle_commit_command("", registry=reg)
    assert "/commit ready" in out
    assert "next message" in out
