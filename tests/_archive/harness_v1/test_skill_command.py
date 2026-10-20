"""Tests for mullet.harness.commands.skill."""

from __future__ import annotations

import pytest

from mullet.harness.commands.skill import (
    handle_skill_command,
    register_skill_command,
)
from mullet.harness.hooks.registry import HookRegistry
from mullet.harness.hooks.types import MessagePayload

pytestmark = pytest.mark.harness


def test_skill_list_returns_loaded_skills() -> None:
    out = handle_skill_command("list")
    assert "commit" in out
    assert "review" in out


def test_handle_skill_no_args_lists_skills() -> None:
    out = handle_skill_command("")
    assert "commit" in out
    assert "review" in out


def test_skill_show_existing_returns_body() -> None:
    out = handle_skill_command("show commit")
    assert "## When to use" in out


def test_skill_show_unknown() -> None:
    out = handle_skill_command("show no-such-skill-xyz")
    assert "No skill named" in out
    assert "no-such-skill-xyz" in out


def test_skill_show_without_name_returns_usage() -> None:
    out = handle_skill_command("show")
    assert "Usage:" in out


def test_skill_use_registers_message_hook() -> None:
    reg = HookRegistry()
    out = handle_skill_command("use commit", registry=reg)
    assert "commit" in out
    assert "next prompt" in out
    assert reg.count("Message") == 1


def test_skill_use_unknown_returns_message() -> None:
    reg = HookRegistry()
    out = handle_skill_command("use does-not-exist", registry=reg)
    assert "No skill named" in out
    assert reg.count("Message") == 0


def test_skill_use_without_name_returns_usage() -> None:
    reg = HookRegistry()
    out = handle_skill_command("use", registry=reg)
    assert "Usage:" in out
    assert reg.count("Message") == 0


def test_skill_use_hook_fires_once_then_unregisters() -> None:
    reg = HookRegistry()
    handle_skill_command("use commit", registry=reg)
    assert reg.count("Message") == 1

    payload = MessagePayload(
        message={"role": "user", "content": "hi"},
        session_id="s1",
        direction="user",
    )
    reg.fire("Message", payload)
    assert reg.count("Message") == 0


def test_skill_use_hook_injects_skill_body_into_messages() -> None:
    reg = HookRegistry()
    handle_skill_command("use commit", registry=reg)

    payload = MessagePayload(
        message={"role": "user", "content": "please commit"},
        session_id="s1",
        direction="user",
    )
    result = reg.fire("Message", payload)

    # The hook should have prepended a system entry with the skill body.
    messages = getattr(result, "messages", None)
    assert isinstance(messages, list)
    assert messages, "messages list must not be empty"
    first = messages[0]
    assert first.get("role") == "system"
    assert "## When to use" in first.get("content", "")


def test_skill_use_hook_injects_into_existing_messages_list() -> None:
    reg = HookRegistry()
    handle_skill_command("use commit", registry=reg)

    payload = MessagePayload(
        message={"role": "user", "content": "hi"},
        session_id="s1",
        direction="user",
    )
    # Attach a pre-existing messages list (extra="allow" permits this).
    payload.messages = [{"role": "user", "content": "hi"}]  # type: ignore[attr-defined]

    reg.fire("Message", payload)

    messages = payload.messages  # type: ignore[attr-defined]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_register_skill_command_populates_dict_registry() -> None:
    handlers: dict = {}
    register_skill_command(handlers)
    assert "skill" in handlers
    assert callable(handlers["skill"])
    # Smoke: invoking the registered handler returns a string listing.
    result = handlers["skill"]("list")
    assert isinstance(result, str)
    assert "commit" in result


def test_skill_unknown_subcommand_returns_usage() -> None:
    out = handle_skill_command("frobnicate commit")
    assert "Unknown" in out or "Usage" in out


def test_skill_use_does_not_affect_other_registries() -> None:
    reg_a = HookRegistry()
    reg_b = HookRegistry()
    handle_skill_command("use commit", registry=reg_a)
    assert reg_a.count("Message") == 1
    assert reg_b.count("Message") == 0
