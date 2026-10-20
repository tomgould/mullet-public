"""Tests for mullet.harness.commands.plan."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mullet.harness.commands.plan import handle_plan_command
from mullet.harness.hooks.registry import HookRegistry
from mullet.harness.hooks.types import MessagePayload
from mullet.harness.permissions.checker import PermissionChecker, PermissionsConfig

pytestmark = pytest.mark.harness


def _engine_with_perms(mode: str = "default") -> SimpleNamespace:
    perms = PermissionChecker(PermissionsConfig(mode=mode))
    return SimpleNamespace(perms=perms)


def test_plan_enter_mode_switches_to_plan() -> None:
    reg = HookRegistry()
    engine = _engine_with_perms("default")
    out = handle_plan_command("", registry=reg, engine=engine)
    assert engine.perms.mode == "plan"
    assert "plan mode" in out.lower()


def test_plan_exit_returns_to_default() -> None:
    reg = HookRegistry()
    engine = _engine_with_perms("plan")
    out = handle_plan_command("exit", registry=reg, engine=engine)
    assert engine.perms.mode == "default"
    assert "Exited plan mode." in out


def test_plan_with_topic_injects_topic() -> None:
    reg = HookRegistry()
    engine = _engine_with_perms("default")
    handle_plan_command("auth refactor", registry=reg, engine=engine)

    payload = MessagePayload(
        message={"role": "user", "content": "go"},
        session_id="s1",
        direction="user",
    )
    result = reg.fire("Message", payload)
    messages = getattr(result, "messages", None)
    assert isinstance(messages, list) and messages
    first = messages[0]
    assert first.get("role") == "system"
    content = first.get("content", "")
    assert "Topic: auth refactor" in content


def test_plan_registers_one_shot_message_hook() -> None:
    reg = HookRegistry()
    engine = _engine_with_perms("default")
    handle_plan_command("", registry=reg, engine=engine)
    assert reg.count("Message") == 1

    payload = MessagePayload(
        message={"role": "user", "content": "hi"},
        session_id="s1",
        direction="user",
    )
    reg.fire("Message", payload)
    # One-shot: hook unregistered itself after firing.
    assert reg.count("Message") == 0
