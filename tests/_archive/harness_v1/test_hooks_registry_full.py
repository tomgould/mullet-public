"""Tests for the full HookRegistry: payload routing, async, logging, listing."""

from __future__ import annotations

import asyncio
import logging

import pytest

from mullet.harness.hooks.registry import Block, HookRegistry
from mullet.harness.hooks.types import (
    EventPayload,
    MessagePayload,
    ParamsPayload,
    PostToolUsePayload,
    PreToolUsePayload,
    TransformPayload,
)

pytestmark = pytest.mark.harness


def test_register_for_payload_routes_to_correct_event() -> None:
    """register_for_payload(PreToolUsePayload, h) wires h to 'PreToolUse'."""
    reg = HookRegistry()
    captured: list[PreToolUsePayload] = []

    def handler(payload: PreToolUsePayload) -> None:
        captured.append(payload)
        return None

    reg.register_for_payload(PreToolUsePayload, handler)

    payload = PreToolUsePayload(
        tool_name="Read", tool_args={"path": "/tmp/x"}, session_id="s1"
    )
    result = reg.fire("PreToolUse", payload)

    assert result is payload
    assert captured == [payload]
    assert reg.count("PreToolUse") == 1


@pytest.mark.parametrize(
    ("payload_type", "event_name"),
    [
        (PreToolUsePayload, "PreToolUse"),
        (PostToolUsePayload, "PostToolUse"),
        (MessagePayload, "Message"),
        (EventPayload, "Event"),
        (TransformPayload, "Transform"),
        (ParamsPayload, "Params"),
    ],
)
def test_register_for_payload_maps_each_type(payload_type: type, event_name: str) -> None:
    """All 6 payload types must map to the right event name."""
    reg = HookRegistry()
    reg.register_for_payload(payload_type, lambda p: None)
    assert reg.count(event_name) == 1


def test_fire_async_handles_async_handler() -> None:
    """fire_async awaits coroutine handlers and returns their result."""
    reg = HookRegistry()

    async def handler(payload: str) -> str:
        await asyncio.sleep(0)
        return payload + "-async"

    reg.register("Message", handler)
    result = asyncio.run(reg.fire_async("Message", "hi"))
    assert result == "hi-async"


def test_fire_async_handles_mixed_sync_and_async() -> None:
    """fire_async dispatches sync and async handlers in registration order."""
    reg = HookRegistry()

    def sync_h(payload: str) -> str:
        return payload + "-s"

    async def async_h(payload: str) -> str:
        return payload + "-a"

    reg.register("Message", sync_h)
    reg.register("Message", async_h)

    result = asyncio.run(reg.fire_async("Message", "go"))
    assert result == "go-s-a"


def test_fire_sync_with_async_handler_raises() -> None:
    """Sync fire() must raise RuntimeError when it hits a coroutine handler."""
    reg = HookRegistry()

    async def async_h(_: str) -> str:
        return "nope"

    reg.register("Event", async_h)

    with pytest.raises(RuntimeError, match="async"):
        reg.fire("Event", "x")


def test_list_handlers_returns_callables() -> None:
    """list_handlers returns a copy of the chain for the given event."""
    reg = HookRegistry()

    def a(_: str) -> None:
        return None

    def b(_: str) -> None:
        return None

    reg.register("Transform", a)
    reg.register("Transform", b)

    handlers = reg.list_handlers("Transform")
    assert handlers == [a, b]
    # Mutating returned list does not affect the registry.
    handlers.clear()
    assert reg.count("Transform") == 2


def test_list_handlers_unknown_event_raises() -> None:
    """list_handlers must reject unknown event names."""
    reg = HookRegistry()
    with pytest.raises(ValueError, match="Unknown hook event"):
        reg.list_handlers("Nope")


def test_logging_emits_debug_on_fire(caplog: pytest.LogCaptureFixture) -> None:
    """fire() emits a DEBUG log line containing the event name and outcome."""
    reg = HookRegistry()
    reg.register("Params", lambda p: None)

    with caplog.at_level(logging.DEBUG, logger="mullet.harness.hooks.registry"):
        reg.fire("Params", "x")

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("event=Params" in m and "outcome=passthrough" in m for m in debug_messages)


def test_logging_emits_debug_on_block_outcome(caplog: pytest.LogCaptureFixture) -> None:
    """fire() emits an outcome=blocked debug line when a Block is returned."""
    reg = HookRegistry()
    reg.register("PreToolUse", lambda p: Block("denied"))

    with caplog.at_level(logging.DEBUG, logger="mullet.harness.hooks.registry"):
        reg.fire("PreToolUse", "x")

    msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("outcome=blocked" in m for m in msgs)


def test_fire_async_logs_debug(caplog: pytest.LogCaptureFixture) -> None:
    """fire_async emits a DEBUG log line tagged 'fire_async'."""
    reg = HookRegistry()
    reg.register("Event", lambda p: None)

    with caplog.at_level(logging.DEBUG, logger="mullet.harness.hooks.registry"):
        asyncio.run(reg.fire_async("Event", "x"))

    msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("fire_async" in m and "event=Event" in m for m in msgs)


def test_register_for_payload_unknown_type_raises() -> None:
    """register_for_payload rejects classes not in the payload map."""
    reg = HookRegistry()

    class NotAPayload:
        pass

    with pytest.raises(ValueError, match="No event mapping"):
        reg.register_for_payload(NotAPayload, lambda p: p)
