"""Tests for the Phase-1 hook registry stub."""

import pytest

from mullet.harness.hooks.registry import Block, HookRegistry


@pytest.mark.harness
class TestHookRegistry:
    """Cover register, fire, mutation, Block, iteration cap, count, clear, unregister."""

    def test_register_and_fire_passthrough(self) -> None:
        """Handler returns None → payload unchanged."""
        reg = HookRegistry()
        reg.register("PreToolUse", lambda p: None)
        assert reg.fire("PreToolUse", "hello") == "hello"

    def test_register_and_fire_mutation(self) -> None:
        """Handler returns modified payload, downstream sees mutation."""
        reg = HookRegistry()

        def add_suffix(payload: str) -> str:
            return payload + "-world"

        def upper(payload: str) -> str:
            return payload.upper()

        reg.register("Message", add_suffix)
        reg.register("Message", upper)
        assert reg.fire("Message", "hello") == "HELLO-WORLD"

    def test_block_short_circuits(self) -> None:
        """Handler returns Block → fire returns Block, later handlers not called."""
        reg = HookRegistry()
        later_called = False

        def blocker(_: str) -> Block:
            return Block("denied")

        def later(_: str) -> str:
            nonlocal later_called
            later_called = True
            return "nope"

        reg.register("PreToolUse", blocker)
        reg.register("PreToolUse", later)
        result = reg.fire("PreToolUse", "go")
        assert isinstance(result, Block)
        assert result.reason == "denied"
        assert not later_called

    def test_iteration_cap_raises(self) -> None:
        """Register cap+1 passthrough handlers, fire raises RuntimeError."""
        from mullet.harness.hooks.registry import _ITERATION_CAP
        reg = HookRegistry()
        for i in range(_ITERATION_CAP + 1):
            reg.register("Params", lambda p, idx=i: None)
        with pytest.raises(RuntimeError, match="Hook iteration cap exceeded for event Params"):
            reg.fire("Params", "x")

    def test_unknown_event_rejected_on_register(self) -> None:
        """register('FakeEvent', ...) raises ValueError."""
        reg = HookRegistry()
        with pytest.raises(ValueError, match="Unknown hook event: FakeEvent"):
            reg.register("FakeEvent", lambda p: p)

    def test_count_and_clear(self) -> None:
        """register 3, count==3, clear, count==0."""
        reg = HookRegistry()
        for _ in range(3):
            reg.register("Event", lambda p: p)
        assert reg.count("Event") == 3
        assert reg.count() == 3
        reg.clear()
        assert reg.count() == 0
        assert reg.count("Event") == 0

    def test_unregister_removes_handler(self) -> None:
        """register 2, unregister 1, count==1."""
        reg = HookRegistry()

        def a(_: str) -> str:
            return "a"

        def b(_: str) -> str:
            return "b"

        reg.register("Transform", a)
        reg.register("Transform", b)
        assert reg.count("Transform") == 2
        reg.unregister("Transform", a)
        assert reg.count("Transform") == 1
        assert reg.fire("Transform", "x") == "b"
