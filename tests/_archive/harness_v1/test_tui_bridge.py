"""Tests for :mod:`mullet.harness.tui.bridge` — async engine ↔ TUI bridge."""

from __future__ import annotations

import asyncio

import pytest

from mullet.harness.tui.bridge import EngineBridge, EngineEvent, EnginePromptRequest


class _StubEngine:
    """Minimal sync engine that returns a fixed dict from ``run``."""

    def __init__(self, result: dict | None = None, *, raises: Exception | None = None) -> None:
        self.result = result or {
            "session_id": "sess-stub-001",
            "response": "hello world",
            "stop_reason": "end_turn",
            "input_tokens": 10,
            "output_tokens": 20,
        }
        self.raises = raises
        self.calls: list[dict] = []

    def run(self, prompt: str, *, session_id=None, provider="kimi", model=None, **_):
        self.calls.append(
            {"prompt": prompt, "session_id": session_id, "provider": provider, "model": model}
        )
        if self.raises is not None:
            raise self.raises
        return self.result


def _request(prompt: str = "hi", **kwargs) -> EnginePromptRequest:
    return EnginePromptRequest(
        prompt=prompt,
        session_id=kwargs.get("session_id"),
        provider=kwargs.get("provider", "kimi"),
        model=kwargs.get("model"),
        request_id=kwargs.get("request_id", "req-1"),
    )


@pytest.mark.harness
@pytest.mark.tui
class TestEngineBridge:
    async def test_bridge_round_trip(self) -> None:
        engine = _StubEngine(
            result={
                "session_id": "sess-XYZ",
                "response": "hi back",
                "stop_reason": "end_turn",
                "input_tokens": 1,
                "output_tokens": 2,
            }
        )
        bridge = EngineBridge(engine)
        await bridge.start()
        try:
            await bridge.submit(_request("hello"))

            evt1 = await asyncio.wait_for(bridge.next_event(), timeout=2.0)
            assert isinstance(evt1, EngineEvent)
            assert evt1.kind == "session_id"
            assert evt1.payload["session_id"] == "sess-XYZ"
            assert evt1.request_id == "req-1"

            evt2 = await asyncio.wait_for(bridge.next_event(), timeout=2.0)
            assert evt2.kind == "assistant_complete"
            assert evt2.payload["response"] == "hi back"
            assert evt2.payload["stop_reason"] == "end_turn"
            assert evt2.payload["input_tokens"] == 1
            assert evt2.payload["output_tokens"] == 2
            assert evt2.request_id == "req-1"
        finally:
            await bridge.stop()

        assert engine.calls and engine.calls[0]["prompt"] == "hello"

    async def test_bridge_handles_engine_exception(self) -> None:
        engine = _StubEngine(raises=ValueError("boom"))
        bridge = EngineBridge(engine)
        await bridge.start()
        try:
            await bridge.submit(_request("trigger"))
            evt = await asyncio.wait_for(bridge.next_event(), timeout=2.0)
            assert evt.kind == "error"
            assert evt.payload["message"] == "boom"
            assert evt.payload["type"] == "ValueError"
            assert evt.request_id == "req-1"
        finally:
            await bridge.stop()

    async def test_bridge_stop_cancels_worker(self) -> None:
        engine = _StubEngine()
        bridge = EngineBridge(engine)
        await bridge.start()
        worker = bridge._worker_task
        assert worker is not None
        assert not worker.done()

        await bridge.stop()
        # After stop, the worker task is cleared and the previous task is done.
        assert bridge._worker_task is None
        assert worker.done()

    async def test_submit_to_full_queue_blocks(self) -> None:
        # Use a sentinel engine that blocks on a threading.Event the test owns,
        # so we can release it cleanly during teardown and avoid a stuck thread.
        import threading

        release = threading.Event()

        class _BlockingEngine:
            def run(self, *_args, **_kwargs):
                release.wait(timeout=10.0)
                return {
                    "session_id": "sess-blk",
                    "response": "",
                    "stop_reason": "end_turn",
                }

        bridge = EngineBridge(_BlockingEngine(), max_queue_size=1)
        await bridge.start()
        try:
            # First request: worker grabs it from the queue and blocks inside
            # ``engine.run``.  Give the loop a tick to dequeue.
            await bridge.submit(_request("first", request_id="r1"))
            await asyncio.sleep(0.05)

            # Second request fills the (empty again) queue — capacity = 1.
            await bridge.submit(_request("second", request_id="r2"))

            # Third submit must block: queue is full and the worker is still
            # blocked on the first request.  Expect a TimeoutError.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    bridge.submit(_request("third", request_id="r3")),
                    timeout=0.2,
                )
        finally:
            # Unblock the worker thread before tearing the bridge down so the
            # to_thread executor can finalize cleanly.
            release.set()
            await bridge.stop()

    async def test_next_event_timeout_returns_none(self) -> None:
        bridge = EngineBridge(_StubEngine())
        await bridge.start()
        try:
            result = await bridge.next_event(timeout=0.05)
            assert result is None
        finally:
            await bridge.stop()
