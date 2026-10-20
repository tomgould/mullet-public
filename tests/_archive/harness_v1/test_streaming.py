"""Tests for the harness streaming layer."""

from __future__ import annotations

import asyncio
from typing import Iterator

import pytest

from mullet.core.providers.base import LLMResponse, Provider, StreamChunk as ProviderStreamChunk
from mullet.harness.engine.streaming import StreamChunk, StreamingProviderAdapter
from mullet.harness.tui.bridge import EngineBridge, EnginePromptRequest


class _FakeStreamingProvider(Provider):
    """Provider that yields predefined chunks from create_stream."""

    def __init__(self, chunks: list[ProviderStreamChunk]) -> None:
        self.chunks = chunks
        self.create_calls: list[tuple] = []

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        self.create_calls.append((messages, tools))
        return LLMResponse(content="".join(c.content for c in self.chunks))

    def create_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Iterator[ProviderStreamChunk]:
        for chunk in self.chunks:
            yield chunk


class _FakeEngineWithAstream:
    """Engine that yields chunks from astream()."""

    def __init__(self, chunks: list[StreamChunk]) -> None:
        self.chunks = chunks

    async def astream(self, prompt: str, **kwargs):
        for chunk in self.chunks:
            yield chunk

    def run(self, prompt: str, **kwargs):
        return {
            "session_id": "sess-fallback",
            "response": prompt,
            "stop_reason": "stop",
            "input_tokens": 1,
            "output_tokens": 1,
        }


@pytest.mark.harness
class TestStreamingAdapter:
    async def test_streaming_adapter_emits_text_deltas(self) -> None:
        provider = _FakeStreamingProvider([
            ProviderStreamChunk(content="Hello "),
            ProviderStreamChunk(content="world"),
        ])
        adapter = StreamingProviderAdapter(provider)
        chunks = [c async for c in adapter.astream([{"role": "user", "content": "hi"}])]
        deltas = [c for c in chunks if c.kind == "text_delta"]
        assert len(deltas) == 2
        assert deltas[0].text == "Hello "
        assert deltas[1].text == "world"

    async def test_streaming_adapter_handles_stop_finish_reason(self) -> None:
        provider = _FakeStreamingProvider([
            ProviderStreamChunk(content="done", finish_reason="stop"),
        ])
        adapter = StreamingProviderAdapter(provider)
        chunks = [c async for c in adapter.astream([{"role": "user", "content": "hi"}])]
        stop_chunks = [c for c in chunks if c.kind == "stop"]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].metadata == {"reason": "stop"}


@pytest.mark.harness
class TestBridgeStreaming:
    async def test_bridge_emits_chunk_events(self) -> None:
        chunks = [
            StreamChunk("metadata", metadata={"session_id": "sess-stream"}),
            StreamChunk("text_delta", text="chunk1"),
            StreamChunk("text_delta", text="chunk2"),
            StreamChunk("stop", metadata={"reason": "stop"}),
        ]
        engine = _FakeEngineWithAstream(chunks)
        bridge = EngineBridge(engine)
        await bridge.start()
        try:
            req = EnginePromptRequest(
                prompt="hello",
                session_id=None,
                provider="kimi",
                model=None,
                request_id="req-stream",
            )
            await bridge.submit(req)

            events = []
            for _ in range(10):
                evt = await asyncio.wait_for(bridge.next_event(), timeout=2.0)
                events.append(evt)
                if evt.kind == "assistant_complete":
                    break

            kinds = [e.kind for e in events]
            assert "session_id" in kinds
            assert "assistant_chunk" in kinds
            chunk_events = [e for e in events if e.kind == "assistant_chunk"]
            assert len(chunk_events) == 2
            assert chunk_events[0].payload["text"] == "chunk1"
            assert chunk_events[1].payload["text"] == "chunk2"
        finally:
            await bridge.stop()

    async def test_bridge_then_complete_event(self) -> None:
        chunks = [
            StreamChunk("metadata", metadata={"session_id": "sess-stream"}),
            StreamChunk("text_delta", text="full"),
            StreamChunk("text_delta", text=" response"),
            StreamChunk("stop", metadata={"reason": "stop"}),
        ]
        engine = _FakeEngineWithAstream(chunks)
        bridge = EngineBridge(engine)
        await bridge.start()
        try:
            req = EnginePromptRequest(
                prompt="hello",
                session_id=None,
                provider="kimi",
                model=None,
                request_id="req-complete",
            )
            await bridge.submit(req)

            complete_evt = None
            for _ in range(10):
                evt = await asyncio.wait_for(bridge.next_event(), timeout=2.0)
                if evt.kind == "assistant_complete":
                    complete_evt = evt
                    break

            assert complete_evt is not None
            assert complete_evt.payload["response"] == "full response"
        finally:
            await bridge.stop()
