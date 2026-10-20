"""Tests for multimodal / image input support."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Any

import pytest

from mullet.harness.engine.multimodal import ImageAttachment, MultimodalMessageBuilder
from mullet.harness.commands.image import handle_image_command
from mullet.harness.engine.loop import HarnessEngine
from mullet.harness.hooks.registry import HookRegistry
from mullet.harness.sessions.store import SessionStore


class _MockProvider:
    """Synchronous mock provider that records calls and returns a fixed response."""

    supports_tool_calling = True
    name = "mock"

    def __init__(self, content: str = "OK", finish_reason: str = "stop") -> None:
        self.calls: list[tuple] = []
        self._content = content
        self._finish_reason = finish_reason

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> Any:
        from mullet.core.agent.types import TokenUsage
        from mullet.core.providers.base import LLMResponse

        self.calls.append((messages, tools))
        return LLMResponse(
            content=self._content,
            finish_reason=self._finish_reason,
            usage=TokenUsage(),
        )


@pytest.fixture
def mock_sync_provider():
    return _MockProvider()


@pytest.fixture
def engine(tmp_path, mock_sync_provider, monkeypatch):
    store = SessionStore(base_dir=tmp_path / "sessions")
    monkeypatch.setattr(
        "mullet.harness.engine.loop._resolve_provider",
        lambda name, model=None: mock_sync_provider,
    )
    return HarnessEngine(store=store, hooks=HookRegistry())


class TestImageAttachment:
    def test_image_from_path_loads_and_encodes_base64(self, tmp_path: Path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        att = ImageAttachment.from_path(img_path)
        assert att.base64_data is not None
        assert base64.b64decode(att.base64_data) == b"\x89PNG\r\n\x1a\nfake"
        assert att.mime_type == "image/png"
        assert att.path == img_path

    def test_image_from_url_returns_attachment_with_url(self):
        att = ImageAttachment.from_url("https://example.com/foo.jpg")
        assert att.url == "https://example.com/foo.jpg"
        assert att.path is None
        assert att.base64_data is None

    def test_anthropic_block_format(self):
        att = ImageAttachment(base64_data="abc123", mime_type="image/png")
        block = att.to_anthropic_block()
        assert block == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123",
            },
        }

    def test_openai_block_format(self):
        att = ImageAttachment(base64_data="abc123", mime_type="image/png")
        block = att.to_openai_block()
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_openai_block_with_url(self):
        att = ImageAttachment.from_url("https://example.com/pic.jpg")
        block = att.to_openai_block()
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/pic.jpg"


class TestMultimodalMessageBuilder:
    def test_builder_text_then_image_combined(self):
        att = ImageAttachment(base64_data="abc", mime_type="image/png")
        blocks = MultimodalMessageBuilder().text("hello").image(att).build(format="openai")
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "hello"}
        assert blocks[1]["type"] == "image_url"

    def test_builder_multiple_images(self):
        att1 = ImageAttachment(base64_data="a", mime_type="image/png")
        att2 = ImageAttachment(base64_data="b", mime_type="image/jpeg")
        blocks = (
            MultimodalMessageBuilder()
            .text("hi")
            .image(att1)
            .image(att2)
            .build(format="anthropic")
        )
        assert len(blocks) == 3
        assert blocks[0] == {"type": "text", "text": "hi"}
        assert blocks[1]["type"] == "image"
        assert blocks[2]["type"] == "image"

    def test_builder_anthropic_vs_openai_format(self):
        att = ImageAttachment(base64_data="x", mime_type="image/gif")
        anthropic_blocks = MultimodalMessageBuilder().text("t").image(att).build(format="anthropic")
        openai_blocks = MultimodalMessageBuilder().text("t").image(att).build(format="openai")
        assert anthropic_blocks[1]["type"] == "image"
        assert openai_blocks[1]["type"] == "image_url"


class TestCLIImageFlags:
    def test_cli_image_flag_attaches(self, monkeypatch, capsys, tmp_path: Path):
        captured_images = []
        img_file = tmp_path / "fake.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        class MockEngine:
            def run(self, prompt, session_id=None, provider="kimi", model=None, images=None):
                captured_images.extend(images or [])
                return {
                    "session_id": "sess-test",
                    "response": "ok",
                    "stop_reason": "stop",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

        monkeypatch.setattr("mullet.cli.harness_cli.HarnessEngine", MockEngine)
        from mullet.cli.harness_cli import cmd_shell

        args = argparse.Namespace(
            prompt="What's this?",
            prompt_flag=None,
            list_sessions=False,
            session_id=None,
            provider="kimi",
            model=None,
            output_format="text",
            image=[str(img_file)],
            image_url=["https://example.com/x.jpg"],
            dry_run=False,
        )
        rc = cmd_shell(args)
        assert rc == 0
        assert len(captured_images) == 2
        assert captured_images[0].path == img_file
        assert captured_images[1].url == "https://example.com/x.jpg"


class TestEngineRunWithImages:
    def test_engine_run_with_images_passes_to_provider(self, engine, mock_sync_provider, tmp_path: Path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        img = ImageAttachment.from_path(img_path)

        engine.run("hello", images=[img])

        messages, _tools = mock_sync_provider.calls[-1]
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0] == {"type": "text", "text": "hello"}
        assert user_msg["content"][1]["type"] == "image_url"
        assert "base64" in user_msg["content"][1]["image_url"]["url"]

    def test_engine_run_with_images_anthropic_format(self, engine, mock_sync_provider, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name, model=None: mock_sync_provider,
        )
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        img = ImageAttachment.from_path(img_path)

        engine.run("hello", provider="anthropic", images=[img])

        messages, _tools = mock_sync_provider.calls[-1]
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][1]["type"] == "image"
        assert user_msg["content"][1]["source"]["type"] == "base64"


class TestImageCommand:
    def test_image_command_add_clear_list(self, tmp_path: Path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        class FakeEngine:
            pass

        eng = FakeEngine()

        out = handle_image_command(f"add {img_path}", engine=eng)
        assert "Attached" in out
        assert len(eng._pending_images) == 1

        out = handle_image_command("list", engine=eng)
        assert "Pending images" in out
        assert "test.png" in out

        out = handle_image_command("clear", engine=eng)
        assert "cleared" in out
        assert eng._pending_images == []

    def test_image_command_add_missing_file(self):
        class FakeEngine:
            pass

        out = handle_image_command("add /nonexistent.png", engine=FakeEngine())
        assert "not found" in out

    def test_image_command_no_engine(self):
        out = handle_image_command("list")
        assert "requires an active engine" in out

    def test_image_command_unknown_subcommand(self):
        class FakeEngine:
            pass

        out = handle_image_command("zoom", engine=FakeEngine())
        assert "Unknown" in out
