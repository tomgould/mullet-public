"""Tests for mullet.harness.memory.auto_context."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mullet.harness.hooks.types import MessagePayload
from mullet.harness.memory.auto_context import (
    AutoContextResolver,
    ContextChunk,
    make_auto_context_hook,
)

pytestmark = pytest.mark.harness


# ---------------------------------------------------------------------------
# AutoContextResolver
# ---------------------------------------------------------------------------


def test_resolver_returns_chunks_for_pool() -> None:
    fake_result = {
        "content": "def foo(): pass",
        "source": "src/foo.py",
        "score": 0.9,
    }
    with patch("mullet.knowledge.engine.search.searcher.Searcher") as MockSearcher:
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [fake_result]
        MockSearcher.return_value = mock_searcher
        resolver = AutoContextResolver(pool_name="test-pool", top_k=5)
        chunks = resolver.resolve("how does foo work")

    assert len(chunks) == 1
    assert chunks[0].source_path == "src/foo.py"
    assert "def foo(): pass" in chunks[0].snippet
    assert chunks[0].score == 0.9
    mock_searcher.close.assert_called_once()


def test_resolver_returns_empty_when_no_pool() -> None:
    resolver = AutoContextResolver(pool_name=None, top_k=5)
    chunks = resolver.resolve("anything")
    assert chunks == []


def test_resolver_respects_top_k() -> None:
    fake_results = [
        {"content": f"chunk {i}", "source": f"src/{i}.py", "score": 0.9 - i * 0.1}
        for i in range(10)
    ]
    with patch("mullet.knowledge.engine.search.searcher.Searcher") as MockSearcher:
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = fake_results
        MockSearcher.return_value = mock_searcher
        resolver = AutoContextResolver(pool_name="test-pool", top_k=3)
        chunks = resolver.resolve("query")

    assert len(chunks) == 3


def test_resolver_respects_max_chars_budget() -> None:
    big_snippet = "x" * 5000
    fake_results = [
        {"content": big_snippet, "source": "src/a.py", "score": 0.9},
        {"content": big_snippet, "source": "src/b.py", "score": 0.8},
    ]
    with patch("mullet.knowledge.engine.search.searcher.Searcher") as MockSearcher:
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = fake_results
        MockSearcher.return_value = mock_searcher
        resolver = AutoContextResolver(
            pool_name="test-pool", top_k=5, max_chars_total=6000
        )
        chunks = resolver.resolve("query")

    # First chunk is 5000 chars, second would exceed 6000, so only 1.
    assert len(chunks) == 1
    assert chunks[0].source_path == "src/a.py"


def test_assemble_system_message_format() -> None:
    chunks = [
        ContextChunk(
            source_path="src/foo.py",
            snippet="def foo(): pass",
            score=0.87,
            rationale="matched: foo",
        ),
        ContextChunk(
            source_path="docs/X.md",
            snippet="Some docs",
            score=0.75,
            rationale="matched: docs",
        ),
    ]
    resolver = AutoContextResolver()
    msg = resolver.assemble_system_message(chunks)

    assert "# Relevant context\n" in msg
    assert "## src/foo.py (score=0.87)" in msg
    assert "```\ndef foo(): pass\n```" in msg
    assert "## docs/X.md (score=0.75)" in msg
    assert "```\nSome docs\n```" in msg


# ---------------------------------------------------------------------------
# Message hook
# ---------------------------------------------------------------------------


def test_hook_fires_on_user_message() -> None:
    resolver = AutoContextResolver(pool_name="test-pool", top_k=1)
    hook = make_auto_context_hook(resolver)

    with patch.object(
        resolver,
        "resolve",
        return_value=[
            ContextChunk(
                source_path="src/a.py",
                snippet="hello",
                score=0.9,
                rationale="matched: hi",
            )
        ],
    ):
        payload = MessagePayload(
            message={"role": "user", "content": "hi"},
            session_id="s1",
            direction="user",
        )
        payload.messages = [{"role": "user", "content": "hi"}]
        result = hook(payload)

    assert result is payload
    assert payload.messages[0]["role"] == "system"
    assert "src/a.py" in payload.messages[0]["content"]
    assert payload.messages[1]["role"] == "user"


def test_hook_skips_non_user_messages() -> None:
    resolver = AutoContextResolver()
    hook = make_auto_context_hook(resolver)

    payload = MessagePayload(
        message={"role": "assistant", "content": "hello"},
        session_id="s1",
        direction="assistant",
    )
    result = hook(payload)
    assert result is None


def test_hook_skips_tool_followup_iteration() -> None:
    """When direction=user but last message is tool, hook should skip."""
    resolver = AutoContextResolver()
    hook = make_auto_context_hook(resolver)

    payload = MessagePayload(
        message={"role": "user", "content": "hi"},
        session_id="s1",
        direction="user",
    )
    payload.messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "content": "{}"},
    ]
    result = hook(payload)
    assert result is None
