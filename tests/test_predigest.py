"""Tests for the pre-digest pipeline."""

from mullet.core.knowledge.predigest import PreDigestPipeline, PREDIGEST_THRESHOLD_CHARS
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.agent.types import TokenUsage


class MockLocalProvider(Provider):
    name = "local"
    def create(self, messages, **kw):
        return LLMResponse(content="Summary: key points extracted.", usage=TokenUsage(total_tokens=100))


def test_should_predigest_small():
    pipeline = PreDigestPipeline(local_provider=MockLocalProvider())
    assert pipeline.should_predigest("short content") is False


def test_should_predigest_large():
    pipeline = PreDigestPipeline(local_provider=MockLocalProvider())
    large = "x" * (PREDIGEST_THRESHOLD_CHARS + 1)
    assert pipeline.should_predigest(large) is True


def test_should_predigest_no_provider():
    pipeline = PreDigestPipeline(local_provider=None)
    large = "x" * (PREDIGEST_THRESHOLD_CHARS + 1)
    assert pipeline.should_predigest(large) is False


def test_predigest_returns_summary():
    pipeline = PreDigestPipeline(local_provider=MockLocalProvider())
    content = "x" * (PREDIGEST_THRESHOLD_CHARS + 1)
    summary = pipeline.predigest(content, instruction="summarize")
    assert "Summary" in summary or "key points" in summary


def test_predigest_fallback_no_provider():
    pipeline = PreDigestPipeline(local_provider=None)
    content = "x" * 100000
    result = pipeline.predigest(content)
    assert "truncated" in result


def test_build_cloud_prompt():
    pipeline = PreDigestPipeline()
    prompt = pipeline.build_cloud_prompt("fix the bug", "Summary of code")
    assert "fix the bug" in prompt
    assert "Summary of code" in prompt
    assert "pre-processed" in prompt.lower()


def test_predigest_with_task_context():
    """task_context non-empty triggers line 55 (prompt += task context)."""
    pipeline = PreDigestPipeline(local_provider=MockLocalProvider())
    content = "x" * (PREDIGEST_THRESHOLD_CHARS + 1)
    summary = pipeline.predigest(content, instruction="summarize", task_context="fix the bug")
    assert summary  # mock returns its fixed string


def test_predigest_truncates_oversized_prompt():
    """Prompt > 128K chars triggers the truncation branch (line 61)."""
    pipeline = PreDigestPipeline(local_provider=MockLocalProvider())
    # 129K chars of content guarantees prompt exceeds 128_000 chars
    content = "a" * 129_000
    summary = pipeline.predigest(content, instruction="summarize")
    assert summary  # must not crash


def test_predigest_stores_in_knowledge():
    """When knowledge is set and summary non-empty, knowledge.remember is called (lines 76-82)."""
    from unittest.mock import MagicMock

    knowledge = MagicMock()
    pipeline = PreDigestPipeline(local_provider=MockLocalProvider(), knowledge=knowledge)
    content = "x" * (PREDIGEST_THRESHOLD_CHARS + 1)
    pipeline.predigest(content, instruction="summarize")
    knowledge.remember.assert_called_once()


def test_predigest_swallows_knowledge_exception():
    """Exceptions from knowledge.remember are caught and logged (lines 83-84)."""
    from unittest.mock import MagicMock

    knowledge = MagicMock()
    knowledge.remember.side_effect = RuntimeError("store failed")
    pipeline = PreDigestPipeline(local_provider=MockLocalProvider(), knowledge=knowledge)
    content = "x" * (PREDIGEST_THRESHOLD_CHARS + 1)
    summary = pipeline.predigest(content)  # must not raise
    assert summary


def test_predigest_provider_exception_returns_truncated():
    """If local_provider.create raises, predigest falls back to truncated content (lines 89-91)."""
    from unittest.mock import MagicMock

    bad_provider = MagicMock()
    bad_provider.create.side_effect = RuntimeError("model down")
    pipeline = PreDigestPipeline(local_provider=bad_provider)
    content = "y" * (PREDIGEST_THRESHOLD_CHARS + 10)
    result = pipeline.predigest(content)
    assert "truncated" in result
