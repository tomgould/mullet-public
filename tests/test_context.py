"""Tests for multi-source context injection system."""

from mullet.core.agent.context import ContextCollector, Priority


def test_empty_collector():
    c = ContextCollector()
    assert c.entry_count == 0
    assert c.build() == ""


def test_single_entry():
    c = ContextCollector()
    c.add("test", "Hello world")
    assert c.entry_count == 1
    result = c.build()
    assert "Hello world" in result


def test_priority_ordering():
    c = ContextCollector()
    c.add("low", "Low priority", Priority.LOW)
    c.add("critical", "Critical info", Priority.CRITICAL)
    c.add("normal", "Normal stuff", Priority.NORMAL)

    result = c.build()
    # Critical should come before normal, which comes before low
    crit_pos = result.index("Critical")
    norm_pos = result.index("Normal")
    low_pos = result.index("Low")
    assert crit_pos < norm_pos < low_pos


def test_token_budget():
    c = ContextCollector()
    c.add("big", "x" * 40000, Priority.NORMAL)  # ~10K tokens
    c.add("small", "Important", Priority.CRITICAL)

    result = c.build(max_tokens=100)
    # Small (critical) should be included, big might be cut
    assert "Important" in result


def test_one_shot_consumed():
    c = ContextCollector()
    c.add("persistent", "Always here", Priority.NORMAL)
    c.add("temp", "One time only", Priority.NORMAL, one_shot=True)

    # First build: both present
    result1 = c.build()
    assert "Always here" in result1
    assert "One time only" in result1

    # Second build: one-shot consumed
    result2 = c.build()
    assert "Always here" in result2
    assert "One time only" not in result2


def test_clear():
    c = ContextCollector()
    c.add("test", "data")
    assert c.entry_count == 1
    c.clear()
    assert c.entry_count == 0


def test_summary():
    c = ContextCollector()
    c.add("knowledge", "project info", Priority.HIGH)
    c.add("knowledge", "more info", Priority.HIGH)
    c.add("skill", "instructions", Priority.NORMAL)

    summary = c.summary()
    assert summary["entries"] == 3
    assert summary["sources"]["knowledge"] == 2
    assert summary["sources"]["skill"] == 1


def test_empty_content_ignored():
    c = ContextCollector()
    c.add("test", "")
    c.add("test", "   ")
    assert c.entry_count == 0


def test_multiple_sources_merged():
    c = ContextCollector()
    c.add("knowledge", "## Project Context\nAuth module info")
    c.add("memory", "User prefers Python 3.11+")
    c.add("skill", "## Code Review\nCheck for security issues")

    result = c.build()
    assert "Project Context" in result
    assert "prefers Python" in result
    assert "Code Review" in result
