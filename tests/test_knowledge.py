"""Tests for the knowledge layer — context building, tools, prompt assembly."""

import json
from mullet.core.knowledge.context import build_system_prompt, estimate_tokens, truncate_to_budget
from mullet.core.knowledge.manager import KnowledgeContext


def test_build_system_prompt_no_context():
    result = build_system_prompt("You are an agent.")
    assert result == "You are an agent."


def test_build_system_prompt_with_context():
    ctx = KnowledgeContext(
        markdown="## Project Context\nThis is the auth module.",
        chunks_used=1,
        memories_used=0,
        tokens_estimated=20,
    )
    result = build_system_prompt("You are an agent.", ctx, "backend")
    assert "You are an agent." in result
    assert "# Project: backend" in result
    assert "auth module" in result


def test_build_system_prompt_empty_context():
    ctx = KnowledgeContext(markdown="", chunks_used=0, memories_used=0, tokens_estimated=0)
    result = build_system_prompt("Base prompt.", ctx)
    assert result == "Base prompt."


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world!!") == 3  # 13 chars / 4


def test_truncate_to_budget():
    short = "hello"
    assert truncate_to_budget(short, 100) == short

    long = "x" * 1000
    result = truncate_to_budget(long, 50)  # 50 tokens = 200 chars
    assert len(result) < 1000
    assert "truncated" in result


def test_knowledge_context_defaults():
    ctx = KnowledgeContext()
    assert ctx.markdown == ""
    assert ctx.chunks_used == 0
    assert ctx.memories_used == 0
    assert ctx.tokens_estimated == 0


# Tool factory tests (don't need a real KnowledgeManager)

class FakeKM:
    """Minimal KnowledgeManager stand-in for tool tests."""
    def search(self, query, top_k=10):
        return [{"content": f"Result for: {query}", "source": "test.py", "score": 0.95}]

    def remember(self, key, content, namespace="agent", ttl=None):
        self._stored = (key, content)

    def recall(self, query, namespace=None, top_k=5):
        return [{"key": "test", "content": "remembered value"}]


def test_trawl_search_tool():
    from mullet.core.knowledge.tools import make_trawl_search
    km = FakeKM()
    tool = make_trawl_search(km)
    result = tool(query="auth bug")
    assert "Result for: auth bug" in result
    assert "test.py" in result


def test_trawl_remember_tool():
    from mullet.core.knowledge.tools import make_trawl_remember
    km = FakeKM()
    tool = make_trawl_remember(km)
    result = tool(key="pattern", content="always use early returns")
    parsed = json.loads(result)
    assert parsed["success"] is True


def test_trawl_recall_tool():
    from mullet.core.knowledge.tools import make_trawl_recall
    km = FakeKM()
    tool = make_trawl_recall(km)
    result = tool(query="test")
    parsed = json.loads(result)
    assert len(parsed) >= 1
    assert parsed[0]["key"] == "test"


def test_register_knowledge_tools():
    from mullet.core.knowledge.tools import register
    from mullet.core.tools.registry import ToolRegistry
    km = FakeKM()
    registry = ToolRegistry()
    register(registry, km)
    assert "trawl_search" in registry.list_tools()
    assert "trawl_remember" in registry.list_tools()
    assert "trawl_recall" in registry.list_tools()
    schemas = registry.get_schemas(["trove"])
    assert len(schemas) == 4  # search, remember, recall, search_multi
