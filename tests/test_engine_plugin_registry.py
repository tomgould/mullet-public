import pytest

from mullet.knowledge.engine.plugins.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseLLMProvider,
    BaseMergeStrategy,
    BaseParser,
    BaseReranker,
)
from mullet.knowledge.engine.plugins.registry import PluginRegistry


class E1(BaseEmbedder):
    name = "e1"
    dimension = 2
    def embed(self, texts): return []


class E2(BaseEmbedder):
    name = "e1"
    dimension = 4
    def embed(self, texts): return []


class E3(BaseEmbedder):
    name = "a-emb"
    dimension = 1
    def embed(self, texts): return []


class C1(BaseChunker):
    name = "c1"
    def chunk(self, elements, max_chars=1500, overlap=100, **kwargs): return []


class P1(BaseParser):
    name = "p1"
    supported_extensions = [".txt"]
    def parse(self, path): return []


class R1(BaseReranker):
    name = "r1"
    def predict(self, pairs): return []


class L1(BaseLLMProvider):
    name = "l1"
    def generate(self, messages, system_prompt=None, max_tokens=2000): return ""


class M1(BaseMergeStrategy):
    name = "m1"
    def merge(self, bm25_results, vector_results, config): return []


def test_register_and_get_embedder():
    reg = PluginRegistry()
    reg.register_embedder(E1)
    assert reg.get_embedder("e1") is E1


def test_register_duplicate_overwrites():
    reg = PluginRegistry()
    reg.register_embedder(E1)
    reg.register_embedder(E2)
    assert reg.get_embedder("e1") is E2


def test_register_missing_name_raises():
    class Bad(BaseEmbedder):
        name = ""
        dimension = 1
        def embed(self, texts): return []
    reg = PluginRegistry()
    with pytest.raises(ValueError):
        reg.register_embedder(Bad)


def test_get_missing_returns_none():
    reg = PluginRegistry()
    assert reg.get_embedder("missing") is None
    assert reg.get_chunker("missing") is None
    assert reg.get_parser("missing") is None


def test_list_sorted():
    reg = PluginRegistry()
    reg.register_embedder(E3)
    reg.register_embedder(E1)
    assert reg.list_embedders() == ["a-emb", "e1"]


def test_list_all():
    reg = PluginRegistry()
    reg.register_embedder(E1)
    reg.register_chunker(C1)
    result = reg.list_all()
    assert result["embedders"] == ["e1"]
    assert result["chunkers"] == ["c1"]
    assert result["parsers"] == []
    assert result["rerankers"] == []
    assert result["llm_providers"] == []
    assert result["merge_strategies"] == []


def test_register_reranker():
    reg = PluginRegistry()
    reg.register_reranker(R1)
    assert reg.get_reranker("r1") is R1


def test_register_llm_and_merge():
    reg = PluginRegistry()
    reg.register_llm_provider(L1)
    reg.register_merge_strategy(M1)
    assert reg.get_llm_provider("l1") is L1
    assert reg.get_merge_strategy("m1") is M1
