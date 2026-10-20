"""Tests for mullet.knowledge.engine.plugins.interfaces — ABC contracts."""
from __future__ import annotations

from pathlib import Path

import pytest

from mullet.knowledge.engine.plugins.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseLLMProvider,
    BaseMergeStrategy,
    BaseParser,
    BaseReranker,
)


# ---------------------------------------------------------------------------
# BaseEmbedder.embed_query default implementation
# ---------------------------------------------------------------------------

class ConcreteEmbedder(BaseEmbedder):
    name = "test-embedder"
    dimension = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class TestBaseEmbedder:
    def test_embed_query_delegates_to_embed(self) -> None:
        e = ConcreteEmbedder()
        result = e.embed_query("test query")
        assert result == [0.1, 0.2, 0.3, 0.4]

    def test_embed_query_passes_single_text(self) -> None:
        calls = []

        class TrackingEmbedder(BaseEmbedder):
            name = "tracker"
            dimension = 2

            def embed(self, texts: list[str]) -> list[list[float]]:
                calls.append(texts)
                return [[0.0, 0.0] for _ in texts]

        e = TrackingEmbedder()
        e.embed_query("hello")
        assert calls == [["hello"]]

    def test_close_is_noop(self) -> None:
        e = ConcreteEmbedder()
        e.close()  # Should not raise


# ---------------------------------------------------------------------------
# BaseParser.can_parse default implementation
# ---------------------------------------------------------------------------

class ConcreteParser(BaseParser):
    name = "test-parser"
    supported_extensions = [".pdf", ".docx"]

    def parse(self, path: Path) -> list:
        return []


class TestBaseParser:
    def test_can_parse_supported_extension(self) -> None:
        p = ConcreteParser()
        assert p.can_parse(Path("document.pdf")) is True
        assert p.can_parse(Path("file.docx")) is True

    def test_cannot_parse_unsupported_extension(self) -> None:
        p = ConcreteParser()
        assert p.can_parse(Path("file.txt")) is False
        assert p.can_parse(Path("code.py")) is False

    def test_case_insensitive_extension(self) -> None:
        p = ConcreteParser()
        assert p.can_parse(Path("file.PDF")) is True
        assert p.can_parse(Path("file.DOCX")) is True

    def test_no_extensions_list_allows_all(self) -> None:
        class UniversalParser(BaseParser):
            name = "universal"
            supported_extensions = []

            def parse(self, path: Path) -> list:
                return []

        p = UniversalParser()
        assert p.can_parse(Path("anything.xyz")) is True


# ---------------------------------------------------------------------------
# ABC enforcement — cannot instantiate abstract classes
# ---------------------------------------------------------------------------

class TestABCEnforcement:
    def test_base_embedder_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_base_chunker_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseChunker()

    def test_base_parser_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseParser()

    def test_base_reranker_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseReranker()

    def test_base_llm_provider_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_base_merge_strategy_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseMergeStrategy()


# ---------------------------------------------------------------------------
# BaseReranker.close default implementation
# ---------------------------------------------------------------------------

class ConcreteReranker(BaseReranker):
    name = "test-reranker"

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.5 for _ in pairs]


class TestBaseReranker:
    def test_close_is_noop(self) -> None:
        r = ConcreteReranker()
        r.close()  # Should not raise
