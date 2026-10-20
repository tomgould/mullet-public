import pytest
from pathlib import Path

from mullet.knowledge.engine.plugins.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseLLMProvider,
    BaseMergeStrategy,
    BaseParser,
    BaseReranker,
)


class DummyEmbedder(BaseEmbedder):
    name = "dummy-emb"
    dimension = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), float(i + 1)] for i in range(len(texts))]


class DummyParser(BaseParser):
    name = "dummy-parser"
    supported_extensions = [".txt", ".md"]

    def parse(self, path: Path) -> list:
        return []


class DummyReranker(BaseReranker):
    name = "dummy-rank"

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.5 for _ in pairs]


def test_embed_query_delegates():
    e = DummyEmbedder()
    result = e.embed_query("hello")
    assert result == [0.0, 1.0]


def test_embed_query_tracks_call():
    calls = []

    class Tracker(BaseEmbedder):
        name = "tracker"
        dimension = 1

        def embed(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return [[0.0] for _ in texts]

    t = Tracker()
    t.embed_query("single")
    assert calls == [["single"]]


def test_parser_can_parse_supported():
    p = DummyParser()
    assert p.can_parse(Path("file.txt")) is True
    assert p.can_parse(Path("file.md")) is True


def test_parser_can_parse_unsupported():
    p = DummyParser()
    assert p.can_parse(Path("file.pdf")) is False


def test_parser_can_parse_case_insensitive():
    p = DummyParser()
    assert p.can_parse(Path("file.TXT")) is True
    assert p.can_parse(Path("file.MD")) is True


def test_parser_can_parse_all_when_no_extensions():
    class Universal(BaseParser):
        name = "uni"
        supported_extensions = []

        def parse(self, path: Path) -> list:
            return []

    p = Universal()
    assert p.can_parse(Path("anything.xyz")) is True


def test_reranker_close_noop():
    r = DummyReranker()
    r.close()


@pytest.mark.parametrize("cls", [
    BaseEmbedder,
    BaseChunker,
    BaseParser,
    BaseReranker,
    BaseLLMProvider,
    BaseMergeStrategy,
])
def test_abstract_class_cannot_instantiate(cls):
    with pytest.raises(TypeError):
        cls()
