from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.knowledge.code.parser import CodeParser


@pytest.fixture
def parser():
    return CodeParser()


def test_get_extractor_python(parser):
    assert parser.get_extractor(Path("foo.py")) is not None


def test_get_extractor_unsupported(parser):
    assert parser.get_extractor(Path("file.xyz")) is None


def test_get_supported_extensions_defaults(parser):
    exts = parser.get_supported_extensions()
    assert isinstance(exts, set)
    assert ".py" in exts


def test_get_supported_extensions_custom():
    parser = CodeParser({"lang": {"extensions": [".abc"]}})
    assert ".abc" in parser.get_supported_extensions()


def test_parse_file_no_extractor():
    parser = CodeParser()
    with patch.object(parser, "get_extractor", return_value=None):
        assert parser.parse_file(Path("a.unk")) == []


def test_parse_file_read_error():
    parser = CodeParser()
    mock_ext = MagicMock()
    with patch.object(parser, "get_extractor", return_value=mock_ext):
        with patch.object(Path, "read_text", side_effect=OSError("fail")):
            assert parser.parse_file(Path("a.py")) == []


def test_parse_file_extracts_chunks():
    parser = CodeParser()
    chunk = MagicMock()
    mock_ext = MagicMock()
    mock_ext.extract.return_value = [chunk]
    with patch.object(parser, "get_extractor", return_value=mock_ext):
        with patch.object(Path, "read_text", return_value="code"):
            assert parser.parse_file(Path("a.py")) == [chunk]


def test_parse_files_combines():
    parser = CodeParser()
    c1 = MagicMock()
    c2 = MagicMock()
    with patch.object(parser, "parse_file", side_effect=[[c1], [c2]]):
        assert parser.parse_files([Path("a.py"), Path("b.py")]) == [c1, c2]
