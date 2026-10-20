"""Tests for mullet.knowledge.code.extractors.base — coverage for missing lines."""
from __future__ import annotations

import pytest

from mullet.knowledge.code.extractors.base import BaseExtractor, CodeChunk


# ---------------------------------------------------------------------------
# Concrete extractor for testing (BaseExtractor is abstract)
# ---------------------------------------------------------------------------

class _ConcreteExtractor(BaseExtractor):
    language = "test"
    tree_sitter_language = "python"  # valid grammar for parser test

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        return []


@pytest.fixture
def ext() -> _ConcreteExtractor:
    return _ConcreteExtractor()


# ---------------------------------------------------------------------------
# CodeChunk.to_dict — optional fields that are currently uncovered
# ---------------------------------------------------------------------------

class TestCodeChunkToDictOptionalFields:
    """Lines 67, 73, 75, 77, 79 — optional fields only included when set."""

    def _base_chunk(self, **kwargs) -> CodeChunk:
        defaults = dict(
            content="x = 1",
            source_file="f.py",
            language="python",
            symbol_type="function",
            symbol_name="foo",
            line_start=1,
            line_end=1,
        )
        defaults.update(kwargs)
        return CodeChunk(**defaults)

    def test_parent_module_in_dict_when_set(self) -> None:
        # line 67
        chunk = self._base_chunk(parent_module="mymodule")
        d = chunk.to_dict()
        assert d["metadata"]["parent_module"] == "mymodule"

    def test_parent_module_absent_when_none(self) -> None:
        chunk = self._base_chunk()
        d = chunk.to_dict()
        assert "parent_module" not in d["metadata"]

    def test_repo_name_in_dict_when_set(self) -> None:
        # line 73
        chunk = self._base_chunk(repo_name="my-repo")
        d = chunk.to_dict()
        assert d["metadata"]["repo_name"] == "my-repo"

    def test_repo_remote_in_dict_when_set(self) -> None:
        # line 75
        chunk = self._base_chunk(repo_remote="https://github.com/org/repo")
        d = chunk.to_dict()
        assert d["metadata"]["repo_remote"] == "https://github.com/org/repo"

    def test_repo_branch_in_dict_when_set(self) -> None:
        # line 77
        chunk = self._base_chunk(repo_branch="main")
        d = chunk.to_dict()
        assert d["metadata"]["repo_branch"] == "main"

    def test_repo_relative_path_in_dict_when_set(self) -> None:
        # line 79
        chunk = self._base_chunk(repo_relative_path="src/f.py")
        d = chunk.to_dict()
        assert d["metadata"]["repo_relative_path"] == "src/f.py"

    def test_all_repo_fields_together(self) -> None:
        chunk = self._base_chunk(
            repo_name="r",
            repo_remote="https://x.com",
            repo_branch="dev",
            repo_relative_path="a/b.py",
        )
        meta = chunk.to_dict()["metadata"]
        assert meta["repo_name"] == "r"
        assert meta["repo_remote"] == "https://x.com"
        assert meta["repo_branch"] == "dev"
        assert meta["repo_relative_path"] == "a/b.py"

    def test_repo_fields_absent_when_none(self) -> None:
        chunk = self._base_chunk()
        meta = chunk.to_dict()["metadata"]
        for key in ("repo_name", "repo_remote", "repo_branch", "repo_relative_path"):
            assert key not in meta


# ---------------------------------------------------------------------------
# BaseExtractor.extract_docstring — line 167 (returns None by default)
# ---------------------------------------------------------------------------

class TestExtractDocstring:
    def test_extract_docstring_returns_none(self, ext: _ConcreteExtractor) -> None:
        # line 167 — default implementation always returns None
        result = ext.extract_docstring(object(), b"hello")
        assert result is None


# ---------------------------------------------------------------------------
# BaseExtractor.determine_visibility — lines 175-179
# ---------------------------------------------------------------------------

class TestDetermineVisibility:
    def test_dunder_name_is_protected(self, ext: _ConcreteExtractor) -> None:
        # __init__ ends with __ so the first branch is False, falls to elif startswith("_") → protected
        assert ext.determine_visibility("__init__") == "protected"

    def test_double_underscore_no_suffix_is_private(self, ext: _ConcreteExtractor) -> None:
        assert ext.determine_visibility("__secret") == "private"

    def test_single_underscore_is_protected(self, ext: _ConcreteExtractor) -> None:
        assert ext.determine_visibility("_helper") == "protected"

    def test_plain_name_is_public(self, ext: _ConcreteExtractor) -> None:
        assert ext.determine_visibility("my_func") == "public"


# ---------------------------------------------------------------------------
# CodeChunk.to_dict — remaining optional fields (lines 61, 63, 65, 69)
# ---------------------------------------------------------------------------

class TestCodeChunkToDictSignatureDocstringDecoratos:
    def _base_chunk(self, **kwargs) -> CodeChunk:
        defaults = dict(
            content="x = 1",
            source_file="f.py",
            language="python",
            symbol_type="function",
            symbol_name="foo",
            line_start=1,
            line_end=1,
        )
        defaults.update(kwargs)
        return CodeChunk(**defaults)

    def test_signature_in_dict_when_set(self) -> None:
        # line 61
        chunk = self._base_chunk(signature="def foo(x: int) -> str")
        assert chunk.to_dict()["metadata"]["signature"] == "def foo(x: int) -> str"

    def test_docstring_in_dict_when_set(self) -> None:
        # line 63
        chunk = self._base_chunk(docstring="Does something.")
        assert chunk.to_dict()["metadata"]["docstring"] == "Does something."

    def test_parent_class_in_dict_when_set(self) -> None:
        # line 65
        chunk = self._base_chunk(parent_class="MyClass")
        assert chunk.to_dict()["metadata"]["parent_class"] == "MyClass"

    def test_decorators_in_dict_when_set(self) -> None:
        # line 69
        chunk = self._base_chunk(decorators=["@staticmethod"])
        assert chunk.to_dict()["metadata"]["decorators"] == ["@staticmethod"]

    def test_empty_decorators_not_in_dict(self) -> None:
        chunk = self._base_chunk()
        assert "decorators" not in chunk.to_dict()["metadata"]


# ---------------------------------------------------------------------------
# BaseExtractor.find_child_by_type — line 145 (returns None when not found)
# ---------------------------------------------------------------------------

class TestFindChildByType:
    def test_returns_none_when_not_found(self, ext: _ConcreteExtractor) -> None:
        # line 145 — no children match → return None
        class _FakeNode:
            children = []

        result = ext.find_child_by_type(_FakeNode(), "missing_type")
        assert result is None

    def test_returns_match_when_found(self, ext: _ConcreteExtractor) -> None:
        class _FakeChild:
            type = "target"

        class _FakeNode:
            children = [_FakeChild()]

        result = ext.find_child_by_type(_FakeNode(), "target")
        assert result is _FakeNode.children[0]
