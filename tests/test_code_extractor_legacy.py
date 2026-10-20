"""Tests for mullet.knowledge.engine.code_extractor (legacy PythonCodeExtractor)."""
from __future__ import annotations


from mullet.knowledge.engine.code_extractor import CodeChunk, PythonCodeExtractor, register


_SIMPLE_PY = """\
def greet(name):
    return f'Hello, {name}!'

def farewell(name):
    return f'Goodbye, {name}!'
"""

_CLASS_PY = """\
class Animal:
    pass

class Dog(Animal):
    pass
"""

_IMPORT_PY = """\
import os
from pathlib import Path
"""

_MIXED_PY = """\
import sys

def helper():
    pass

class Manager:
    pass
"""


class TestCodeChunk:
    def test_repr_contains_name(self) -> None:
        c = CodeChunk("my_func", "function", "def my_func(): pass", 1, 1)
        assert "my_func" in repr(c)

    def test_repr_contains_type(self) -> None:
        c = CodeChunk("Foo", "class", "class Foo: pass", 3, 5)
        assert "class" in repr(c)

    def test_repr_contains_line_numbers(self) -> None:
        c = CodeChunk("fn", "function", "def fn(): pass", 10, 12)
        r = repr(c)
        assert "10" in r
        assert "12" in r

    def test_attributes_stored(self) -> None:
        c = CodeChunk("x", "import", "import x", 7, 7)
        assert c.name == "x"
        assert c.type == "import"
        assert c.content == "import x"
        assert c.start_line == 7
        assert c.end_line == 7


class TestPythonCodeExtractor:
    def test_extracts_functions(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract(_SIMPLE_PY)
        func_names = [c.name for c in chunks if c.type == "function"]
        assert "greet" in func_names
        assert "farewell" in func_names

    def test_extracts_classes(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract(_CLASS_PY)
        class_names = [c.name for c in chunks if c.type == "class"]
        assert "Animal" in class_names
        assert "Dog" in class_names

    def test_extracts_imports(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract(_IMPORT_PY)
        import_chunks = [c for c in chunks if c.type == "import"]
        assert len(import_chunks) == 2

    def test_import_chunk_name_includes_line(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract(_IMPORT_PY)
        names = [c.name for c in chunks if c.type == "import"]
        assert any("import_" in n for n in names)

    def test_mixed_file_extracts_all_types(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract(_MIXED_PY)
        types = {c.type for c in chunks}
        assert "function" in types
        assert "class" in types
        assert "import" in types

    def test_empty_source_returns_empty(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract("")
        assert chunks == []

    def test_chunk_content_is_source_text(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract(_SIMPLE_PY)
        greet = next(c for c in chunks if c.name == "greet")
        assert "def greet" in greet.content

    def test_line_numbers_are_one_based(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract(_SIMPLE_PY)
        greet = next(c for c in chunks if c.name == "greet")
        assert greet.start_line >= 1

    def test_function_end_line_gte_start_line(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract(_MIXED_PY)
        for c in chunks:
            assert c.end_line >= c.start_line

    def test_returns_list_of_codechunk(self) -> None:
        ext = PythonCodeExtractor()
        chunks = ext.extract(_SIMPLE_PY)
        assert isinstance(chunks, list)
        assert all(isinstance(c, CodeChunk) for c in chunks)


class TestRegister:
    def test_register_returns_dict(self) -> None:
        result = register()
        assert isinstance(result, dict)

    def test_register_maps_python_to_extractor(self) -> None:
        result = register()
        assert "python" in result
        assert result["python"] is PythonCodeExtractor
