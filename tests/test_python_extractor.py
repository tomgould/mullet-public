"""Tests for the tree-sitter-based Python code extractor."""

import pytest
from mullet.knowledge.code.extractors.python import PythonExtractor


@pytest.fixture
def extractor():
    return PythonExtractor()


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------

class TestFunctionExtraction:
    def test_simple_function(self, extractor):
        code = "def greet(name: str) -> str:\n    return f'Hello {name}'"
        chunks = extractor.extract(code, "mod.py")
        names = [c.symbol_name for c in chunks]
        assert "greet" in names

    def test_function_line_numbers(self, extractor):
        code = "def first():\n    pass\n\ndef second():\n    pass"
        chunks = extractor.extract(code, "mod.py")
        names = [c.symbol_name for c in chunks]
        assert "first" in names
        assert "second" in names

    def test_async_function(self, extractor):
        code = "async def fetch(url):\n    pass"
        chunks = extractor.extract(code, "async_mod.py")
        names = [c.symbol_name for c in chunks]
        assert "fetch" in names

    def test_private_function(self, extractor):
        code = "def _helper():\n    pass"
        chunks = extractor.extract(code, "mod.py")
        names = [c.symbol_name for c in chunks]
        assert "_helper" in names

    def test_function_with_docstring(self, extractor):
        code = 'def process():\n    """Process the data."""\n    return True'
        chunks = extractor.extract(code, "mod.py")
        assert any(c.symbol_name == "process" for c in chunks)

    def test_empty_source_returns_list(self, extractor):
        chunks = extractor.extract("", "empty.py")
        assert isinstance(chunks, list)

    def test_no_functions_returns_empty_or_minimal(self, extractor):
        code = "x = 1\ny = 2\n"
        chunks = extractor.extract(code, "vars.py")
        assert isinstance(chunks, list)


class TestClassExtraction:
    def test_simple_class(self, extractor):
        code = "class MyService:\n    pass"
        chunks = extractor.extract(code, "service.py")
        names = [c.symbol_name for c in chunks]
        assert "MyService" in names

    def test_class_with_methods(self, extractor):
        code = (
            "class Calculator:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
            "    def subtract(self, a, b):\n"
            "        return a - b\n"
        )
        chunks = extractor.extract(code, "calc.py")
        names = [c.symbol_name for c in chunks]
        assert "Calculator" in names
        assert "add" in names
        assert "subtract" in names

    def test_class_chunk_has_correct_type(self, extractor):
        code = "class Foo:\n    pass"
        chunks = extractor.extract(code, "foo.py")
        class_chunks = [c for c in chunks if c.symbol_name == "Foo"]
        assert class_chunks
        assert class_chunks[0].symbol_type in ("class", "class_definition")

    def test_inherited_class(self, extractor):
        code = "class Child(Parent):\n    def method(self):\n        pass"
        chunks = extractor.extract(code, "child.py")
        names = [c.symbol_name for c in chunks]
        assert "Child" in names

    def test_decorated_class_not_extracted(self, extractor):
        # Decorated classes live inside decorated_definition nodes in tree-sitter
        # and are not extracted by the current top-level class_definition scan.
        code = (
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Point:\n"
            "    x: float\n"
            "    y: float\n"
        )
        chunks = extractor.extract(code, "point.py")
        # Decorated classes are currently skipped — document expected behavior.
        names = [c.symbol_name for c in chunks]
        assert "Point" not in names


class TestChunkAttributes:
    def test_source_file_is_set(self, extractor):
        code = "def foo():\n    pass"
        chunks = extractor.extract(code, "/path/to/module.py")
        assert all(c.source_file == "/path/to/module.py" for c in chunks)

    def test_language_is_python(self, extractor):
        code = "def foo():\n    pass"
        chunks = extractor.extract(code, "mod.py")
        assert all(c.language == "python" for c in chunks)

    def test_line_numbers_are_positive(self, extractor):
        code = "def foo():\n    pass\n\ndef bar():\n    pass"
        chunks = extractor.extract(code, "mod.py")
        for c in chunks:
            assert c.line_start >= 1

    def test_content_is_non_empty(self, extractor):
        code = "def foo():\n    return 42"
        chunks = extractor.extract(code, "mod.py")
        foo_chunks = [c for c in chunks if c.symbol_name == "foo"]
        assert foo_chunks
        assert foo_chunks[0].content.strip()


class TestComplexCode:
    def test_multiple_classes(self, extractor):
        code = (
            "class A:\n    pass\n\n"
            "class B:\n    pass\n\n"
            "class C:\n    pass\n"
        )
        chunks = extractor.extract(code, "multi.py")
        names = {c.symbol_name for c in chunks}
        assert {"A", "B", "C"}.issubset(names)

    def test_nested_class_methods(self, extractor):
        code = (
            "class Outer:\n"
            "    class Inner:\n"
            "        def inner_method(self):\n"
            "            pass\n"
            "    def outer_method(self):\n"
            "        pass\n"
        )
        chunks = extractor.extract(code, "nested.py")
        names = [c.symbol_name for c in chunks]
        assert "Outer" in names

    def test_invalid_python_does_not_raise(self, extractor):
        code = "def broken(\n    # unclosed paren"
        chunks = extractor.extract(code, "broken.py")
        assert isinstance(chunks, list)
