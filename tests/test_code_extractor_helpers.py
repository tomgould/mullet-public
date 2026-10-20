"""Tests for mullet.knowledge.engine.code_extractor — PythonCodeExtractor."""

from mullet.knowledge.engine.code_extractor import CodeChunk, PythonCodeExtractor, register


# ---------------------------------------------------------------------------
# CodeChunk dataclass
# ---------------------------------------------------------------------------

class TestCodeChunk:
    def test_stores_attributes(self):
        c = CodeChunk("my_func", "function", "def my_func(): pass", 1, 1)
        assert c.name == "my_func"
        assert c.type == "function"
        assert c.content == "def my_func(): pass"
        assert c.start_line == 1
        assert c.end_line == 1

    def test_repr_contains_name(self):
        c = CodeChunk("foo", "function", "def foo(): pass", 3, 5)
        assert "foo" in repr(c)

    def test_repr_contains_type(self):
        c = CodeChunk("Bar", "class", "class Bar: pass", 10, 15)
        assert "class" in repr(c)

    def test_repr_contains_start_line(self):
        c = CodeChunk("x", "function", "def x(): pass", 42, 50)
        assert "42" in repr(c)


# ---------------------------------------------------------------------------
# PythonCodeExtractor.extract — functions
# ---------------------------------------------------------------------------

class TestExtractFunctions:
    def setup_method(self):
        self.extractor = PythonCodeExtractor()

    def test_empty_source_returns_empty(self):
        assert self.extractor.extract("") == []

    def test_simple_function_extracted(self):
        source = "def hello():\n    pass\n"
        chunks = self.extractor.extract(source)
        funcs = [c for c in chunks if c.type == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "hello"

    def test_function_content_includes_body(self):
        source = "def greet(name):\n    return f'Hello {name}'\n"
        chunks = self.extractor.extract(source)
        funcs = [c for c in chunks if c.type == "function"]
        assert "greet" in funcs[0].content

    def test_function_line_numbers(self):
        source = "def foo():\n    pass\n"
        chunks = self.extractor.extract(source)
        funcs = [c for c in chunks if c.type == "function"]
        assert funcs[0].start_line == 1

    def test_multiple_functions(self):
        source = "def a():\n    pass\n\ndef b():\n    pass\n"
        chunks = self.extractor.extract(source)
        funcs = [c for c in chunks if c.type == "function"]
        names = {f.name for f in funcs}
        assert "a" in names
        assert "b" in names

    def test_function_type_is_function(self):
        source = "def my_function():\n    pass\n"
        chunks = self.extractor.extract(source)
        assert all(c.type == "function" for c in chunks if c.name == "my_function")


# ---------------------------------------------------------------------------
# PythonCodeExtractor.extract — classes
# ---------------------------------------------------------------------------

class TestExtractClasses:
    def setup_method(self):
        self.extractor = PythonCodeExtractor()

    def test_simple_class_extracted(self):
        source = "class MyClass:\n    pass\n"
        chunks = self.extractor.extract(source)
        classes = [c for c in chunks if c.type == "class"]
        assert len(classes) == 1
        assert classes[0].name == "MyClass"

    def test_class_type_is_class(self):
        source = "class Foo:\n    pass\n"
        chunks = self.extractor.extract(source)
        assert any(c.type == "class" for c in chunks)

    def test_class_content_non_empty(self):
        source = "class Bar:\n    x = 1\n"
        chunks = self.extractor.extract(source)
        classes = [c for c in chunks if c.type == "class"]
        assert len(classes[0].content) > 0

    def test_class_and_function_both_extracted(self):
        source = "class MyClass:\n    pass\n\ndef standalone():\n    pass\n"
        chunks = self.extractor.extract(source)
        types = {c.type for c in chunks}
        assert "class" in types
        assert "function" in types


# ---------------------------------------------------------------------------
# PythonCodeExtractor.extract — imports
# ---------------------------------------------------------------------------

class TestExtractImports:
    def setup_method(self):
        self.extractor = PythonCodeExtractor()

    def test_import_statement_extracted(self):
        source = "import os\n"
        chunks = self.extractor.extract(source)
        imports = [c for c in chunks if c.type == "import"]
        assert len(imports) == 1

    def test_from_import_extracted(self):
        source = "from pathlib import Path\n"
        chunks = self.extractor.extract(source)
        imports = [c for c in chunks if c.type == "import"]
        assert len(imports) == 1

    def test_import_content_has_module(self):
        source = "import json\n"
        chunks = self.extractor.extract(source)
        imports = [c for c in chunks if c.type == "import"]
        assert "json" in imports[0].content

    def test_import_name_has_line_number(self):
        source = "import sys\n"
        chunks = self.extractor.extract(source)
        imports = [c for c in chunks if c.type == "import"]
        assert imports[0].name.startswith("import_")


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------

class TestRegister:
    def test_returns_dict(self):
        assert isinstance(register(), dict)

    def test_python_key_present(self):
        assert "python" in register()

    def test_python_value_is_class(self):
        assert register()["python"] is PythonCodeExtractor
