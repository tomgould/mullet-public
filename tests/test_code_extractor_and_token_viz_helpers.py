"""Tests for code_extractor pure helpers and token_viz pure helpers."""


from mullet.knowledge.engine.code_extractor import (
    CodeChunk,
    PythonCodeExtractor,
    register,
)
from mullet.core.agent.token_viz import TokenUsageViz


# ---------------------------------------------------------------------------
# CodeChunk
# ---------------------------------------------------------------------------

class TestCodeChunk:
    def test_fields_stored(self):
        c = CodeChunk("my_func", "function", "def my_func(): pass", 1, 3)
        assert c.name == "my_func"
        assert c.type == "function"
        assert c.content == "def my_func(): pass"
        assert c.start_line == 1
        assert c.end_line == 3

    def test_repr_contains_name(self):
        c = CodeChunk("my_func", "function", "code", 1, 2)
        assert "my_func" in repr(c)

    def test_repr_contains_type(self):
        c = CodeChunk("cls", "class", "code", 5, 10)
        assert "class" in repr(c)

    def test_repr_contains_start_line(self):
        c = CodeChunk("f", "function", "code", 42, 50)
        assert "42" in repr(c)


# ---------------------------------------------------------------------------
# PythonCodeExtractor
# ---------------------------------------------------------------------------

class TestPythonCodeExtractor:
    def setup_method(self):
        self.ext = PythonCodeExtractor()

    def test_empty_source_returns_empty(self):
        assert self.ext.extract("") == []

    def test_function_extracted(self):
        src = "def hello_world():\n    pass\n"
        chunks = self.ext.extract(src)
        funcs = [c for c in chunks if c.type == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "hello_world"

    def test_class_extracted(self):
        src = "class MyClass:\n    pass\n"
        chunks = self.ext.extract(src)
        classes = [c for c in chunks if c.type == "class"]
        assert len(classes) == 1
        assert classes[0].name == "MyClass"

    def test_import_extracted(self):
        src = "import os\n"
        chunks = self.ext.extract(src)
        imports = [c for c in chunks if c.type == "import"]
        assert len(imports) == 1

    def test_from_import_extracted(self):
        src = "from pathlib import Path\n"
        chunks = self.ext.extract(src)
        imports = [c for c in chunks if c.type == "import"]
        assert len(imports) == 1
        assert "pathlib" in imports[0].content

    def test_function_line_numbers_positive(self):
        src = "def foo():\n    pass\n"
        chunks = self.ext.extract(src)
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= 1

    def test_multiple_functions_extracted(self):
        src = "def foo():\n    pass\n\ndef bar():\n    return 1\n"
        chunks = self.ext.extract(src)
        funcs = [c for c in chunks if c.type == "function"]
        names = {c.name for c in funcs}
        assert "foo" in names
        assert "bar" in names

    def test_mixed_content_extracted(self):
        src = "import os\n\nclass A:\n    pass\n\ndef b():\n    pass\n"
        chunks = self.ext.extract(src)
        types = {c.type for c in chunks}
        assert "import" in types
        assert "class" in types
        assert "function" in types

    def test_content_preserved(self):
        src = "def my_func():\n    return 42\n"
        chunks = self.ext.extract(src)
        funcs = [c for c in chunks if c.type == "function"]
        assert "my_func" in funcs[0].content

    def test_returns_list(self):
        assert isinstance(self.ext.extract("x = 1"), list)


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_returns_dict(self):
        assert isinstance(register(), dict)

    def test_python_key_present(self):
        assert "python" in register()

    def test_python_value_is_class(self):
        assert register()["python"] is PythonCodeExtractor


# ---------------------------------------------------------------------------
# TokenUsageViz
# ---------------------------------------------------------------------------

class TestTokenUsageVizInit:
    def test_turns_empty_initially(self):
        viz = TokenUsageViz()
        assert viz.turns == []


class TestRecordTurn:
    def test_records_turn(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert len(viz.turns) == 1

    def test_stores_tokens_in(self):
        viz = TokenUsageViz()
        viz.record_turn(200, 75)
        assert viz.turns[0]["tokens_in"] == 200

    def test_stores_tokens_out(self):
        viz = TokenUsageViz()
        viz.record_turn(200, 75)
        assert viz.turns[0]["tokens_out"] == 75

    def test_cached_defaults_false(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert viz.turns[0]["cached"] is False

    def test_cached_true_stored(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=True)
        assert viz.turns[0]["cached"] is True

    def test_multiple_turns(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75, cached=True)
        assert len(viz.turns) == 2


class TestFormatTable:
    def test_empty_returns_no_usage_message(self):
        viz = TokenUsageViz()
        result = viz.format_table()
        assert "No token usage" in result

    def test_with_turns_returns_string(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert isinstance(viz.format_table(), str)

    def test_table_contains_tokens_in(self):
        viz = TokenUsageViz()
        viz.record_turn(1234, 567)
        assert "1234" in viz.format_table()

    def test_table_contains_tokens_out(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 999)
        assert "999" in viz.format_table()

    def test_cached_yes_in_table(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=True)
        assert "Yes" in viz.format_table()

    def test_cached_no_in_table(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=False)
        assert "No" in viz.format_table()

    def test_turn_number_in_table(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        table = viz.format_table()
        assert "2" in table


class TestTotal:
    def test_empty_returns_zeros(self):
        viz = TokenUsageViz()
        t = viz.total()
        assert t["total_tokens_in"] == 0
        assert t["total_tokens_out"] == 0

    def test_total_tokens_in(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        assert viz.total()["total_tokens_in"] == 300

    def test_total_tokens_out(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        assert viz.total()["total_tokens_out"] == 125

    def test_total_combined(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        t = viz.total()
        assert t["total_tokens_combined"] == 150

    def test_cached_totals(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=True)
        viz.record_turn(200, 75, cached=False)
        t = viz.total()
        assert t["total_cached_tokens_in"] == 100
        assert t["total_cached_tokens_out"] == 50

    def test_no_cached_turns(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=False)
        t = viz.total()
        assert t["total_cached_tokens_combined"] == 0

    def test_returns_dict(self):
        assert isinstance(TokenUsageViz().total(), dict)
