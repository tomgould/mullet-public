"""Tests for swarm_types dataclasses and TypeScriptExtractor pure helpers."""

import pytest

from mullet.core.orchestration.swarm_types import (
    SwarmTask,
    SwarmContext,
    TaskResult,
    SwarmResult,
)
from mullet.knowledge.code.extractors.typescript import TypeScriptExtractor


# ---------------------------------------------------------------------------
# SwarmTask
# ---------------------------------------------------------------------------

class TestSwarmTask:
    def test_fields_stored(self):
        t = SwarmTask(id="t1", instruction="Do the thing")
        assert t.id == "t1"
        assert t.instruction == "Do the thing"

    def test_defaults(self):
        t = SwarmTask(id="t1", instruction="x")
        assert t.context_refs == []
        assert t.expected_output_shape == "markdown report"
        assert t.apply is False
        assert t.target_dir == ""


# ---------------------------------------------------------------------------
# SwarmContext
# ---------------------------------------------------------------------------

class TestSwarmContext:
    def test_defaults(self):
        ctx = SwarmContext()
        assert ctx.files == []
        assert ctx.items == []
        assert ctx.concurrency == 20

    def test_valid_files(self):
        ctx = SwarmContext(files=["a.py", "b.py"])
        assert ctx.files == ["a.py", "b.py"]

    def test_invalid_files_type_raises(self):
        with pytest.raises(TypeError):
            SwarmContext(files="not_a_list")

    def test_non_string_files_element_raises(self):
        with pytest.raises(TypeError):
            SwarmContext(files=[123])

    def test_invalid_items_type_raises(self):
        with pytest.raises(TypeError):
            SwarmContext(items="not_a_list")


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_fields_stored(self):
        r = TaskResult(task_id="t1", success=True, response="done", error=None)
        assert r.task_id == "t1"
        assert r.success is True
        assert r.response == "done"
        assert r.error is None

    def test_elapsed_defaults_zero(self):
        r = TaskResult(task_id="t1", success=True, response="x", error=None)
        assert r.elapsed_seconds == 0.0

    def test_files_changed_defaults_empty(self):
        r = TaskResult(task_id="t1", success=True, response="x", error=None)
        assert r.files_changed == []


# ---------------------------------------------------------------------------
# SwarmResult
# ---------------------------------------------------------------------------

class TestSwarmResult:
    def test_fields_stored(self):
        r = SwarmResult(goal="Fix everything")
        assert r.goal == "Fix everything"

    def test_defaults(self):
        r = SwarmResult(goal="g")
        assert r.tasks == []
        assert r.results == []
        assert r.synthesis == ""
        assert r.fell_back is False
        assert r.wall_clock_seconds == 0.0


# ---------------------------------------------------------------------------
# TypeScriptExtractor
# ---------------------------------------------------------------------------

class TestTypeScriptExtractor:
    def setup_method(self):
        self.e = TypeScriptExtractor()

    def test_language_is_typescript(self):
        assert self.e.language == "typescript"

    def test_empty_source_returns_list(self):
        result = self.e.extract("", "empty.ts")
        assert isinstance(result, list)

    def test_interface_extracted(self):
        src = "interface User {\n  name: string;\n  age: number;\n}\n"
        chunks = self.e.extract(src, "user.ts")
        interfaces = [c for c in chunks if c.symbol_type == "interface"]
        assert len(interfaces) == 1
        assert interfaces[0].symbol_name == "User"

    def test_type_alias_extracted(self):
        src = "type ID = string | number;\n"
        chunks = self.e.extract(src, "types.ts")
        aliases = [c for c in chunks if c.symbol_type == "type_alias"]
        assert len(aliases) == 1
        assert aliases[0].symbol_name == "ID"

    def test_enum_extracted(self):
        src = "enum Status {\n  Active = 'ACTIVE',\n  Inactive = 'INACTIVE',\n}\n"
        chunks = self.e.extract(src, "status.ts")
        enums = [c for c in chunks if c.symbol_type == "enum"]
        assert len(enums) == 1
        assert enums[0].symbol_name == "Status"

    def test_chunks_have_source_file(self):
        src = "interface Foo {\n  x: number;\n}\n"
        chunks = self.e.extract(src, "foo.ts")
        assert all(c.source_file == "foo.ts" for c in chunks)

    def test_multiple_types_in_file(self):
        src = (
            "interface User {\n  name: string;\n}\n"
            "type ID = string;\n"
            "enum Status {\n  Active,\n}\n"
        )
        chunks = self.e.extract(src, "multi.ts")
        types_found = {c.symbol_type for c in chunks}
        assert "interface" in types_found
        assert "type_alias" in types_found
        assert "enum" in types_found

    def test_returns_list(self):
        assert isinstance(self.e.extract("const x = 1;", "x.ts"), list)

    def test_line_start_positive(self):
        src = "interface Foo {\n  x: number;\n}\n"
        chunks = self.e.extract(src, "foo.ts")
        for c in chunks:
            assert c.line_start >= 1
