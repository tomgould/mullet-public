"""Tests for lsp module pure helpers (no subprocess or LSP connections)."""

import json

from mullet.core.tools.lsp import (
    LSP_SCHEMA,
    MAX_DIAGNOSTICS,
    _parse_text_diagnostics,
    _regex_symbols,
    _find_project_root,
    _extract_symbol,
    _parse_pyright_output,
)


# ---------------------------------------------------------------------------
# LSP_SCHEMA constant
# ---------------------------------------------------------------------------

class TestLSPSchema:
    def test_has_description(self):
        assert "description" in LSP_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in LSP_SCHEMA

    def test_action_in_properties(self):
        assert "action" in LSP_SCHEMA["parameters"]["properties"]

    def test_path_in_properties(self):
        assert "path" in LSP_SCHEMA["parameters"]["properties"]

    def test_action_required(self):
        assert "action" in LSP_SCHEMA["parameters"]["required"]

    def test_action_enum_has_diagnostics(self):
        enum = LSP_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "diagnostics" in enum

    def test_action_enum_has_definition(self):
        enum = LSP_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "definition" in enum

    def test_action_enum_has_references(self):
        enum = LSP_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "references" in enum

    def test_action_enum_has_hover(self):
        enum = LSP_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "hover" in enum


class TestMaxDiagnostics:
    def test_is_positive(self):
        assert MAX_DIAGNOSTICS > 0

    def test_is_int(self):
        assert isinstance(MAX_DIAGNOSTICS, int)


# ---------------------------------------------------------------------------
# _parse_text_diagnostics
# ---------------------------------------------------------------------------

class TestParseTextDiagnostics:
    def test_returns_json_string(self):
        result = _parse_text_diagnostics("", "foo.py", "mypy")
        json.loads(result)

    def test_empty_output_empty_diagnostics(self):
        result = json.loads(_parse_text_diagnostics("", "foo.py", "mypy"))
        assert result["diagnostics"] == []

    def test_tool_preserved(self):
        result = json.loads(_parse_text_diagnostics("", "foo.py", "mypy"))
        assert result["tool"] == "mypy"

    def test_error_line_parsed(self):
        output = "foo.py:10:5: error: undefined name"
        result = json.loads(_parse_text_diagnostics(output, "foo.py", "mypy"))
        assert len(result["diagnostics"]) >= 1

    def test_error_severity(self):
        output = "foo.py:10:5: error: undefined name"
        result = json.loads(_parse_text_diagnostics(output, "foo.py", "mypy"))
        assert result["diagnostics"][0]["severity"] == "error"

    def test_warning_severity(self):
        output = "foo.py:10:5: warning: unused variable"
        result = json.loads(_parse_text_diagnostics(output, "foo.py", "mypy"))
        assert result["diagnostics"][0]["severity"] == "warning"

    def test_line_number_parsed(self):
        output = "foo.py:42:3: error: bad"
        result = json.loads(_parse_text_diagnostics(output, "foo.py", "mypy"))
        assert result["diagnostics"][0]["line"] == 42

    def test_column_number_parsed(self):
        output = "foo.py:42:7: error: bad"
        result = json.loads(_parse_text_diagnostics(output, "foo.py", "mypy"))
        assert result["diagnostics"][0]["column"] == 7

    def test_multiple_lines_parsed(self):
        output = "foo.py:10:5: error: err1\nfoo.py:20:1: warning: warn1"
        result = json.loads(_parse_text_diagnostics(output, "foo.py", "mypy"))
        assert len(result["diagnostics"]) == 2

    def test_blank_lines_skipped(self):
        output = "foo.py:10:5: error: err\n\n\nfoo.py:20:1: warning: warn"
        result = json.loads(_parse_text_diagnostics(output, "foo.py", "mypy"))
        assert len(result["diagnostics"]) == 2


# ---------------------------------------------------------------------------
# _regex_symbols
# ---------------------------------------------------------------------------

class TestRegexSymbols:
    def test_python_function_extracted(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def hello_world():\n    pass\n")
        result = json.loads(_regex_symbols(str(f), ".py"))
        names = [s["name"] for s in result["symbols"]]
        assert "hello_world" in names

    def test_python_class_extracted(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("class MyClass:\n    pass\n")
        result = json.loads(_regex_symbols(str(f), ".py"))
        names = [s["name"] for s in result["symbols"]]
        assert "MyClass" in names

    def test_multiple_symbols_extracted(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")
        result = json.loads(_regex_symbols(str(f), ".py"))
        names = [s["name"] for s in result["symbols"]]
        assert "foo" in names
        assert "bar" in names

    def test_line_number_positive(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n")
        result = json.loads(_regex_symbols(str(f), ".py"))
        assert result["symbols"][0]["line"] >= 1

    def test_unsupported_ext_returns_error(self, tmp_path):
        f = tmp_path / "test.ruby"
        f.write_text("def hello; end\n")
        result = json.loads(_regex_symbols(str(f), ".ruby"))
        assert "error" in result

    def test_returns_json_string(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo(): pass\n")
        result = _regex_symbols(str(f), ".py")
        assert isinstance(result, str)
        json.loads(result)


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------

class TestFindProjectRoot:
    def test_returns_string(self, tmp_path):
        assert isinstance(_find_project_root(str(tmp_path)), str)

    def test_finds_git_root(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        sub = tmp_path / "src" / "module"
        sub.mkdir(parents=True)
        f = sub / "file.py"
        f.write_text("x = 1\n")
        root = _find_project_root(str(f))
        assert root == str(tmp_path)

    def test_finds_pyproject_root(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
        sub = tmp_path / "src"
        sub.mkdir()
        f = sub / "module.py"
        f.write_text("x = 1\n")
        root = _find_project_root(str(f))
        assert root == str(tmp_path)

    def test_no_markers_returns_parent(self, tmp_path):
        # No markers → should still return a valid path
        f = tmp_path / "isolated.py"
        f.write_text("x = 1\n")
        root = _find_project_root(str(f))
        assert isinstance(root, str)


# ---------------------------------------------------------------------------
# _extract_symbol
# ---------------------------------------------------------------------------

class TestExtractSymbol:
    def test_extracts_word_at_position(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def my_function():\n    pass\n")
        sym = _extract_symbol(str(f), 1, 5)
        assert isinstance(sym, str)
        assert len(sym) > 0

    def test_out_of_range_line_returns_empty(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        assert _extract_symbol(str(f), 999, 1) == ""

    def test_zero_line_returns_empty(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        assert _extract_symbol(str(f), 0, 1) == ""

    def test_missing_file_returns_empty(self):
        assert _extract_symbol("/nonexistent/file.py", 1, 1) == ""

    def test_returns_string(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        assert isinstance(_extract_symbol(str(f), 1, 1), str)


# ---------------------------------------------------------------------------
# _parse_pyright_output
# ---------------------------------------------------------------------------

class FakeResult:
    def __init__(self, stdout, stderr=""):
        self.stdout = stdout
        self.stderr = stderr


class TestParsePyrightOutput:
    def test_valid_json_returns_diagnostics(self):
        data = {
            "generalDiagnostics": [
                {
                    "severity": "error",
                    "message": "Type mismatch",
                    "file": "foo.py",
                    "range": {"start": {"line": 5, "character": 3}},
                    "rule": "reportMissingImports",
                }
            ],
            "summary": {"errorCount": 1},
        }
        result = json.loads(_parse_pyright_output(FakeResult(json.dumps(data)), "foo.py"))
        assert len(result["diagnostics"]) == 1

    def test_tool_is_pyright(self):
        data = {"generalDiagnostics": [], "summary": {}}
        result = json.loads(_parse_pyright_output(FakeResult(json.dumps(data)), "foo.py"))
        assert result["tool"] == "pyright"

    def test_severity_preserved(self):
        data = {
            "generalDiagnostics": [
                {"severity": "warning", "message": "x", "file": "f.py",
                 "range": {"start": {"line": 1, "character": 0}}, "rule": ""}
            ],
            "summary": {},
        }
        result = json.loads(_parse_pyright_output(FakeResult(json.dumps(data)), "f.py"))
        assert result["diagnostics"][0]["severity"] == "warning"

    def test_invalid_json_fallback_to_text_parse(self):
        result = json.loads(
            _parse_pyright_output(FakeResult("not json\nfoo.py:5:3: error: bad"), "foo.py")
        )
        assert result["tool"] == "pyright"

    def test_empty_diagnostics_returns_empty_list(self):
        data = {"generalDiagnostics": [], "summary": {}}
        result = json.loads(_parse_pyright_output(FakeResult(json.dumps(data)), "foo.py"))
        assert result["diagnostics"] == []
