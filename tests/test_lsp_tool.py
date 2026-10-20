"""Tests for the LSP integration tool."""

import json
import subprocess
import sys
from unittest.mock import patch, MagicMock

from mullet.core.tools.lsp import (
    handle_lsp,
    _find_project_root,
    _extract_symbol,
    _regex_symbols,
    _parse_text_diagnostics,
    _parse_pyright_output,
    _parse_mypy_output,
    _parse_tsc_output,
    register,
    _diagnostics,
    _python_diagnostics,
    _typescript_diagnostics,
    _go_diagnostics,
    _definition,
    _references,
    _hover,
    _symbols,
    _try_tool,
    _grep_definition,
)


def test_handle_lsp_file_not_found():
    result = json.loads(handle_lsp("diagnostics", "/nonexistent/file.py"))
    assert "error" in result
    assert "not found" in result["error"]


def test_handle_lsp_unknown_action(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")
    result = json.loads(handle_lsp("invalid", str(f)))
    assert "error" in result
    assert "Unknown LSP action" in result["error"]


def test_find_project_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    sub = tmp_path / "src"
    sub.mkdir()
    f = sub / "test.py"
    f.write_text("")
    root = _find_project_root(str(f))
    assert root == str(tmp_path)


def test_find_project_root_no_marker(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("")
    root = _find_project_root(str(f))
    assert root == str(tmp_path)


def test_extract_symbol(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def hello_world():\n    pass\n")
    sym = _extract_symbol(str(f), 1, 5)
    assert "hello" in sym or "def" in sym


def test_extract_symbol_bad_line(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")
    assert _extract_symbol(str(f), 999, 1) == ""


def test_regex_symbols_python(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("class Foo:\n    pass\n\ndef bar():\n    pass\n\nasync def baz():\n    pass\n")
    result = json.loads(_regex_symbols(str(f), ".py"))
    names = [s["name"] for s in result["symbols"]]
    assert "Foo" in names
    assert "bar" in names


def test_regex_symbols_unsupported(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello\n")
    result = json.loads(_regex_symbols(str(f), ".txt"))
    assert "error" in result


def test_parse_text_diagnostics():
    output = "test.py:10:5: error: Name 'x' is not defined\ntest.py:20:1: note: See docs"
    result = json.loads(_parse_text_diagnostics(output, "test.py", "mypy"))
    assert len(result["diagnostics"]) >= 1
    assert result["tool"] == "mypy"


def test_parse_text_diagnostics_empty():
    result = json.loads(_parse_text_diagnostics("", "test.py", "pyright"))
    assert result["diagnostics"] == []


@patch("mullet.core.tools.lsp._try_tool")
def test_diagnostics_py_compile_fallback(mock_tool, tmp_path):
    """Falls back to py_compile when pyright and mypy unavailable."""
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")

    # pyright not found, mypy not found, py_compile succeeds
    mock_tool.side_effect = [None, None, MagicMock(returncode=0, stderr="")]
    result = json.loads(handle_lsp("diagnostics", str(f)))
    assert result["tool"] == "py_compile"
    assert result["diagnostics"] == []


@patch("mullet.core.tools.lsp._try_tool")
def test_diagnostics_no_tools(mock_tool, tmp_path):
    """Returns error when no diagnostic tools available."""
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")
    mock_tool.return_value = None
    result = json.loads(handle_lsp("diagnostics", str(f)))
    assert "error" in result


def test_references_needs_symbol(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")
    result = json.loads(handle_lsp("references", str(f)))
    assert "error" in result


def test_definition_needs_line(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")
    result = json.loads(handle_lsp("definition", str(f)))
    assert "error" in result


def test_hover_needs_line(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")
    result = json.loads(handle_lsp("hover", str(f)))
    assert "error" in result


def test_register():
    registry = MagicMock()
    register(registry)
    registry.register.assert_called_once()
    call_kwargs = registry.register.call_args
    assert call_kwargs[1]["name"] == "lsp" or call_kwargs[0][0] == "lsp"


def test_symbols_action(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("class MyClass:\n    pass\n\ndef my_func():\n    pass\n")
    result = json.loads(handle_lsp("symbols", str(f)))
    assert "symbols" in result
    names = [s["name"] for s in result["symbols"]]
    assert "MyClass" in names
    assert "my_func" in names


def _make_completed(stdout="", stderr="", returncode=0):
    import subprocess
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_parse_pyright_output_valid_json():
    result = _make_completed(json.dumps({
        "generalDiagnostics": [
            {"severity": "error", "message": "Type mismatch",
             "range": {"start": {"line": 5, "character": 10}}, "rule": "reportAttributeAccessIssue"}
        ],
        "summary": {"filesAnalyzed": 1}
    }))
    out = json.loads(_parse_pyright_output(result, "test.py"))
    assert out["tool"] == "pyright"
    assert len(out["diagnostics"]) == 1
    assert out["diagnostics"][0]["severity"] == "error"
    assert out["diagnostics"][0]["line"] == 5


def test_parse_pyright_output_invalid_json_falls_back():
    result = _make_completed(stdout="test.py:10:1: error: bad type")
    out = json.loads(_parse_pyright_output(result, "test.py"))
    assert out["tool"] == "pyright"


def test_parse_mypy_output():
    result = _make_completed(stdout="test.py:5:1: error: Incompatible types\n")
    out = json.loads(_parse_mypy_output(result, "test.py"))
    assert out["tool"] == "mypy"
    assert len(out["diagnostics"]) >= 1


def test_parse_tsc_output():
    result = _make_completed(stdout="test.ts(10,5): error TS2345: bad arg\n")
    out = json.loads(_parse_tsc_output(result, "test.ts"))
    assert out["tool"] == "tsc"


class TestLspMissingLines:
    """Cover missing lines in mullet/core/tools/lsp.py."""

    # Lines 80-82: handle_lsp exception raised by handler
    @patch("mullet.core.tools.lsp._python_diagnostics")
    def test_handle_lsp_exception(self, mock_pd, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        mock_pd.side_effect = RuntimeError("boom")
        result = json.loads(handle_lsp("diagnostics", str(f)))
        assert "error" in result
        assert "RuntimeError" in result["error"]

    # Lines 91-97: _diagnostics routing by extension
    def test_diagnostics_ts(self, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("let x = 1;\n")
        with patch("mullet.core.tools.lsp._typescript_diagnostics", return_value='{"tool":"tsc"}') as mock_td:
            result = _diagnostics(str(f), 0, 0, "")
            assert json.loads(result)["tool"] == "tsc"

    def test_diagnostics_go(self, tmp_path):
        f = tmp_path / "test.go"
        f.write_text("package main\n")
        with patch("mullet.core.tools.lsp._go_diagnostics", return_value='{"tool":"go"}') as mock_gd:
            result = _diagnostics(str(f), 0, 0, "")
            assert json.loads(result)["tool"] == "go"

    def test_diagnostics_unknown_ext(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello\n")
        with patch("mullet.core.tools.lsp._python_diagnostics", return_value='{"tool":"py"}') as mock_pd:
            result = _diagnostics(str(f), 0, 0, "")
            assert json.loads(result)["tool"] == "py"

    # Line 110: pyright found in _python_diagnostics
    @patch("mullet.core.tools.lsp._try_tool")
    def test_python_diagnostics_pyright(self, mock_tool, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        mock_tool.side_effect = [
            MagicMock(returncode=0, stdout='{"generalDiagnostics":[],"summary":{}}', stderr=""),
            None,
            None,
        ]
        result = json.loads(_python_diagnostics(str(f)))
        assert result["tool"] == "pyright"

    # Line 118: mypy found in _python_diagnostics
    @patch("mullet.core.tools.lsp._try_tool")
    def test_python_diagnostics_mypy(self, mock_tool, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        mock_tool.side_effect = [
            None,
            MagicMock(returncode=0, stdout="test.py:1:1: error: bad\n", stderr=""),
            None,
        ]
        result = json.loads(_python_diagnostics(str(f)))
        assert result["tool"] == "mypy"

    # Line 128: py_compile returns non-zero (syntax error)
    @patch("mullet.core.tools.lsp._try_tool")
    def test_python_diagnostics_pycompile_error(self, mock_tool, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("bad syntax!!!\n")
        mock_tool.side_effect = [
            None,
            None,
            MagicMock(returncode=1, stdout="", stderr="SyntaxError: invalid syntax"),
        ]
        result = json.loads(_python_diagnostics(str(f)))
        assert result["tool"] == "py_compile"
        assert len(result["diagnostics"]) == 1
        assert result["diagnostics"][0]["severity"] == "error"

    # Lines 138-147: _typescript_diagnostics
    @patch("mullet.core.tools.lsp._try_tool")
    def test_typescript_diagnostics_tsc_found(self, mock_tool, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("let x = 1;\n")
        mock_tool.return_value = MagicMock(returncode=0, stdout="test.ts(1,1): error TS1234: msg\n", stderr="")
        result = json.loads(_typescript_diagnostics(str(f)))
        assert result["tool"] == "tsc"

    @patch("mullet.core.tools.lsp._try_tool")
    def test_typescript_diagnostics_tsc_not_found(self, mock_tool, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("let x = 1;\n")
        mock_tool.return_value = None
        result = json.loads(_typescript_diagnostics(str(f)))
        assert "error" in result

    # Lines 152-163: _go_diagnostics
    @patch("mullet.core.tools.lsp._try_tool")
    def test_go_diagnostics_vet_found(self, mock_tool, tmp_path):
        f = tmp_path / "test.go"
        f.write_text("package main\n")
        mock_tool.return_value = MagicMock(returncode=0, stdout="", stderr="test.go:1:1: warning: something")
        result = json.loads(_go_diagnostics(str(f)))
        assert result["tool"] == "go vet"
        assert len(result["diagnostics"]) >= 1

    @patch("mullet.core.tools.lsp._try_tool")
    def test_go_diagnostics_vet_not_found(self, mock_tool, tmp_path):
        f = tmp_path / "test.go"
        f.write_text("package main\n")
        mock_tool.return_value = None
        result = json.loads(_go_diagnostics(str(f)))
        assert "error" in result

    # Lines 171-201: _definition
    def test_definition_no_line(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        result = json.loads(_definition(str(f), 0, 0, ""))
        assert "error" in result

    def test_definition_py_jedi_success(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n")
        mock_jedi = MagicMock()
        mock_def = MagicMock()
        mock_def.name = "foo"
        mock_def.module_name = "test"
        mock_def.module_path = None
        mock_def.line = 1
        mock_def.column = 0
        mock_def.type = "function"
        mock_def.description = "def foo"
        mock_script = MagicMock()
        mock_script.goto.return_value = [mock_def]
        mock_jedi.Script.return_value = mock_script
        mock_jedi.Project.return_value = MagicMock()
        with patch.dict(sys.modules, {"jedi": mock_jedi}):
            result = json.loads(_definition(str(f), 1, 1, ""))
        assert result["tool"] == "jedi"
        assert result["definitions"][0]["name"] == "foo"

    def test_definition_py_jedi_import_error(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        with patch.dict(sys.modules, {"jedi": None}):
            result = json.loads(_definition(str(f), 1, 1, ""))
        assert "jedi not installed" in result["error"]

    def test_definition_py_jedi_exception(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        mock_jedi = MagicMock()
        mock_jedi.Script.side_effect = ValueError("boom")
        with patch.dict(sys.modules, {"jedi": mock_jedi}):
            result = json.loads(_definition(str(f), 1, 1, ""))
        assert "Definition lookup failed" in result["error"]

    @patch("mullet.core.tools.lsp._grep_definition")
    def test_definition_nonpy_with_symbol(self, mock_grep, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("let x = 1;\n")
        mock_grep.return_value = '{"definitions":[]}'
        _definition(str(f), 1, 1, "foo")
        mock_grep.assert_called_once()

    def test_definition_nonpy_without_symbol(self, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("let x = 1;\n")
        result = json.loads(_definition(str(f), 1, 1, ""))
        assert "not supported" in result["error"]

    # Line 209: _references with line but no symbol
    def test_references_no_symbol_with_line(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        with patch("mullet.core.tools.lsp._extract_symbol", return_value="x") as mock_extract:
            with patch("mullet.core.tools.lsp._try_tool", return_value=MagicMock(stdout="", stderr="", returncode=0)):
                json.loads(_references(str(f), 1, 1, ""))
        mock_extract.assert_called_once_with(str(f), 1, 1)

    # Lines 213-249: _references body
    @patch("mullet.core.tools.lsp._try_tool")
    def test_references_grep_available(self, mock_tool, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        mock_tool.return_value = MagicMock(stdout="test.py:1: x = 1\n", stderr="", returncode=0)
        result = json.loads(_references(str(f), 0, 0, "x"))
        assert result["symbol"] == "x"
        assert len(result["references"]) >= 1

    @patch("mullet.core.tools.lsp._try_tool")
    def test_references_grep_not_available(self, mock_tool, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        mock_tool.return_value = None
        result = json.loads(_references(str(f), 0, 0, "x"))
        assert "grep not available" in result["error"]

    # Lines 257-280: _hover
    def test_hover_no_line(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        result = json.loads(_hover(str(f), 0, 0, ""))
        assert "error" in result

    def test_hover_py_jedi_success(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n")
        mock_jedi = MagicMock()
        mock_h = MagicMock()
        mock_h.name = "foo"
        mock_h.type = "function"
        mock_h.description = "def foo"
        mock_h.docstring.return_value = "doc"
        mock_h.module_name = "test"
        mock_script = MagicMock()
        mock_script.help.return_value = [mock_h]
        mock_jedi.Script.return_value = mock_script
        mock_jedi.Project.return_value = MagicMock()
        with patch.dict(sys.modules, {"jedi": mock_jedi}):
            result = json.loads(_hover(str(f), 1, 1, ""))
        assert result["tool"] == "jedi"
        assert result["hover"][0]["name"] == "foo"

    def test_hover_py_jedi_import_error(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        with patch.dict(sys.modules, {"jedi": None}):
            result = json.loads(_hover(str(f), 1, 1, ""))
        assert "jedi not installed" in result["error"]

    def test_hover_py_jedi_exception(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        mock_jedi = MagicMock()
        mock_jedi.Script.side_effect = ValueError("boom")
        with patch.dict(sys.modules, {"jedi": mock_jedi}):
            result = json.loads(_hover(str(f), 1, 1, ""))
        assert "Hover lookup failed" in result["error"]

    def test_hover_nonpy(self, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("let x = 1;\n")
        result = json.loads(_hover(str(f), 1, 1, ""))
        assert "not supported" in result["error"]

    # Lines 303-308: _symbols — .py jedi ImportError → fallback to _regex_symbols
    def test_symbols_py_jedi_import_error(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("class Foo:\n    pass\n")
        with patch.dict(sys.modules, {"jedi": None}):
            result = json.loads(_symbols(str(f), 0, 0, ""))
        assert "symbols" in result

    # Lines 315-323: _symbols — non-.py ext → _regex_symbols directly
    def test_symbols_nonpy_ext(self, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("function foo() {}\n")
        result = json.loads(_symbols(str(f), 0, 0, ""))
        assert "symbols" in result
        names = [s["name"] for s in result["symbols"]]
        assert "foo" in names

    # Lines 355-381: _grep_definition body
    @patch("mullet.core.tools.lsp._try_tool")
    def test_grep_definition_found(self, mock_tool, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("function foo() {}\n")
        mock_tool.return_value = MagicMock(stdout="src/test.ts:1:function foo() {}\n", stderr="", returncode=0)
        result = json.loads(_grep_definition("foo", str(tmp_path), ".ts"))
        assert result["tool"] == "grep"
        assert len(result["definitions"]) >= 1

    @patch("mullet.core.tools.lsp._try_tool")
    def test_grep_definition_unknown_ext(self, mock_tool, tmp_path):
        f = tmp_path / "test.rb"
        f.write_text("def foo\nend\n")
        mock_tool.return_value = MagicMock(stdout="test.rb:1:def foo\n", stderr="", returncode=0)
        result = json.loads(_grep_definition("foo", str(tmp_path), ".rb"))
        assert result["tool"] == "grep"

    @patch("mullet.core.tools.lsp._try_tool")
    def test_grep_definition_no_results(self, mock_tool, tmp_path):
        f = tmp_path / "test.ts"
        f.write_text("function foo() {}\n")
        mock_tool.return_value = MagicMock(stdout="", stderr="", returncode=0)
        result = json.loads(_grep_definition("bar", str(tmp_path), ".ts"))
        assert "No definition found" in result["message"]

    # Lines 405-406: _regex_symbols with unsupported ext (.rb) → error JSON
    def test_regex_symbols_unsupported_rb(self, tmp_path):
        f = tmp_path / "test.rb"
        f.write_text("def foo\nend\n")
        result = json.loads(_regex_symbols(str(f), ".rb"))
        assert "error" in result

    # Branch 159->157: _go_diagnostics loop with multiple lines
    @patch("mullet.core.tools.lsp._try_tool")
    def test_go_diagnostics_vet_multiple_lines(self, mock_tool, tmp_path):
        f = tmp_path / "test.go"
        f.write_text("package main\n")
        mock_tool.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="test.go:1:1: warning: something\ntest.go:2:1: error: else\n"
        )
        result = json.loads(_go_diagnostics(str(f)))
        assert len(result["diagnostics"]) == 2

    # Branch 240->246: _references with malformed grep line
    @patch("mullet.core.tools.lsp._try_tool")
    def test_references_malformed_grep_line(self, mock_tool, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        mock_tool.return_value = MagicMock(stdout="a:b\n", stderr="", returncode=0)
        result = json.loads(_references(str(f), 0, 0, "x"))
        assert result["references"] == []

    # Line 247: _references break when >=30 refs
    @patch("mullet.core.tools.lsp._try_tool")
    def test_references_max_30_break(self, mock_tool, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        lines = "\n".join([f"test.py:{i}: x = 1" for i in range(1, 35)])
        mock_tool.return_value = MagicMock(stdout=lines, stderr="", returncode=0)
        result = json.loads(_references(str(f), 0, 0, "x"))
        assert len(result["references"]) == 30

    # Branch 295->294: _symbols with jedi returning non-matching type
    def test_symbols_py_jedi_nonmatching_type(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        mock_jedi = MagicMock()
        mock_name = MagicMock()
        mock_name.type = "param"  # not in ("class", "function", "statement")
        mock_name.name = "x"
        mock_name.line = 1
        mock_name.description = "x"
        mock_script = MagicMock()
        mock_script.get_names.return_value = [mock_name]
        mock_jedi.Script.return_value = mock_script
        mock_jedi.Project.return_value = MagicMock()
        with patch.dict(sys.modules, {"jedi": mock_jedi}):
            result = json.loads(_symbols(str(f), 0, 0, ""))
        assert result["symbols"] == []

    # Lines 315-323: _try_tool FileNotFoundError
    def test_try_tool_file_not_found(self, tmp_path):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = _try_tool(["nonexistent_tool"])
        assert result is None

    # Lines 315-323: _try_tool TimeoutExpired
    def test_try_tool_timeout(self, tmp_path):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            result = _try_tool(["slow_tool"])
        assert result is None

    # Lines 349-350: _extract_symbol exception
    def test_extract_symbol_exception(self, tmp_path):
        with patch("builtins.open", side_effect=OSError("boom")):
            assert _extract_symbol(str(tmp_path / "x.py"), 1, 1) == ""

    # Lines 405-406: _regex_symbols exception
    def test_regex_symbols_exception(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo(): pass\n")
        with patch("pathlib.Path.read_text", side_effect=OSError("boom")):
            result = json.loads(_regex_symbols(str(f), ".py"))
        assert "error" in result

    # Branch 471->456: _parse_text_diagnostics with info line and skipped line
    def test_parse_text_diagnostics_info_line(self):
        output = "test.py:10:5: error: bad\nsome random info\nFound 1 error"
        result = json.loads(_parse_text_diagnostics(output, "test.py", "tool"))
        assert len(result["diagnostics"]) == 2
        assert result["diagnostics"][1]["severity"] == "info"
