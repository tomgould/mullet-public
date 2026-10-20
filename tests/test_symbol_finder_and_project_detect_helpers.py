"""Tests for symbol_finder pure helpers and project_detect pure functions."""

import os
import tempfile
import pytest

from mullet.core.knowledge.symbol_finder import (
    SymbolMatch,
    _extensions_for,
    _scan_python_file,
    detect_language,
    SUPPORTED_LANGUAGES,
)
from mullet.core.agent.project_detect import (
    ProjectInfo,
    detect_project,
    format_project_context,
)


# ---------------------------------------------------------------------------
# SymbolMatch dataclass
# ---------------------------------------------------------------------------

class TestSymbolMatch:
    def test_fields_stored(self):
        sm = SymbolMatch(
            name="MyFunc",
            type="function",
            file_path="src/utils.py",
            line_number=42,
            signature="def MyFunc(x: int) -> str:",
        )
        assert sm.name == "MyFunc"
        assert sm.type == "function"
        assert sm.line_number == 42

    def test_parent_class_defaults_none(self):
        sm = SymbolMatch(name="f", type="function", file_path="f.py", line_number=1, signature="def f():")
        assert sm.parent_class is None


# ---------------------------------------------------------------------------
# _extensions_for
# ---------------------------------------------------------------------------

class TestExtensionsFor:
    def test_python_extensions(self):
        exts = _extensions_for("python")
        assert ".py" in exts

    def test_typescript_extensions(self):
        exts = _extensions_for("typescript")
        assert ".ts" in exts
        assert ".tsx" in exts

    def test_go_extensions(self):
        exts = _extensions_for("go")
        assert ".go" in exts

    def test_php_extensions(self):
        exts = _extensions_for("php")
        assert ".php" in exts

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError):
            _extensions_for("brainfuck")

    def test_returns_list(self):
        assert isinstance(_extensions_for("python"), list)

    def test_all_supported_languages_have_extensions(self):
        for lang in SUPPORTED_LANGUAGES:
            exts = _extensions_for(lang)
            assert len(exts) > 0


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_py_is_python(self):
        assert detect_language("main.py") == "python"

    def test_ts_is_typescript(self):
        assert detect_language("app.ts") == "typescript"

    def test_tsx_is_typescript(self):
        assert detect_language("component.tsx") == "typescript"

    def test_js_is_javascript(self):
        assert detect_language("index.js") == "javascript"

    def test_go_is_go(self):
        assert detect_language("main.go") == "go"

    def test_php_is_php(self):
        assert detect_language("index.php") == "php"

    def test_unknown_returns_none(self):
        assert detect_language("file.xyz") is None

    def test_path_with_directory(self):
        assert detect_language("src/utils.py") == "python"


# ---------------------------------------------------------------------------
# _scan_python_file
# ---------------------------------------------------------------------------

def _write_temp_py(src: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    f.write(src)
    f.close()
    return f.name


class TestScanPythonFile:
    def test_class_detected(self):
        fname = _write_temp_py("class MyClass:\n    pass\n")
        try:
            results = _scan_python_file(fname)
            classes = [r for r in results if r.type == "class"]
            assert len(classes) == 1
            assert classes[0].name == "MyClass"
        finally:
            os.unlink(fname)

    def test_function_detected(self):
        fname = _write_temp_py("def my_func(x):\n    return x\n")
        try:
            results = _scan_python_file(fname)
            funcs = [r for r in results if r.type == "function"]
            assert len(funcs) == 1
            assert funcs[0].name == "my_func"
        finally:
            os.unlink(fname)

    def test_async_function_detected(self):
        fname = _write_temp_py("async def async_fn(x: int) -> str:\n    return str(x)\n")
        try:
            results = _scan_python_file(fname)
            async_funcs = [r for r in results if r.type == "async_function"]
            assert len(async_funcs) == 1
            assert async_funcs[0].name == "async_fn"
        finally:
            os.unlink(fname)

    def test_method_inside_class(self):
        src = "class Foo:\n    def my_method(self):\n        pass\n"
        fname = _write_temp_py(src)
        try:
            results = _scan_python_file(fname)
            methods = [r for r in results if r.type == "method"]
            assert len(methods) == 1
            assert methods[0].name == "my_method"
        finally:
            os.unlink(fname)

    def test_line_number_correct(self):
        src = "\n\ndef func_on_line_3():\n    pass\n"
        fname = _write_temp_py(src)
        try:
            results = _scan_python_file(fname)
            funcs = [r for r in results if r.type == "function"]
            assert funcs[0].line_number == 3
        finally:
            os.unlink(fname)

    def test_empty_file_returns_empty(self):
        fname = _write_temp_py("")
        try:
            assert _scan_python_file(fname) == []
        finally:
            os.unlink(fname)

    def test_nonexistent_file_returns_empty(self):
        assert _scan_python_file("/nonexistent/file.py") == []

    def test_returns_list(self):
        fname = _write_temp_py("x = 1\n")
        try:
            assert isinstance(_scan_python_file(fname), list)
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# ProjectInfo dataclass
# ---------------------------------------------------------------------------

class TestProjectInfo:
    def test_defaults(self):
        info = ProjectInfo()
        assert info.language == "unknown"
        assert info.framework == ""
        assert info.has_git is False
        assert info.has_docker is False

    def test_custom_values(self):
        info = ProjectInfo(language="python", framework="fastapi")
        assert info.language == "python"
        assert info.framework == "fastapi"


# ---------------------------------------------------------------------------
# detect_project
# ---------------------------------------------------------------------------

class TestDetectProject:
    def test_empty_dir_returns_unknown(self, tmp_path):
        info = detect_project(str(tmp_path))
        assert info.language == "unknown"

    def test_detects_python_from_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask")
        info = detect_project(str(tmp_path))
        assert info.language == "python"

    def test_detects_git(self, tmp_path):
        (tmp_path / ".git").mkdir()
        info = detect_project(str(tmp_path))
        assert info.has_git is True

    def test_detects_docker(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.11")
        info = detect_project(str(tmp_path))
        assert info.has_docker is True

    def test_detects_typescript_from_tsconfig(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app"}')
        (tmp_path / "tsconfig.json").write_text("{}")
        info = detect_project(str(tmp_path))
        assert info.language == "typescript"

    def test_detects_javascript_from_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "app"}')
        info = detect_project(str(tmp_path))
        assert info.language == "javascript"

    def test_returns_project_info(self, tmp_path):
        assert isinstance(detect_project(str(tmp_path)), ProjectInfo)


# ---------------------------------------------------------------------------
# format_project_context
# ---------------------------------------------------------------------------

class TestFormatProjectContext:
    def test_returns_string(self):
        info = ProjectInfo(language="python")
        assert isinstance(format_project_context(info), str)

    def test_language_in_output(self):
        info = ProjectInfo(language="python")
        result = format_project_context(info)
        assert "python" in result

    def test_framework_in_output_when_set(self):
        info = ProjectInfo(language="python", framework="fastapi")
        result = format_project_context(info)
        assert "fastapi" in result

    def test_unknown_language_handled(self):
        info = ProjectInfo()
        result = format_project_context(info)
        assert isinstance(result, str)
