"""Tests for symbol_finder pure helpers: _scan_python_file, _extensions_for, SUPPORTED_LANGUAGES."""

import pytest

from mullet.core.knowledge.symbol_finder import (
    SymbolMatch,
    _scan_python_file,
    _scan_ts_file,
    _scan_go_file,
    _scan_php_file,
    _walk_files,
    _extensions_for,
    SUPPORTED_LANGUAGES,
)


# ---------------------------------------------------------------------------
# SUPPORTED_LANGUAGES
# ---------------------------------------------------------------------------

class TestSupportedLanguages:
    def test_is_set(self):
        assert isinstance(SUPPORTED_LANGUAGES, (set, frozenset))

    def test_nonempty(self):
        assert len(SUPPORTED_LANGUAGES) > 0

    def test_python_supported(self):
        assert "python" in SUPPORTED_LANGUAGES

    def test_typescript_supported(self):
        assert "typescript" in SUPPORTED_LANGUAGES

    def test_javascript_supported(self):
        assert "javascript" in SUPPORTED_LANGUAGES

    def test_go_supported(self):
        assert "go" in SUPPORTED_LANGUAGES

    def test_php_supported(self):
        assert "php" in SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# _extensions_for
# ---------------------------------------------------------------------------

class TestExtensionsFor:
    def test_python_extensions(self):
        assert ".py" in _extensions_for("python")

    def test_typescript_extensions(self):
        exts = _extensions_for("typescript")
        assert ".ts" in exts
        assert ".tsx" in exts

    def test_javascript_extensions(self):
        exts = _extensions_for("javascript")
        assert ".js" in exts

    def test_go_extension(self):
        assert ".go" in _extensions_for("go")

    def test_php_extensions(self):
        exts = _extensions_for("php")
        assert ".php" in exts

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            _extensions_for("cobol")

    def test_returns_list(self):
        assert isinstance(_extensions_for("python"), list)


# ---------------------------------------------------------------------------
# SymbolMatch
# ---------------------------------------------------------------------------

class TestSymbolMatch:
    def test_fields_stored(self):
        sm = SymbolMatch(
            name="my_func", type="function",
            file_path="app.py", line_number=10, signature="def my_func():"
        )
        assert sm.name == "my_func"
        assert sm.type == "function"
        assert sm.file_path == "app.py"
        assert sm.line_number == 10
        assert sm.signature == "def my_func():"

    def test_parent_class_defaults_none(self):
        sm = SymbolMatch("f", "function", "f.py", 1, "def f():")
        assert sm.parent_class is None

    def test_parent_class_stored(self):
        sm = SymbolMatch("method", "method", "f.py", 5, "def method():", parent_class="MyClass")
        assert sm.parent_class == "MyClass"


# ---------------------------------------------------------------------------
# _scan_python_file
# ---------------------------------------------------------------------------

def _write_py_file(tmp_path, content):
    """Write Python code to a temp file and return its path."""
    path = tmp_path / "test_code.py"
    path.write_text(content)
    return str(path)


class TestScanPythonFileBasic:
    def test_empty_file_returns_empty(self, tmp_path):
        path = _write_py_file(tmp_path, "")
        assert _scan_python_file(path) == []

    def test_returns_list(self, tmp_path):
        path = _write_py_file(tmp_path, "def f(): pass")
        assert isinstance(_scan_python_file(path), list)

    def test_nonexistent_file_returns_empty(self, tmp_path):
        assert _scan_python_file(str(tmp_path / "nope.py")) == []

    def test_all_results_are_symbol_match(self, tmp_path):
        path = _write_py_file(tmp_path, "def f(): pass")
        result = _scan_python_file(path)
        assert all(isinstance(s, SymbolMatch) for s in result)


class TestScanPythonFileClasses:
    def test_detects_class(self, tmp_path):
        path = _write_py_file(tmp_path, "class MyClass:\n    pass\n")
        symbols = _scan_python_file(path)
        names = [s.name for s in symbols]
        assert "MyClass" in names

    def test_class_type(self, tmp_path):
        path = _write_py_file(tmp_path, "class MyClass:\n    pass\n")
        cls_sym = next(s for s in _scan_python_file(path) if s.name == "MyClass")
        assert cls_sym.type == "class"

    def test_class_line_number(self, tmp_path):
        path = _write_py_file(tmp_path, "\n\nclass MyClass:\n    pass\n")
        cls_sym = next(s for s in _scan_python_file(path) if s.name == "MyClass")
        assert cls_sym.line_number == 3

    def test_class_signature(self, tmp_path):
        path = _write_py_file(tmp_path, "class MyClass(Base):\n    pass\n")
        cls_sym = next(s for s in _scan_python_file(path) if s.name == "MyClass")
        assert "class MyClass" in cls_sym.signature

    def test_class_no_parent(self, tmp_path):
        path = _write_py_file(tmp_path, "class MyClass:\n    pass\n")
        cls_sym = next(s for s in _scan_python_file(path) if s.name == "MyClass")
        assert cls_sym.parent_class is None


class TestScanPythonFileFunctions:
    def test_detects_function(self, tmp_path):
        path = _write_py_file(tmp_path, "def my_func():\n    pass\n")
        names = [s.name for s in _scan_python_file(path)]
        assert "my_func" in names

    def test_function_type(self, tmp_path):
        path = _write_py_file(tmp_path, "def my_func():\n    pass\n")
        sym = next(s for s in _scan_python_file(path) if s.name == "my_func")
        assert sym.type == "function"

    def test_async_function_type(self, tmp_path):
        path = _write_py_file(tmp_path, "async def async_func():\n    pass\n")
        sym = next(s for s in _scan_python_file(path) if s.name == "async_func")
        assert sym.type == "async_function"

    def test_function_no_parent(self, tmp_path):
        path = _write_py_file(tmp_path, "def my_func():\n    pass\n")
        sym = next(s for s in _scan_python_file(path) if s.name == "my_func")
        assert sym.parent_class is None

    def test_function_signature_includes_name(self, tmp_path):
        path = _write_py_file(tmp_path, "def my_func(a, b):\n    pass\n")
        sym = next(s for s in _scan_python_file(path) if s.name == "my_func")
        assert "my_func" in sym.signature


class TestScanPythonFileMethods:
    def _class_with_method(self, tmp_path, method_def):
        code = f"class MyClass:\n    {method_def}\n        pass\n"
        return _write_py_file(tmp_path, code)

    def test_method_detected(self, tmp_path):
        path = _write_py_file(tmp_path, "class MyClass:\n    def method(self):\n        pass\n")
        names = [s.name for s in _scan_python_file(path)]
        assert "method" in names

    def test_method_type(self, tmp_path):
        path = _write_py_file(tmp_path, "class MyClass:\n    def method(self):\n        pass\n")
        sym = next(s for s in _scan_python_file(path) if s.name == "method")
        assert sym.type == "method"

    def test_method_parent_class_set(self, tmp_path):
        path = _write_py_file(tmp_path, "class MyClass:\n    def method(self):\n        pass\n")
        sym = next(s for s in _scan_python_file(path) if s.name == "method")
        assert sym.parent_class == "MyClass"

    def test_async_method_type(self, tmp_path):
        path = _write_py_file(tmp_path, "class MyClass:\n    async def async_m(self):\n        pass\n")
        sym = next(s for s in _scan_python_file(path) if s.name == "async_m")
        assert sym.type == "async_method"

    def test_multiple_symbols_found(self, tmp_path):
        code = (
            "class A:\n    def m1(self): pass\n    def m2(self): pass\n"
            "def top_func(): pass\n"
        )
        path = _write_py_file(tmp_path, code)
        symbols = _scan_python_file(path)
        names = [s.name for s in symbols]
        assert "A" in names
        assert "m1" in names
        assert "m2" in names
        assert "top_func" in names

    def test_file_path_stored(self, tmp_path):
        path = _write_py_file(tmp_path, "def f(): pass\n")
        sym = _scan_python_file(path)[0]
        assert sym.file_path == path

    def test_class_end_detected_by_indent_reset(self, tmp_path):
        """Lines 118-119: code at module level after a class resets current_class."""
        code = (
            "class MyClass:\n"
            "    def method(self):\n"
            "        pass\n"
            "\n"
            "x = 1\n"  # module-level statement after class: indent=0 <= class_indent → reset
            "def top_func():\n"
            "    pass\n"
        )
        path = _write_py_file(tmp_path, code)
        symbols = _scan_python_file(path)
        names = [s.name for s in symbols]
        assert "top_func" in names


class TestTsScannerGaps:
    def test_oserror_returns_empty(self, tmp_path):
        """Lines 164-165: _scan_ts_file on unreadable file → return []."""
        p = tmp_path / "test.ts"
        p.write_text("export function x(): void {}")
        p.chmod(0o000)
        try:
            result = _scan_ts_file(str(p))
            assert result == []
        finally:
            p.chmod(0o644)

    def test_keyword_skip_in_method_scan(self, tmp_path):
        """Line 252: TS method scanner skips names like 'if', 'for', etc."""
        code = "class A {\n  if(x: any) { return x; }\n  myMethod() { return 1; }\n}\n"
        p = tmp_path / "test.ts"
        p.write_text(code)
        symbols = _scan_ts_file(str(p))
        names = [s.name for s in symbols]
        assert "if" not in names
        assert "myMethod" in names


class TestGoScannerGaps:
    def test_oserror_returns_empty(self, tmp_path):
        """Lines 290-291: _scan_go_file on unreadable file → return []."""
        p = tmp_path / "test.go"
        p.write_text("func Hello() {}")
        p.chmod(0o000)
        try:
            result = _scan_go_file(str(p))
            assert result == []
        finally:
            p.chmod(0o644)


class TestPhpScannerGaps:
    def test_oserror_returns_empty(self, tmp_path):
        """Lines 383-384: _scan_php_file on unreadable file → return []."""
        p = tmp_path / "test.php"
        p.write_text("<?php function hello() {}")
        p.chmod(0o000)
        try:
            result = _scan_php_file(str(p))
            assert result == []
        finally:
            p.chmod(0o644)


class TestWalkFilesGaps:
    def test_nonexistent_root_returns_empty(self, tmp_path):
        """Line 576: _walk_files with non-directory root → return []."""
        result = _walk_files(str(tmp_path / "no_such_dir"), ".py")
        assert result == []
