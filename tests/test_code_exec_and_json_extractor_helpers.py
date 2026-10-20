"""Tests for code_exec (ExecutionResult, CodeExecutor) and json_extractor (_strip_jsonc, _detect_json_type)."""

import pytest

from mullet.core.tools.code_exec import (
    ExecutionResult,
    CodeExecutor,
    CODE_EXEC_SCHEMA,
)
from mullet.knowledge.code.extractors.json_extractor import (
    _strip_jsonc,
    JSONExtractor,
)


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

class TestExecutionResult:
    def test_stdout_stored(self):
        er = ExecutionResult(stdout="output", stderr="", return_code=0, timed_out=False, duration_s=0.1)
        assert er.stdout == "output"

    def test_stderr_stored(self):
        er = ExecutionResult(stdout="", stderr="error", return_code=1, timed_out=False, duration_s=0.1)
        assert er.stderr == "error"

    def test_return_code_stored(self):
        er = ExecutionResult(stdout="", stderr="", return_code=42, timed_out=False, duration_s=0.1)
        assert er.return_code == 42

    def test_timed_out_stored(self):
        er = ExecutionResult(stdout="", stderr="", return_code=-1, timed_out=True, duration_s=30.0)
        assert er.timed_out is True

    def test_duration_stored(self):
        er = ExecutionResult(stdout="", stderr="", return_code=0, timed_out=False, duration_s=1.23)
        assert er.duration_s == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# CodeExecutor.execute — basic behavior
# ---------------------------------------------------------------------------

class TestCodeExecutorBasic:
    def setup_method(self):
        self.executor = CodeExecutor()

    def test_returns_execution_result(self):
        result = self.executor.execute('x = 1')
        assert isinstance(result, ExecutionResult)

    def test_print_captured_in_stdout(self):
        result = self.executor.execute('print("hello world")')
        assert "hello world" in result.stdout

    def test_successful_return_code_zero(self):
        result = self.executor.execute('x = 1 + 1')
        assert result.return_code == 0

    def test_not_timed_out_for_fast_code(self):
        result = self.executor.execute('x = 1')
        assert result.timed_out is False

    def test_duration_positive(self):
        result = self.executor.execute('x = 1')
        assert result.duration_s > 0

    def test_nonzero_return_code_on_error(self):
        result = self.executor.execute('raise ValueError("oops")')
        assert result.return_code != 0

    def test_error_captured_in_stderr(self):
        result = self.executor.execute('raise ValueError("oops")')
        assert "ValueError" in result.stderr

    def test_non_python_raises_value_error(self):
        with pytest.raises(ValueError, match="python"):
            self.executor.execute("code", language="ruby")

    def test_multiline_code_executed(self):
        result = self.executor.execute("x = 2\ny = 3\nprint(x + y)")
        assert "5" in result.stdout

    def test_syntax_error_returns_nonzero(self):
        result = self.executor.execute("def broken(:")
        assert result.return_code != 0


# ---------------------------------------------------------------------------
# CODE_EXEC_SCHEMA
# ---------------------------------------------------------------------------

class TestCodeExecSchema:
    def test_schema_is_dict(self):
        assert isinstance(CODE_EXEC_SCHEMA, dict)

    def test_name_is_code_exec(self):
        assert CODE_EXEC_SCHEMA["name"] == "code_exec"

    def test_has_description(self):
        assert "description" in CODE_EXEC_SCHEMA
        assert isinstance(CODE_EXEC_SCHEMA["description"], str)

    def test_has_parameters(self):
        assert "parameters" in CODE_EXEC_SCHEMA

    def test_code_required(self):
        assert "code" in CODE_EXEC_SCHEMA["parameters"]["required"]

    def test_has_code_property(self):
        props = CODE_EXEC_SCHEMA["parameters"]["properties"]
        assert "code" in props


# ---------------------------------------------------------------------------
# _strip_jsonc
# ---------------------------------------------------------------------------

class TestStripJsonc:
    def test_empty_object(self):
        assert _strip_jsonc("{}") == "{}"

    def test_valid_json_unchanged(self):
        result = _strip_jsonc('{"key": "val"}')
        assert result == '{"key": "val"}'

    def test_removes_single_line_comment(self):
        result = _strip_jsonc('{"key": "val"} // trailing comment')
        assert "//" not in result

    def test_removes_inline_comment(self):
        text = '{\n  // comment here\n  "key": "val"\n}'
        result = _strip_jsonc(text)
        assert "// comment here" not in result
        assert '"key"' in result

    def test_removes_block_comment(self):
        text = '{"key": /* block comment */ "val"}'
        result = _strip_jsonc(text)
        assert "/*" not in result
        assert "*/" not in result

    def test_removes_trailing_comma_before_brace(self):
        result = _strip_jsonc('{"key": "val", }')
        assert ", }" not in result

    def test_removes_trailing_comma_before_bracket(self):
        result = _strip_jsonc('{"arr": [1, 2, 3,]}')
        assert "3,]" not in result

    def test_url_in_string_not_broken(self):
        result = _strip_jsonc('{"url": "http://example.com/path"}')
        assert "http://example.com/path" in result

    def test_slash_slash_in_string_preserved(self):
        result = _strip_jsonc('{"url": "http://example.com"}')
        assert "http://example.com" in result

    def test_returns_string(self):
        assert isinstance(_strip_jsonc("{}"), str)

    def test_result_is_valid_json(self):
        import json
        result = _strip_jsonc('{"key": "val", // comment\n"other": 1, }')
        parsed = json.loads(result)
        assert parsed["other"] == 1


# ---------------------------------------------------------------------------
# JSONExtractor._detect_json_type
# ---------------------------------------------------------------------------

class TestDetectJsonType:
    def setup_method(self):
        self.ext = JSONExtractor()

    def test_composer_json_by_filename(self):
        assert self.ext._detect_json_type({}, "composer.json") == "composer"

    def test_composer_lock_by_filename(self):
        assert self.ext._detect_json_type({}, "composer.lock") == "composer-lock"

    def test_package_json_by_filename(self):
        assert self.ext._detect_json_type({}, "package.json") == "package"

    def test_package_lock_by_filename(self):
        assert self.ext._detect_json_type({}, "package-lock.json") == "package-lock"

    def test_tsconfig_json_by_filename(self):
        assert self.ext._detect_json_type({}, "tsconfig.json") == "tsconfig"

    def test_jsconfig_json_by_filename(self):
        assert self.ext._detect_json_type({}, "jsconfig.json") == "tsconfig"

    def test_compiler_options_detected_as_tsconfig(self):
        result = self.ext._detect_json_type({"compilerOptions": {}}, "config.json")
        assert result == "tsconfig"

    def test_dependencies_and_scripts_detected_as_package(self):
        result = self.ext._detect_json_type({"dependencies": {}, "scripts": {}}, "app.json")
        assert result == "package"

    def test_generic_for_unknown(self):
        assert self.ext._detect_json_type({}, "data.json") == "generic"

    def test_generic_for_empty_content(self):
        assert self.ext._detect_json_type({}, "misc.json") == "generic"

    def test_case_insensitive_filename(self):
        # filename is lowercased before comparison
        result = self.ext._detect_json_type({}, "/path/to/PACKAGE.json")
        assert result == "package"
