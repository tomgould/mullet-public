"""Tests for predigest pure helpers and python_repl pure schema/blocklist."""

import json

from mullet.core.knowledge.predigest import (
    PreDigestPipeline,
    PREDIGEST_THRESHOLD_CHARS,
)
from mullet.core.tools.python_repl import (
    python_repl_handler,
    PYTHON_REPL_SCHEMA,
    _DANGEROUS_PATTERNS,
)


# ---------------------------------------------------------------------------
# PreDigestPipeline.should_predigest
# ---------------------------------------------------------------------------

class _FakeProvider:
    """Minimal stub that satisfies the 'truthy' check."""


class TestShouldPredigest:
    def test_no_provider_always_false(self):
        p = PreDigestPipeline()
        assert p.should_predigest("x" * 100_000) is False

    def test_with_provider_short_content_false(self):
        p = PreDigestPipeline(local_provider=_FakeProvider())
        assert p.should_predigest("short text") is False

    def test_with_provider_above_threshold_true(self):
        p = PreDigestPipeline(local_provider=_FakeProvider())
        assert p.should_predigest("x" * (PREDIGEST_THRESHOLD_CHARS + 1)) is True

    def test_with_provider_at_threshold_false(self):
        p = PreDigestPipeline(local_provider=_FakeProvider())
        assert p.should_predigest("x" * PREDIGEST_THRESHOLD_CHARS) is False

    def test_empty_content_false(self):
        p = PreDigestPipeline(local_provider=_FakeProvider())
        assert p.should_predigest("") is False


# ---------------------------------------------------------------------------
# PreDigestPipeline.build_cloud_prompt
# ---------------------------------------------------------------------------

class TestBuildCloudPrompt:
    def setup_method(self):
        self.p = PreDigestPipeline()

    def test_task_in_output(self):
        result = self.p.build_cloud_prompt("analyze the data", "Summary here")
        assert "analyze the data" in result

    def test_summary_in_output(self):
        result = self.p.build_cloud_prompt("analyze", "key facts: X, Y, Z")
        assert "key facts: X, Y, Z" in result

    def test_returns_string(self):
        result = self.p.build_cloud_prompt("task", "summary")
        assert isinstance(result, str)

    def test_has_separator(self):
        result = self.p.build_cloud_prompt("task", "summary")
        assert "---" in result


# ---------------------------------------------------------------------------
# PreDigestPipeline.predigest (no local provider path)
# ---------------------------------------------------------------------------

class TestPredigestNoProvider:
    def setup_method(self):
        self.p = PreDigestPipeline()

    def test_truncates_content(self):
        content = "x" * 100_000
        result = self.p.predigest(content)
        assert len(result) < len(content)

    def test_result_has_truncated_marker(self):
        result = self.p.predigest("x" * 100_000)
        assert "[..." in result

    def test_returns_string(self):
        assert isinstance(self.p.predigest("hello"), str)


# ---------------------------------------------------------------------------
# PREDIGEST_THRESHOLD_CHARS constant
# ---------------------------------------------------------------------------

class TestPredigestConstant:
    def test_threshold_is_positive(self):
        assert PREDIGEST_THRESHOLD_CHARS > 0

    def test_threshold_is_int(self):
        assert isinstance(PREDIGEST_THRESHOLD_CHARS, int)


# ---------------------------------------------------------------------------
# PYTHON_REPL_SCHEMA
# ---------------------------------------------------------------------------

class TestPythonReplSchema:
    def test_has_description(self):
        assert "description" in PYTHON_REPL_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in PYTHON_REPL_SCHEMA

    def test_code_is_required(self):
        required = PYTHON_REPL_SCHEMA["parameters"]["required"]
        assert "code" in required

    def test_code_param_exists(self):
        props = PYTHON_REPL_SCHEMA["parameters"]["properties"]
        assert "code" in props

    def test_timeout_param_exists(self):
        props = PYTHON_REPL_SCHEMA["parameters"]["properties"]
        assert "timeout" in props


# ---------------------------------------------------------------------------
# _DANGEROUS_PATTERNS
# ---------------------------------------------------------------------------

class TestDangerousPatterns:
    def test_patterns_nonempty(self):
        assert len(_DANGEROUS_PATTERNS) > 0

    def test_each_pattern_has_label_and_regex(self):
        for label, pattern in _DANGEROUS_PATTERNS:
            assert isinstance(label, str)
            assert hasattr(pattern, "match")

    def test_import_os_in_patterns(self):
        labels = [label for label, _ in _DANGEROUS_PATTERNS]
        assert "import os" in labels

    def test_eval_in_patterns(self):
        labels = [label for label, _ in _DANGEROUS_PATTERNS]
        assert any("eval" in label for label in labels)


# ---------------------------------------------------------------------------
# python_repl_handler — blocked patterns
# ---------------------------------------------------------------------------

class TestPythonReplHandlerBlocked:
    def _run(self, code):
        return json.loads(python_repl_handler(code))

    def test_import_os_blocked(self):
        result = self._run("import os")
        assert result["success"] is False
        assert "Blocked" in result["stderr"]

    def test_import_subprocess_blocked(self):
        result = self._run("import subprocess")
        assert result["success"] is False

    def test_eval_blocked(self):
        result = self._run("eval('1+1')")
        assert result["success"] is False

    def test_exec_blocked(self):
        result = self._run("exec('x=1')")
        assert result["success"] is False

    def test_import_shutil_blocked(self):
        result = self._run("import shutil")
        assert result["success"] is False

    def test_blocked_returns_return_code_1(self):
        result = self._run("import os")
        assert result["return_code"] == 1


# ---------------------------------------------------------------------------
# python_repl_handler — safe code
# ---------------------------------------------------------------------------

class TestPythonReplHandlerSafe:
    def _run(self, code):
        return json.loads(python_repl_handler(code))

    def test_simple_print_succeeds(self):
        result = self._run("print('hello')")
        assert result["success"] is True

    def test_stdout_captured(self):
        result = self._run("print(42)")
        assert "42" in result["stdout"]

    def test_return_code_zero_on_success(self):
        result = self._run("x = 1 + 1")
        assert result["return_code"] == 0

    def test_returns_json_string(self):
        raw = python_repl_handler("x = 1")
        assert isinstance(raw, str)
        parsed = json.loads(raw)
        assert "success" in parsed

    def test_syntax_error_returns_failed(self):
        result = self._run("def (")
        assert result["success"] is False

    def test_stderr_captured_on_error(self):
        result = self._run("raise ValueError('test error')")
        assert result["success"] is False
        assert result["stderr"] != ""
