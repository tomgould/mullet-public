import pytest
import time
from unittest.mock import MagicMock
from mullet.core.tools.code_exec import CodeExecutor, code_exec_handler, register

@pytest.fixture
def executor():
    return CodeExecutor()

def test_basic_execution(executor):
    code = "print('Hello, Mullet!')"
    result = executor.execute(code)
    assert result.stdout.strip() == 'Hello, Mullet!'
    assert result.stderr == ''
    assert result.return_code == 0
    assert not result.timed_out
    assert result.duration_s > 0

def test_stderr_capture(executor):
    code = "import sys; sys.stderr.write('Error message')"
    result = executor.execute(code)
    assert result.stdout == ''
    assert result.stderr.strip() == 'Error message'
    assert result.return_code == 0

def test_return_code_success(executor):
    code = "exit(0)"
    result = executor.execute(code)
    assert result.return_code == 0

def test_return_code_failure(executor):
    code = "exit(1)"
    result = executor.execute(code)
    assert result.return_code == 1

def test_timeout_enforcement(executor):
    code = "import time; time.sleep(2)"
    start_time = time.time()
    result = executor.execute(code, timeout=1)
    end_time = time.time()
    assert result.timed_out
    assert result.return_code == -1  # Indicates timeout
    assert end_time - start_time >= 1
    assert end_time - start_time < 2 # Should not run for full 2 seconds

def test_large_stdout_truncation(executor):
    code = "print('a' * 1024 * 105)" # 105KB
    result = executor.execute(code)
    assert len(result.stdout) == 102400 # Max 100KB
    assert result.stdout.startswith('a' * 1024 * 90)
    assert result.return_code == 0

def test_large_stderr_truncation(executor):
    code = "import sys; sys.stderr.write('b' * 1024 * 105)" # 105KB
    result = executor.execute(code)
    assert len(result.stderr) == 102400 # Max 100KB
    assert result.stderr.startswith('b' * 1024 * 90)
    assert result.return_code == 0

def test_empty_code(executor):
    code = ""
    result = executor.execute(code)
    assert result.stdout == ''
    assert result.stderr == ''
    assert result.return_code == 0
    assert not result.timed_out

def test_syntax_error(executor):
    code = "print('hello"
    result = executor.execute(code)
    assert result.stdout == ''
    assert 'SyntaxError' in result.stderr
    assert result.return_code == 1 # Python exits with 1 for unhandled exceptions/syntax errors

def test_runtime_error(executor):
    code = "1 / 0"
    result = executor.execute(code)
    assert result.stdout == ''
    assert 'ZeroDivisionError' in result.stderr
    assert result.return_code == 1

def test_unsupported_language(executor):
    with pytest.raises(ValueError, match="Only 'python' language is supported for now."):
        executor.execute("echo 'hello'", language='bash')

def test_multiple_lines_code(executor):
    code = "a = 1\nb = 2\nprint(a + b)"
    result = executor.execute(code)
    assert result.stdout.strip() == '3'
    assert result.return_code == 0


def test_code_exec_handler_returns_formatted_string():
    result = code_exec_handler(code="print('hi')", timeout=30)
    assert "stdout:" in result
    assert "return_code:" in result
    assert "timed_out:" in result
    assert "duration_s:" in result


def test_code_exec_handler_includes_stderr():
    result = code_exec_handler(code="import sys; sys.stderr.write('err')", timeout=30)
    assert "stderr:" in result
    assert "err" in result


def test_register_adds_code_exec_tool():
    from unittest.mock import ANY
    registry = MagicMock()
    register(registry)
    registry.register.assert_called_once_with(
        name="code_exec",
        func=code_exec_handler,
        schema=ANY,
        toolset="code",
    )
