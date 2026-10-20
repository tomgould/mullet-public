"""Sandboxed code-execution tool for agents (subprocess-based with timeout)."""

import subprocess
import tempfile
import shutil
import time
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    duration_s: float

class CodeExecutor:
    """Execute Python code in a temporary sandboxed subprocess."""

    def execute(self, code: str, language: str = 'python', timeout: int = 30) -> ExecutionResult:
        """Run code in a subprocess and return captured output."""
        if language != 'python':
            raise ValueError("Only 'python' language is supported for now.")

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            script_path = os.path.join(temp_dir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)

            start_time = time.time()
            try:
                process = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=temp_dir,
                    check=False
                )
                timed_out = False
            except subprocess.TimeoutExpired as e:
                process = e
                timed_out = True

            end_time = time.time()
            duration = end_time - start_time

            if timed_out:
                # TimeoutExpired.stdout/stderr are bytes or None, not str
                raw_stdout = process.stdout
                raw_stderr = process.stderr
                stdout = raw_stdout.decode("utf-8", errors="replace")[:102400] if isinstance(raw_stdout, bytes) else (raw_stdout[:102400] if raw_stdout else "")
                stderr = raw_stderr.decode("utf-8", errors="replace")[:102400] if isinstance(raw_stderr, bytes) else (raw_stderr[:102400] if raw_stderr else "")
            else:
                stdout = process.stdout[:102400] if process.stdout else ""  # Max 100KB
                stderr = process.stderr[:102400] if process.stderr else ""  # Max 100KB

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode if not timed_out else -1,
                timed_out=timed_out,
                duration_s=duration
            )
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

CODE_EXEC_SCHEMA: Dict[str, Any] = {
    "name": "code_exec",
    "description": "Execute Python code in a sandboxed subprocess.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute."
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default: 30.",
                "default": 30
            }
        },
        "required": ["code"]
    }
}

def code_exec_handler(**kwargs) -> str:
    """Tool handler that executes code and formats the result."""
    executor = CodeExecutor()
    code = kwargs.get("code")
    timeout = kwargs.get("timeout", 30)
    result = executor.execute(code=code, timeout=timeout)

    output = f"stdout:\n{result.stdout}\n"
    if result.stderr:
        output += f"stderr:\n{result.stderr}\n"
    output += f"return_code: {result.return_code}\n"
    output += f"timed_out: {result.timed_out}\n"
    output += f"duration_s: {result.duration_s:.2f}\n"
    return output

def register(registry):
    """Register the code_exec tool with a registry."""
    registry.register(
        name="code_exec",
        func=code_exec_handler,
        schema=CODE_EXEC_SCHEMA,
        toolset="code",
    )
