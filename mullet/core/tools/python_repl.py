"""Python REPL tool for executing Python code in a subprocess.

NOTE: This is NOT a sandbox. The blocklist below is defense-in-depth only.
It reduces accidental damage but cannot prevent a determined bypass.
"""

import json
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

PYTHON_REPL_SCHEMA = {
    "description": (
        "Execute Python code in a subprocess. "
        "Captures stdout, stderr, and returns exit code and execution duration. "
        "Blocks common dangerous patterns as defense-in-depth (not a sandbox)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default: 30.",
                "default": 30,
            },
        },
        "required": ["code"],
    },
}

# Defense-in-depth blocklist -- NOT a real sandbox.
# These regex patterns catch common dangerous operations but can be bypassed.
_DANGEROUS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("import os", re.compile(r"\bimport\s+os\b")),
    ("import subprocess", re.compile(r"\bimport\s+subprocess\b")),
    ("import shutil", re.compile(r"\bimport\s+shutil\b")),
    ("__import__", re.compile(r"__import__\s*\(")),
    ("eval()", re.compile(r"\beval\s*\(")),
    ("exec()", re.compile(r"\bexec\s*\(")),
    ("compile()", re.compile(r"\bcompile\s*\(")),
    ("open() with write", re.compile(r"\bopen\s*\([^)]*['\"][waxWAX][^)]*\)")),
    ("os.system", re.compile(r"\bos\.system\s*\(")),
    ("os.popen", re.compile(r"\bos\.popen\s*\(")),
    ("subprocess.*", re.compile(r"\bsubprocess\.\w+")),
    ("sys.modules", re.compile(r"\bsys\.modules\b")),
]


def python_repl_handler(code: str, timeout: int = 30, **kwargs) -> str:
    """Tool handler for Python REPL."""
    for label, pattern in _DANGEROUS_PATTERNS:
        if pattern.search(code):
            return json.dumps({
                "success": False,
                "stdout": "",
                "stderr": f"Error: Blocked pattern '{label}' detected (defense-in-depth).",
                "return_code": 1,
                "duration_s": 0,
            })

    try:
        process = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            timeout=timeout,
            text=True,
            check=False,
        )
        duration = process.elapsed.total_seconds() if hasattr(process, 'elapsed') else 0

        return json.dumps({
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "return_code": process.returncode,
            "duration_s": duration,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "stdout": "",
            "stderr": f"Error: Execution timed out after {timeout} seconds.",
            "return_code": 1,
            "duration_s": timeout,
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "stdout": "",
            "stderr": f"Error: {e!s}",
            "return_code": 1,
            "duration_s": 0,
        })


def register(registry):
    """Register python_repl tool into the tool registry."""
    registry.register(
        name="python_repl",
        func=python_repl_handler,
        schema=PYTHON_REPL_SCHEMA,
        toolset="code",
    )
