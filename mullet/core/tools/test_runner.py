"""Test runner tool — run and parse test suite results.

Gives agents structured test output: pass/fail counts, failure details,
and duration. Supports pytest (Python) and npm test (JavaScript).
"""

import json
import os
import subprocess
import re

MAX_OUTPUT_CHARS = 20_000

TEST_RUNNER_SCHEMA = {
    "description": (
        "Run tests and return structured results. Auto-detects pytest (Python) "
        "or npm test (JavaScript). Returns pass/fail counts and failure details."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Test file or directory to run. Default: auto-detect.",
            },
            "filter": {
                "type": "string",
                "description": "Filter expression (pytest -k, or test name pattern).",
            },
            "verbose": {
                "type": "boolean",
                "description": "Show individual test names. Default: false.",
            },
        },
    },
}


def _detect_test_framework() -> str:
    """Auto-detect which test framework to use."""
    if os.path.exists("pyproject.toml") or os.path.exists("setup.py") or os.path.exists("pytest.ini"):
        return "pytest"
    if os.path.exists("package.json"):
        return "npm"
    if os.path.exists("tests") or os.path.exists("test"):
        return "pytest"
    return "pytest"


def _run_pytest(path: str = "", filter_expr: str = "", verbose: bool = False) -> dict:
    """Run pytest and parse results."""
    cmd = ["python", "-m", "pytest"]
    if path:
        cmd.append(path)
    if filter_expr:
        cmd.extend(["-k", filter_expr])
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    cmd.append("--tb=short")
    cmd.append("--no-header")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return {"error": "Tests timed out (120s)", "framework": "pytest"}

    output = result.stdout + result.stderr
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n[truncated]"

    # Parse pytest summary line: "X passed, Y failed, Z errors in Ns"
    summary_match = re.search(
        r"(\d+)\s+passed(?:.*?(\d+)\s+failed)?(?:.*?(\d+)\s+error)?(?:.*?in\s+([\d.]+)s)?",
        output,
    )

    passed = int(summary_match.group(1)) if summary_match else 0
    failed = int(summary_match.group(2)) if summary_match and summary_match.group(2) else 0
    errors = int(summary_match.group(3)) if summary_match and summary_match.group(3) else 0
    duration = float(summary_match.group(4)) if summary_match and summary_match.group(4) else 0.0

    # Extract failure details
    failures = []
    failure_blocks = re.findall(r"FAILED\s+(\S+)", output)
    for fb in failure_blocks[:10]:  # Limit to 10
        failures.append(fb)

    return {
        "framework": "pytest",
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": passed + failed + errors,
        "duration_s": duration,
        "all_passed": failed == 0 and errors == 0 and passed > 0,
        "failures": failures,
        "output": output,
        "exit_code": result.returncode,
    }


def _run_npm(path: str = "", filter_expr: str = "", verbose: bool = False) -> dict:
    """Run npm test and parse results."""
    cmd = ["npm", "test", "--"]
    if filter_expr:
        cmd.extend(["--testPathPattern", filter_expr])
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return {"error": "Tests timed out (120s)", "framework": "npm"}
    except FileNotFoundError:
        return {"error": "npm not found", "framework": "npm"}

    output = result.stdout + result.stderr
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n[truncated]"

    # Try to parse Jest/Vitest summary
    pass_match = re.search(r"Tests:\s+(?:(\d+)\s+failed,\s+)?(\d+)\s+passed", output)
    passed = int(pass_match.group(2)) if pass_match else 0
    failed = int(pass_match.group(1)) if pass_match and pass_match.group(1) else 0

    return {
        "framework": "npm",
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "all_passed": result.returncode == 0,
        "output": output,
        "exit_code": result.returncode,
    }


def run_tests_tool(args_json: str = "", **kwargs) -> str:
    """Execute tests and return structured results."""
    if kwargs and not args_json:
        args = kwargs
    else:
        try:
            args = json.loads(args_json) if isinstance(args_json, str) else args_json
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments"})

    path = args.get("path", "")
    filter_expr = args.get("filter", "")
    verbose = args.get("verbose", False)

    framework = _detect_test_framework()

    if framework == "pytest":
        result = _run_pytest(path, filter_expr, verbose)
    elif framework == "npm":
        result = _run_npm(path, filter_expr, verbose)
    else:
        return json.dumps({"error": f"Unknown framework: {framework}"})

    return json.dumps(result)


def register(registry):
    """Register the test runner tool."""
    registry.register(
        name="run_tests",
        func=run_tests_tool,
        schema=TEST_RUNNER_SCHEMA,
        toolset="testing",
    )
