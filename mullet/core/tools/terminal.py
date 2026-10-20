"""Shell execution tool with working directory tracking and output limits.

Ported from Hermes terminal tool patterns: tracks cwd across calls,
limits output size to prevent context bloat, captures exit codes.
"""

import json
import os
import signal
import subprocess

from mullet.core.tools.injection_guard import InjectionGuard

_injection_guard = InjectionGuard()

# Persistent working directory across tool calls within a session
_current_cwd: str = os.getcwd()

MAX_OUTPUT_CHARS = 50_000  # Prevent context bloat from huge outputs
MAX_OUTPUT_LINES = 500


TERMINAL_SCHEMA = {
    "description": (
        "Execute a shell command and return its output. "
        "Use absolute paths. The working directory persists between calls. "
        "Use 'cd /path && command' to change directory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute. Use absolute paths.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default: 30.",
            },
            "working_directory": {
                "type": "string",
                "description": "Working directory for this command. If omitted, uses the persistent cwd.",
            },
        },
        "required": ["command"],
    },
}


def terminal(command: str, timeout: int = 30, working_directory: str | None = None, **_kwargs) -> str:
    """Execute a shell command with cwd tracking and output limits."""
    global _current_cwd

    cwd = working_directory or _current_cwd

    # Track cd commands to update persistent cwd
    if command.strip().startswith("cd "):
        target = command.strip()[3:].strip().rstrip(";").strip()
        if target:
            expanded = os.path.expanduser(target)
            if os.path.isabs(expanded):
                new_cwd = expanded
            else:
                new_cwd = os.path.normpath(os.path.join(cwd, expanded))
            if os.path.isdir(new_cwd):
                _current_cwd = new_cwd
                return f"Changed directory to: {_current_cwd}"
            else:
                return json.dumps({"error": f"Directory not found: {new_cwd}"})

    # Check for injection patterns before executing
    guard_result = _injection_guard.check(command)
    if guard_result.is_suspicious:
        return (
            f"[error] Command blocked by injection guard "
            f"(confidence={guard_result.confidence}, "
            f"patterns={guard_result.patterns_matched}): {command[:100]}"
        )

    # Append sentinel to capture cwd after execution (avoids second subprocess call)
    _CWD_SENTINEL = "__MULLET_CWD__"
    augmented_command = f"{command}; echo {_CWD_SENTINEL}=$(pwd)"

    try:
        proc = subprocess.Popen(
            augmented_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"},
            start_new_session=True,
        )
        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            raise

        # Update cwd from sentinel in stdout (no second subprocess needed)
        stdout = stdout_data
        sentinel_prefix = f"{_CWD_SENTINEL}="
        if sentinel_prefix in stdout:
            sentinel_idx = stdout.rfind(sentinel_prefix)
            sentinel_line = stdout[sentinel_idx + len(sentinel_prefix):].split("\n")[0].strip()
            # Strip sentinel from output shown to user
            stdout = stdout[:sentinel_idx].rstrip("\n")
            if sentinel_line and os.path.isdir(sentinel_line):
                _current_cwd = sentinel_line

        output = stdout
        if stderr_data:
            # Only add stderr if it contains meaningful content
            stderr_clean = stderr_data.strip()
            if stderr_clean and stderr_clean != output.strip():
                output += f"\n[stderr]\n{stderr_clean}"

        # Truncate large outputs
        output = _truncate_output(output)

        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"

        # Add context for the LLM
        prefix = f"[cwd: {cwd}]\n" if cwd != os.getcwd() else ""
        return f"{prefix}{output}".strip() or "(no output)"

    except subprocess.TimeoutExpired:
        return f"[error] Command timed out after {timeout}s: {command[:100]}"
    except FileNotFoundError:
        return f"[error] Working directory not found: {cwd}"
    except Exception as e:
        return f"[error] {e}"


def _truncate_output(output: str) -> str:
    """Truncate output to prevent context bloat."""
    lines = output.split("\n")

    if len(lines) > MAX_OUTPUT_LINES:
        kept = lines[:MAX_OUTPUT_LINES // 2] + [
            f"\n... ({len(lines) - MAX_OUTPUT_LINES} lines omitted) ...\n"
        ] + lines[-MAX_OUTPUT_LINES // 2:]
        output = "\n".join(kept)

    if len(output) > MAX_OUTPUT_CHARS:
        # Head/tail truncation: keep 70% head + 30% tail (preserves errors at end)
        head_size = int(MAX_OUTPUT_CHARS * 0.7)
        tail_size = int(MAX_OUTPUT_CHARS * 0.3)
        omitted = len(output) - head_size - tail_size
        output = (
            output[:head_size]
            + f"\n\n... ({omitted:,} chars omitted) ...\n\n"
            + output[-tail_size:]
        )

    return output


def get_cwd() -> str:
    """Get the current persistent working directory."""
    return _current_cwd


def set_cwd(path: str):
    """Set the persistent working directory."""
    global _current_cwd
    _current_cwd = path


def register(registry):
    """Register terminal tools into a ToolRegistry."""
    registry.register(
        name="terminal",
        func=terminal,
        schema=TERMINAL_SCHEMA,
        toolset="terminal",
    )
