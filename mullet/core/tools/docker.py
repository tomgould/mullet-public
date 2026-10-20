"""Docker tool — container management for agents.

Provides structured access to Docker operations: build, run, exec,
logs, ps, and images. Requires Docker to be installed and accessible.
"""

import json
import logging
import shlex
import subprocess

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 30_000

# Dangerous Docker flags that could escape container isolation
_BLOCKED_FLAGS = {
    "--privileged",
    "--cap-add",
    "--security-opt",
    "--pid=host",
    "--network=host",
    "-v /:/",
    "--volume=/:/",
}


def _check_blocked_flags(options: str) -> str | None:
    """Return an error message if options contain blocked flags, else None."""
    opts_lower = options.lower()
    for flag in _BLOCKED_FLAGS:
        if flag.lower() in opts_lower:
            return f"Blocked flag '{flag}' detected in options. This flag is not allowed for security reasons."
    return None

DOCKER_SCHEMA = {
    "description": (
        "Interact with Docker containers. Build images, run containers, "
        "execute commands, view logs, and list containers/images."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["ps", "images", "build", "run", "exec", "logs", "stop", "rm"],
                "description": "Docker action to perform.",
            },
            "target": {
                "type": "string",
                "description": "Container/image name, Dockerfile path, or container ID.",
            },
            "command": {
                "type": "string",
                "description": "Command to run (for exec) or build args.",
            },
            "options": {
                "type": "string",
                "description": "Additional flags (e.g., '-d' for detached, '--rm' for auto-remove).",
            },
        },
        "required": ["action"],
    },
}


def _run_docker(args: list[str], timeout: int = 60) -> tuple[str, int]:
    """Run a docker command."""
    try:
        result = subprocess.run(
            ["docker"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n[truncated]"
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return f"Docker command timed out ({timeout}s)", 1
    except FileNotFoundError:
        return "Docker not found. Install Docker to use this tool.", 1


def docker_tool(action: str = "ps", target: str = "", command: str = "",
                options: str = "", **kwargs) -> str:
    """Execute a Docker operation."""
    if action == "ps":
        output, code = _run_docker(["ps", "--format", "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}"])
        if code != 0:
            return json.dumps({"error": output})
        return json.dumps({"containers": output, "exit_code": code})

    elif action == "images":
        output, code = _run_docker(["images", "--format", "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"])
        if code != 0:
            return json.dumps({"error": output})
        return json.dumps({"images": output, "exit_code": code})

    elif action == "build":
        if not target:
            return json.dumps({"error": "build requires target (Dockerfile path or directory)"})
        if options:
            blocked = _check_blocked_flags(options)
            if blocked:
                return json.dumps({"error": blocked})
        args = ["build"]
        if options:
            args.extend(shlex.split(options))
        args.extend(["-t", command or "mullet-build", target])
        output, code = _run_docker(args, timeout=300)
        return json.dumps({"output": output, "exit_code": code, "success": code == 0})

    elif action == "run":
        if not target:
            return json.dumps({"error": "run requires target (image name)"})
        if options:
            blocked = _check_blocked_flags(options)
            if blocked:
                return json.dumps({"error": blocked})
        args = ["run"]
        if options:
            args.extend(shlex.split(options))
        args.append(target)
        if command:
            args.extend(shlex.split(command))
        output, code = _run_docker(args, timeout=120)
        return json.dumps({"output": output, "exit_code": code})

    elif action == "exec":
        if not target:
            return json.dumps({"error": "exec requires target (container ID or name)"})
        if not command:
            return json.dumps({"error": "exec requires command"})
        args = ["exec", target] + shlex.split(command)
        output, code = _run_docker(args)
        return json.dumps({"output": output, "exit_code": code})

    elif action == "logs":
        if not target:
            return json.dumps({"error": "logs requires target (container ID or name)"})
        args = ["logs", "--tail", "100", target]
        output, code = _run_docker(args)
        return json.dumps({"logs": output, "exit_code": code})

    elif action == "stop":
        if not target:
            return json.dumps({"error": "stop requires target (container ID or name)"})
        output, code = _run_docker(["stop", target])
        return json.dumps({"output": output, "exit_code": code, "stopped": code == 0})

    elif action == "rm":
        if not target:
            return json.dumps({"error": "rm requires target (container ID or name)"})
        output, code = _run_docker(["rm", target])
        return json.dumps({"output": output, "exit_code": code, "removed": code == 0})

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


def register(registry):
    """Register the Docker tool."""
    registry.register(
        name="docker",
        func=docker_tool,
        schema=DOCKER_SCHEMA,
        toolset="docker",
    )
