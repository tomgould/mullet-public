"""Git integration tool — structured access to repository state.

Gives agents structured access to git status, diff, log, and blame
without needing to parse shell output. Read-only operations only
(no commit, push, reset).
"""

import json
import os
import subprocess

MAX_DIFF_CHARS = 30_000
MAX_LOG_ENTRIES = 20

GIT_SCHEMA = {
    "description": (
        "Interact with the git repository. Provides structured access to "
        "status, diff, log, blame, and branch info. Read-only (no commit/push)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "diff", "log", "blame", "branches", "show"],
                "description": "Git action to perform.",
            },
            "path": {
                "type": "string",
                "description": "File path for blame/diff/show. Optional for status/log/branches.",
            },
            "ref": {
                "type": "string",
                "description": "Git ref for log/show/diff (branch, commit, HEAD~3). Default: HEAD.",
            },
            "count": {
                "type": "integer",
                "description": "Number of log entries. Default: 10.",
            },
        },
        "required": ["action"],
    },
}


def _run_git(args: list[str], cwd: str | None = None) -> tuple[str, int]:
    """Run a git command and return (output, exit_code)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=15,
            cwd=cwd or os.getcwd(),
        )
        output = result.stdout or result.stderr or ""
        return output.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "Git command timed out (15s)", 1
    except FileNotFoundError:
        return "Git not found. Install git to use this tool.", 1


def git_tool(args_json: str = "", **kwargs) -> str:
    """Execute a git operation and return structured result."""
    if kwargs and not args_json:
        # Called with unpacked keyword args (from ToolRegistry.dispatch)
        args = kwargs
    else:
        try:
            args = json.loads(args_json) if isinstance(args_json, str) else args_json
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments"})

    action = args.get("action", "status")
    path = args.get("path", "")
    ref = args.get("ref", "HEAD")
    count = min(args.get("count", 10), MAX_LOG_ENTRIES)

    if action == "status":
        output, code = _run_git(["status", "--porcelain", "-b"])
        if code != 0:
            return json.dumps({"error": output})

        lines = output.split("\n")
        branch = ""
        changes = []
        for line in lines:
            if line.startswith("##"):
                branch = line[3:].split("...")[0] if "..." in line else line[3:]
            elif line.strip():
                status_code = line[:2].strip()
                file_path = line[3:].strip()
                changes.append({"status": status_code, "file": file_path})

        return json.dumps({
            "branch": branch,
            "changes": changes,
            "clean": len(changes) == 0,
            "total_changes": len(changes),
        })

    elif action == "diff":
        git_args = ["diff"]
        if ref and ref != "HEAD":
            git_args.append(ref)
        if path:
            git_args.extend(["--", path])
        output, code = _run_git(git_args)
        if len(output) > MAX_DIFF_CHARS:
            output = output[:MAX_DIFF_CHARS] + f"\n\n[... truncated, {len(output)} total chars]"
        return json.dumps({"diff": output, "truncated": len(output) >= MAX_DIFF_CHARS})

    elif action == "log":
        git_args = ["log", f"-{count}", "--format=%H|%an|%ai|%s"]
        if path:
            git_args.extend(["--", path])
        output, code = _run_git(git_args)
        if code != 0:
            return json.dumps({"error": output})

        entries = []
        for line in output.split("\n"):
            if "|" in line:
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    entries.append({
                        "hash": parts[0][:8],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3],
                    })
        return json.dumps({"entries": entries, "count": len(entries)})

    elif action == "blame":
        if not path:
            return json.dumps({"error": "blame requires a path"})
        output, code = _run_git(["blame", "--line-porcelain", path])
        if code != 0:
            return json.dumps({"error": output})
        # Return raw blame (structured parsing is complex)
        if len(output) > MAX_DIFF_CHARS:
            output = output[:MAX_DIFF_CHARS] + "\n[truncated]"
        return json.dumps({"blame": output})

    elif action == "branches":
        output, code = _run_git(["branch", "-a", "--format=%(refname:short)|%(objectname:short)|%(committerdate:relative)"])
        if code != 0:
            return json.dumps({"error": output})
        branches = []
        current_output, _ = _run_git(["branch", "--show-current"])
        for line in output.split("\n"):
            if "|" in line:
                parts = line.split("|", 2)
                branches.append({
                    "name": parts[0],
                    "hash": parts[1] if len(parts) > 1 else "",
                    "last_commit": parts[2] if len(parts) > 2 else "",
                    "current": parts[0] == current_output.strip(),
                })
        return json.dumps({"branches": branches, "current": current_output.strip()})

    elif action == "show":
        git_args = ["show", ref, "--stat"]
        if path:
            git_args = ["show", f"{ref}:{path}"]
        output, code = _run_git(git_args)
        if len(output) > MAX_DIFF_CHARS:
            output = output[:MAX_DIFF_CHARS] + "\n[truncated]"
        return json.dumps({"content": output})

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


def register(registry):
    """Register git tool into the tool registry."""
    registry.register(
        name="git",
        func=git_tool,
        schema=GIT_SCHEMA,
        toolset="git",
    )
