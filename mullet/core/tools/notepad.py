"""Notepad tool — persistent agent scratchpad.

Provides a simple file-based notepad for agents to store and retrieve
notes across multi-step work.

"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Literal

from mullet.core.paths import data_dir

logger = logging.getLogger(__name__)

# Tests point this at a temporary directory; None means notepad/ in the data
# directory. It is created on the first write.
NOTEPAD_DIR: str | None = None


def _notepad_dir() -> str:
    return NOTEPAD_DIR or str(data_dir() / "notepad")


NOTEPAD_SCHEMA = {
    "description": (
        "Manage persistent notes for the current session. "
        "Use for complex tasks or when the user provides multiple tasks. "
        "Call with no parameters to read the default notepad."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "read", "append", "clear", "list"],
                "description": "Action to perform: write, read, append, clear, or list.",
                "default": "read",
            },
            "content": {
                "type": "string",
                "description": "Content to write or append to the notepad. Required for 'write' and 'append' actions.",
            },
            "key": {
                "type": "string",
                "description": "Key for the notepad. Defaults to 'default'.",
                "default": "default",
            },
        },
        "required": [],
    },
}


def _sanitize_key(key: str) -> str:
    """Sanitize notepad key: only allow safe characters, limit length."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", key)[:64]
    return sanitized or "default"


def _get_notepad_path(key: str) -> str:
    sanitized = _sanitize_key(key)
    notepad_dir = _notepad_dir()
    path = Path(os.path.join(notepad_dir, f"{sanitized}.md")).resolve()
    # Verify resolved path is under the notepad directory
    if not path.is_relative_to(Path(notepad_dir).resolve()):
        raise ValueError(f"Path traversal blocked for key: {key}")
    return str(path)


def notepad_handler(
    action: Literal["write", "read", "append", "clear", "list"] = "read",
    content: str | None = None,
    key: str = "default",
    **kwargs,
) -> str:
    """Tool handler — reads or writes the notepad."""
    notepad_path = _get_notepad_path(key)

    if action == "write":
        if content is None:
            return json.dumps({"error": "Content is required for 'write' action."})
        os.makedirs(os.path.dirname(notepad_path), exist_ok=True)
        with open(notepad_path, "w") as f:
            f.write(content)
        return json.dumps({"status": "success", "action": "write", "key": key})

    elif action == "read":
        if not os.path.exists(notepad_path):
            return json.dumps({"status": "success", "action": "read", "key": key, "content": ""})
        with open(notepad_path) as f:
            content = f.read()
        return json.dumps({"status": "success", "action": "read", "key": key, "content": content})

    elif action == "append":
        if content is None:
            return json.dumps({"error": "Content is required for 'append' action."})
        os.makedirs(os.path.dirname(notepad_path), exist_ok=True)
        with open(notepad_path, "a") as f:
            f.write(content)
        return json.dumps({"status": "success", "action": "append", "key": key})

    elif action == "clear":
        if os.path.exists(notepad_path):
            os.remove(notepad_path)
            return json.dumps({"status": "success", "action": "clear", "key": key})
        return json.dumps({"status": "success", "action": "clear", "key": key, "message": "Notepad already empty."})

    elif action == "list":
        notepad_dir = _notepad_dir()
        names = os.listdir(notepad_dir) if os.path.isdir(notepad_dir) else []
        notepad_files = [f for f in names if f.endswith(".md")]
        keys = [os.path.splitext(f)[0] for f in notepad_files]
        return json.dumps({"status": "success", "action": "list", "keys": keys})

    return json.dumps({"error": f"Invalid action: {action}"})


def register(registry):
    """Register notepad tool into the tool registry."""
    registry.register(
        name="notepad",
        func=notepad_handler,
        schema=NOTEPAD_SCHEMA,
        toolset="memory",
    )
