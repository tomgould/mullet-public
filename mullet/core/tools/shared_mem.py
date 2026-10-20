"""Shared Memory tool — persistent agent memory.

Provides a simple file-based shared memory for agents to store and retrieve
data across multi-step work and sessions.
"""

import json
import logging
from typing import Any, Literal

from mullet.core.orchestration.shared_memory import SharedMemory

logger = logging.getLogger(__name__)

# Initialize SharedMemory
shared_memory = SharedMemory()

SHARED_MEM_SCHEMA = {
    "description": (
        "Manage persistent shared memory for the current session. "
        "Use for complex tasks or when the user provides multiple tasks. "
        "Call with no parameters to list all keys in the default namespace."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "get", "delete", "list_keys", "list_namespaces"],
                "description": "Action to perform: set, get, delete, list_keys, or list_namespaces.",
                "default": "list_keys",
            },
            "key": {
                "type": "string",
                "description": "Key for the shared memory item. Required for 'set', 'get', and 'delete' actions.",
            },
            "value": {
                "type": "string",
                "description": "Value to store in shared memory. Required for 'set' action.",
            },
            "namespace": {
                "type": "string",
                "description": "Namespace for the shared memory item. Defaults to 'default'.",
                "default": "default",
            },
            "ttl": {
                "type": "integer",
                "description": "Time-to-live in seconds for the shared memory item. 0 means no expiry. Only applicable for 'set' action.",
                "default": 0,
            },
        },
        "required": [],
    },
}


def shared_mem_handler(
    action: Literal["set", "get", "delete", "list_keys", "list_namespaces"] = "list_keys",
    key: str | None = None,
    value: Any | None = None,
    namespace: str = "default",
    ttl: int = 0,
    **kwargs,
) -> str:
    """Tool handler — interacts with the SharedMemory."""
    try:
        if action == "set":
            if key is None or value is None:
                return json.dumps({"error": "Key and value are required for 'set' action."})
            shared_memory.set(key, value, namespace, ttl)
            return json.dumps({"status": "success", "action": "set", "key": key, "namespace": namespace})

        elif action == "get":
            if key is None:
                return json.dumps({"error": "Key is required for 'get' action."})
            result = shared_memory.get(key, namespace)
            return json.dumps({"status": "success", "action": "get", "key": key, "namespace": namespace, "value": result})

        elif action == "delete":
            if key is None:
                return json.dumps({"error": "Key is required for 'delete' action."})
            result = shared_memory.delete(key, namespace)
            return json.dumps({"status": "success", "action": "delete", "key": key, "namespace": namespace, "deleted": result})

        elif action == "list_keys":
            keys = shared_memory.list_keys(namespace)
            return json.dumps({"status": "success", "action": "list_keys", "namespace": namespace, "keys": keys})

        elif action == "list_namespaces":
            namespaces = shared_memory.list_namespaces()
            return json.dumps({"status": "success", "action": "list_namespaces", "namespaces": namespaces})

        return json.dumps({"error": f"Invalid action: {action}"})
    except Exception as e:
        logger.exception(f"Error in shared_mem_handler for action {action}: {e}")
        return json.dumps({"error": str(e)})


def register(registry):
    """Register shared_mem tool into the tool registry."""
    registry.register(
        name="shared_mem",
        func=shared_mem_handler,
        schema=SHARED_MEM_SCHEMA,
        toolset="memory",
    )
