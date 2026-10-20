"""Persistent agent memory tool: save / recall / forget key-value entries on disk."""

import json
import os
import time
from typing import Dict, Any, List, Optional

class MemoryStore:
    """Persistent key-value memory store with TTL support."""
    def __init__(self, storage_dir: str = '.mullet/memory'):
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.storage_file = os.path.join(self.storage_dir, 'memory.json')
        self._memory: Dict[str, Dict[str, Any]] = self._load()
        self._save() # Ensure the file exists after initialization

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.storage_file):
            with open(self.storage_file) as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self._memory, f, indent=2)

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Set a key-value pair with optional TTL."""
        self._memory[key] = {
            'value': value,
            'created_at': time.time(),
            'ttl': ttl
        }
        self._save()

    def get(self, key: str) -> Optional[str]:
        """Get value for a key, returns None if expired or not found."""
        entry = self._memory.get(key)
        if entry:
            if entry['ttl'] is not None and (time.time() - entry['created_at']) > entry['ttl']:
                self.delete(key)  # Expire
                return None
            return entry['value']
        return None

    def delete(self, key: str) -> bool:
        """Delete a key, returns True if key existed."""
        if key in self._memory:
            del self._memory[key]
            self._save()
            return True
        return False

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for keys or values containing query string."""
        results = []
        for key, entry in self._memory.items():
            if query in key or query in entry['value']:
                results.append({'key': key, 'value': entry['value']})
        return results

    def list_keys(self) -> List[str]:
        """List all non-expired keys."""
        keys = []
        for key, entry in self._memory.items():
            if entry['ttl'] is None or (time.time() - entry['created_at']) <= entry['ttl']:
                keys.append(key)
        return keys

    def clear(self) -> int:
        """Clear all memory entries and return count cleared."""
        count = len(self._memory)
        self.reset()
        return count

    def reset(self):
        """Drop every stored memory entry and persist the empty store."""
        self._memory = {}
        self._save()

MEMORY_SCHEMA = {
    "name": "memory",
    "description": "A persistent key-value memory store for the agent.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "get", "delete", "search", "list_keys", "clear"],
                "description": "The action to perform on the memory store."
            },
            "key": {
                "type": "string",
                "description": "The key for the memory entry."
            },
            "value": {
                "type": "string",
                "description": "The value for the memory entry."
            },
            "ttl": {
                "type": "integer",
                "description": "Time-to-live for the memory entry in seconds."
            },
            "query": {
                "type": "string",
                "description": "Query string for searching memory entries."
            }
        },
        "required": ["action"]
    }
}

_memory_store = MemoryStore()

def memory_handler(**kwargs) -> str:
    """Handle memory tool actions via the MCP tool interface."""
    global _memory_store
    # Allow overriding the memory store instance for testing
    memory_store_instance = kwargs.pop('memory_store_instance', _memory_store)

    action = kwargs.get('action')
    key = kwargs.get('key')
    value = kwargs.get('value')
    ttl = kwargs.get('ttl')
    query = kwargs.get('query')

    if action == "set":
        if key and value:
            memory_store_instance.set(key, value, ttl)
            return f"Key '{key}' set successfully."
        return "Error: 'key' and 'value' are required for 'set' action."
    elif action == "get":
        if key:
            result = memory_store_instance.get(key)
            return result if result is not None else f"Key '{key}' not found or expired."
        return "Error: 'key' is required for 'get' action."
    elif action == "delete":
        if key:
            if memory_store_instance.delete(key):
                return f"Key '{key}' deleted successfully."
            return f"Key '{key}' not found."
        return "Error: 'key' is required for 'delete' action."
    elif action == "search":
        if query:
            results = memory_store_instance.search(query)
            return json.dumps(results)
        return "Error: 'query' is required for 'search' action."
    elif action == "list_keys":
        keys = memory_store_instance.list_keys()
        return json.dumps(keys)
    elif action == "clear":
        count = memory_store_instance.clear()
        return f"Cleared {count} memory entries."
    else:
        return f"Error: Unknown action '{action}'."

def register(registry):
    """Register the memory tool with the MCP registry."""
    registry.register(
        name="memory",
        func=memory_handler,
        schema=MEMORY_SCHEMA,
        toolset="memory",
    )

if __name__ == '__main__':
    # Example usage for testing
    memory = MemoryStore(storage_dir='/tmp/mullet_memory_test')
    memory.clear()
    print("Cleared memory.")

    memory.set("greeting", "Hello, Mullet!", ttl=1)
    print(f"Set 'greeting': {memory.get('greeting')}")
    time.sleep(1.5)
    print(f"Get 'greeting' after TTL: {memory.get('greeting')}") # Should be None

    memory.set("user_name", "Alice")
    memory.set("user_email", "alice@example.com")
    print(f"Keys: {memory.list_keys()}")

    print(f"Search 'user': {memory.search('user')}")

    print(f"Deleted 'user_name': {memory.delete('user_name')}")
    print(f"Keys after delete: {memory.list_keys()}")

    print(f"Total cleared: {memory.clear()}")
    print(f"Keys after clear: {memory.list_keys()}")
