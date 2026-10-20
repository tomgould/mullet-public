"""Todo tool — in-session task planning and tracking.

Provides an in-memory task list the agent uses to decompose complex tasks,
track progress, and maintain focus across long conversations. The state
lives on the AgentRunner instance (one per session) and is re-injected
after context compression events.

Ported from Hermes tools/todo_tool.py.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

TODO_SCHEMA = {
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, "
        "status: pending|in_progress|completed|cancelled}\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write. Omit to read current list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique item identifier"},
                        "content": {"type": "string", "description": "Task description"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            },
            "merge": {
                "type": "boolean",
                "description": "true: update by id + add new. false: replace entire list.",
                "default": False,
            },
        },
        "required": [],
    },
}


class TodoStore:
    """In-memory todo list. One instance per session.

    Items are ordered — list position is priority. Each item has:
      - id: unique string identifier (agent-chosen)
      - content: task description
      - status: pending | in_progress | completed | cancelled
    """

    def __init__(self):
        self._items: list[dict[str, str]] = []

    def write(self, todos: list[dict[str, Any]], merge: bool = False) -> list[dict[str, str]]:
        """Write todos. Returns the full current list after writing."""
        if not merge:
            self._items = [self._validate(t) for t in todos]
        else:
            existing = {item["id"]: item for item in self._items}
            for t in todos:
                item_id = str(t.get("id", "")).strip()
                if not item_id:
                    continue

                if item_id in existing:
                    if t.get("content"):
                        existing[item_id]["content"] = str(t["content"]).strip()
                    if t.get("status"):
                        status = str(t["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                else:
                    validated = self._validate(t)
                    existing[validated["id"]] = validated
                    self._items.append(validated)

            # Rebuild preserving order
            seen: set[str] = set()
            rebuilt = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = rebuilt
        return self.read()

    def read(self) -> list[dict[str, str]]:
        """Return a copy of the current todo items."""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """Return whether any todo items exist."""
        return len(self._items) > 0

    def format_for_injection(self) -> str | None:
        """Render active items for post-compression injection."""
        if not self._items:
            return None

        markers = {
            "completed": "[x]", "in_progress": "[>]",
            "pending": "[ ]", "cancelled": "[~]",
        }

        active = [i for i in self._items if i["status"] in ("pending", "in_progress")]
        if not active:
            return None

        lines = ["[Your active task list was preserved across context compression]"]
        for item in active:
            marker = markers.get(item["status"], "[?]")
            lines.append(f"- {marker} {item['id']}. {item['content']} ({item['status']})")
        return "\n".join(lines)

    @staticmethod
    def _validate(item: dict[str, Any]) -> dict[str, str]:
        item_id = str(item.get("id", "")).strip() or "?"
        content = str(item.get("content", "")).strip() or "(no description)"
        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"
        return {"id": item_id, "content": content, "status": status}


# Module-level store instance (shared per process, replaced per session in REPL)
_default_store = TodoStore()


def handle_todo(todos: list | None = None, merge: bool = False, **kwargs) -> str:
    """Tool handler — reads or writes the todo list."""
    store = kwargs.get("_store", _default_store)

    if todos is not None:
        items = store.write(todos, merge)
    else:
        items = store.read()

    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")

    return json.dumps({
        "todos": items,
        "summary": {
            "total": len(items),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
        },
    })


def register(registry):
    """Register todo tool into the tool registry."""
    registry.register(
        name="todo",
        func=handle_todo,
        schema=TODO_SCHEMA,
        toolset="planning",
    )
