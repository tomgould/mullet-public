"""Tool registration and dispatch system."""

import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Register and resolve tools by name.

    Tools are grouped into toolsets. The agent runner requests schemas
    for specific toolsets and dispatches calls by tool name.

    Supports toolset profiles with probabilistic availability:
    e.g., a "research" profile might give 90% web, 70% browser, 10% terminal.
    """

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}
        self._toolsets: dict[str, list[str]] = {}
        self._profiles: dict[str, dict[str, float]] = {
            "default": {},  # All toolsets at 100%
            "research": {"web": 1.0, "terminal": 0.1, "file": 0.3, "code": 0.1, "delegation": 0.5},
            "coding": {"terminal": 1.0, "file": 1.0, "web": 0.5, "code": 1.0, "delegation": 0.8},
            "minimal": {"terminal": 0.5, "file": 0.5},
        }

    def register(
        self,
        name: str,
        func: Callable[..., str],
        schema: dict,
        toolset: str = "default",
    ):
        """Register a tool with its OpenAI-format schema and handler."""
        self._tools[name] = ToolEntry(name=name, func=func, schema=schema, toolset=toolset)
        self._toolsets.setdefault(toolset, []).append(name)

    def get_schemas(self, toolsets: list[str] | None = None) -> list[dict]:
        """Get OpenAI-format tool schemas for the requested toolsets."""
        if toolsets is None:
            toolsets = list(self._toolsets.keys())

        schemas = []
        for ts in toolsets:
            for tool_name in self._toolsets.get(ts, []):
                entry = self._tools[tool_name]
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": entry.name,
                        "description": entry.schema.get("description", ""),
                        "parameters": entry.schema.get("parameters", {"type": "object", "properties": {}}),
                    },
                })
        return schemas

    def dispatch(self, name: str, arguments: str | dict, task_id: str = "") -> str:
        """Dispatch a tool call by name, returning the result as a string."""
        entry = self._tools.get(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                arguments = {}

        try:
            result = entry.func(**arguments, task_id=task_id)
            if not isinstance(result, str):
                result = json.dumps(result)
            return result
        except TypeError:
            # Handler doesn't accept task_id
            try:
                result = entry.func(**arguments)
                if not isinstance(result, str):
                    result = json.dumps(result)
                return result
            except Exception as e:
                logger.exception("Tool %s failed", name)
                return json.dumps({"error": str(e)})
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return json.dumps({"error": str(e)})

    def get_schemas_for_profile(self, profile: str = "default") -> list[dict]:
        """Get tool schemas filtered by a profile's probability distribution.

        From Hermes: toolset distributions give probabilistic tool availability.
        A "research" profile might suppress terminal but boost web search.
        """
        import random
        profile_weights = self._profiles.get(profile, {})
        if not profile_weights:
            return self.get_schemas()

        schemas = []
        for ts_name, tool_names in self._toolsets.items():
            weight = profile_weights.get(ts_name, 0.0)
            if random.random() < weight:
                for tool_name in tool_names:
                    entry = self._tools[tool_name]
                    schemas.append({
                        "type": "function",
                        "function": {
                            "name": entry.name,
                            "description": entry.schema.get("description", ""),
                            "parameters": entry.schema.get("parameters", {"type": "object", "properties": {}}),
                        },
                    })
        return schemas

    def register_profile(self, name: str, weights: dict[str, float]):
        """Register a custom toolset profile."""
        self._profiles[name] = weights

    def list_tools(self) -> list[str]:
        """Return a list of registered tool names."""
        return list(self._tools.keys())

    def list_toolsets(self) -> list[str]:
        """Return a list of registered toolset names."""
        return list(self._toolsets.keys())

    def list_profiles(self) -> list[str]:
        """Return a list of registered profile names."""
        return list(self._profiles.keys())


class ToolEntry:
    """Internal record for a registered tool."""

    __slots__ = ("func", "name", "schema", "toolset")

    def __init__(self, name: str, func: Callable, schema: dict, toolset: str):
        self.name = name
        self.func = func
        self.schema = schema
        self.toolset = toolset
