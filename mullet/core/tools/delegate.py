"""Delegation tool — allows agents to spawn child agents for subtasks."""

import json

DELEGATE_SCHEMA = {
    "description": "Delegate a subtask to a child agent. The child runs independently with its own context and tools, and returns a summary. Use for tasks that can be handled in isolation.",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Clear description of what the child agent should accomplish.",
            },
            "context": {
                "type": "string",
                "description": "Additional context for the child agent.",
            },
        },
        "required": ["goal"],
    },
}

# Will be set by the CLI/server when building the agent
_parent_agent = None


def set_parent_agent(agent):
    """Set the parent agent for delegation. Called during agent setup."""
    global _parent_agent
    _parent_agent = agent


def delegate(goal: str, context: str = "", **_kwargs) -> str:
    """Delegate a subtask to a child agent."""
    if _parent_agent is None:
        return json.dumps({"error": "Delegation not available — no parent agent set"})

    # Propagate delegation depth from parent
    parent_depth = getattr(_parent_agent, "_delegation_depth", 0)

    from mullet.core.agent.delegation import delegate_task
    result = delegate_task(
        goal=goal,
        context=context,
        parent=_parent_agent,
        _depth=parent_depth,
    )

    return json.dumps({
        "completed": result.completed,
        "summary": result.summary,
        "error": result.error,
    })


def register(registry):
    """Register the delegate tool into a ToolRegistry."""
    registry.register(
        name="delegate",
        func=delegate,
        schema=DELEGATE_SCHEMA,
        toolset="delegation",
    )
