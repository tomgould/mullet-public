"""Clarify tool — structured questions to the user.

Allows the agent to present structured questions with optional
multiple-choice answers. In CLI mode, uses input(). In API mode,
returns the question as a response for the client to handle.

Ported from Hermes tools/clarify_tool.py.
"""

import json
import logging

logger = logging.getLogger(__name__)

MAX_CHOICES = 5

CLARIFY_SCHEMA = {
    "description": (
        "Ask the user a clarifying question when you need more information "
        "before proceeding. Use when the task is ambiguous, has multiple "
        "valid interpretations, or requires user preference.\n\n"
        "Optionally provide up to 5 choices for multiple-choice questions. "
        "The user can pick a choice or type a custom answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user.",
            },
            "choices": {
                "type": "array",
                "description": "Optional multiple-choice answers (max 5).",
                "items": {"type": "string"},
            },
        },
        "required": ["question"],
    },
}

# Platform callback — set by CLI or API to handle user interaction
_user_callback = None


def set_callback(callback):
    """Set the platform-specific callback for user interaction.

    The callback signature: callback(question: str, choices: list[str] | None) -> str
    """
    global _user_callback
    _user_callback = callback


def handle_clarify(question: str, choices: list[str] | None = None, **kwargs) -> str:
    """Ask the user a question and return their response."""
    if not question:
        return json.dumps({"error": "question is required"})

    if choices:
        choices = choices[:MAX_CHOICES]

    callback = kwargs.get("_callback", _user_callback)

    if callback:
        try:
            response = callback(question, choices)
            return json.dumps({"response": response, "question": question})
        except Exception as e:
            logger.warning("Clarify callback failed: %s", e)
            return json.dumps({"error": f"Could not get user input: {e}"})

    # Default: CLI input fallback
    return _cli_clarify(question, choices)


def _cli_clarify(question: str, choices: list[str] | None) -> str:
    """CLI fallback — use input() for user interaction."""
    try:
        print(f"\n{'='*60}")
        print("  Agent needs clarification:")
        print(f"  {question}")
        if choices:
            for i, choice in enumerate(choices, 1):
                print(f"    {i}. {choice}")
            print(f"    {len(choices)+1}. Other (type your answer)")
        print(f"{'='*60}")

        answer = input("  Your answer: ").strip()

        # If they picked a number and we have choices
        if choices and answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(choices):
                answer = choices[idx]

        return json.dumps({"response": answer, "question": question})
    except (EOFError, KeyboardInterrupt):
        return json.dumps({"response": "(no response — non-interactive mode)", "question": question})


def register(registry):
    """Register clarify tool into the tool registry."""
    registry.register(
        name="clarify",
        func=handle_clarify,
        schema=CLARIFY_SCHEMA,
        toolset="interaction",
    )
