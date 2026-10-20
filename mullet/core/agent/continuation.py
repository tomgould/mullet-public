"""Continuation enforcement — prevent premature stopping.

Detects when the agent tries to stop with incomplete work and
nudges it to continue. Checks todo list, tool usage, and response
content for signs of premature completion.

Ported from OMC features/continuation-enforcement.ts.
"""

import logging

logger = logging.getLogger(__name__)

# Escalating reminder messages (rotated)
CONTINUATION_REMINDERS = [
    "You have pending tasks. Continue working — don't stop until all items are complete.",
    "There are still items on your task list. Please proceed with the next pending item.",
    "Your work is not done yet. Focus on the next task and keep going.",
    "Don't stop here — you still have pending work. Continue with the next item.",
    "Almost there, but not done. Pick up the next pending task and continue.",
]

# Phrases that indicate the agent is trying to stop prematurely
_STOP_PHRASES = [
    "let me know if you",
    "feel free to ask",
    "is there anything else",
    "hope this helps",
    "i've completed",
    "that covers",
    "that's all",
    "i'm done",
    "task is complete",
    "everything is done",
    "all done",
    "finished",
    "that should be everything",
]

# Phrases that indicate genuine completion
_COMPLETION_PHRASES = [
    "all tests pass",
    "all items completed",
    "no remaining tasks",
    "verified working",
    "committed and pushed",
]


def should_continue(
    response_content: str,
    tool_calls_made: int,
    todo_items: list[dict] | None = None,
    iteration: int = 0,
) -> tuple[bool, str | None]:
    """Check if the agent should continue instead of stopping.

    Returns (should_continue: bool, nudge_message: str | None).
    If True, the nudge message should be injected as a user message.
    """
    if not response_content:
        return False, None

    lower = response_content.lower().strip()

    # Check for genuine completion signals
    if any(phrase in lower for phrase in _COMPLETION_PHRASES):
        return False, None

    # Check if agent used no tools and is just planning/talking
    if tool_calls_made == 0 and len(lower) < 500:
        # Short response with no tools is often just an ack
        # Let the intermediate ack detector handle this instead
        return False, None

    # Check todo list for pending items
    pending_count = 0
    if todo_items:
        pending_count = sum(1 for t in todo_items if t.get("status") in ("pending", "in_progress"))

    # Check for stop phrases
    has_stop_phrase = any(phrase in lower for phrase in _STOP_PHRASES)

    if has_stop_phrase and pending_count > 0:
        reminder_idx = iteration % len(CONTINUATION_REMINDERS)
        nudge = CONTINUATION_REMINDERS[reminder_idx]
        nudge += f"\n\nPending tasks: {pending_count}"

        logger.info("Continuation enforcement: %d pending tasks, nudging agent", pending_count)
        return True, nudge

    return False, None


def format_todo_reminder(todo_items: list[dict]) -> str | None:
    """Format a todo reminder for injection into the conversation.

    Returns None if no pending items.
    """
    if not todo_items:
        return None

    active = [t for t in todo_items if t.get("status") in ("pending", "in_progress")]
    if not active:
        return None

    lines = [f"[Reminder: {len(active)} tasks remaining]"]
    for t in active:
        status = "[>]" if t.get("status") == "in_progress" else "[ ]"
        lines.append(f"  {status} {t.get('id', '?')}. {t.get('content', '')}")

    return "\n".join(lines)
