"""Orchestration mode registry and selection.

Modes are the high-level execution patterns:
- standard: single agent, no orchestration
- ultrawork: parallel agent execution
- ralph: persistence loop with PRD verification
- autopilot: full lifecycle (plan → execute → verify)
- team: N coordinated agents with file ownership
"""

from enum import Enum


class Mode(str, Enum):
    """Execution mode for Mullet agent orchestration.

    Values map directly to the ``--mode`` CLI flag and ``mode`` API parameter.
    """

    STANDARD = "standard"
    ULTRAWORK = "ultrawork"
    RALPH = "ralph"
    AUTOPILOT = "autopilot"
    TEAM = "team"
    ECO = "eco"


# Mode hierarchy (from OMC):
# autopilot includes ralph which includes ultrawork
MODE_HIERARCHY = {
    Mode.AUTOPILOT: [Mode.RALPH, Mode.ULTRAWORK],
    Mode.RALPH: [Mode.ULTRAWORK],
    Mode.ULTRAWORK: [],
    Mode.TEAM: [],
    Mode.STANDARD: [],
}

# Mode types
MODE_TYPES = {
    Mode.AUTOPILOT: "standalone",
    Mode.TEAM: "standalone",
    Mode.RALPH: "wrapper",
    Mode.ULTRAWORK: "component",
    Mode.STANDARD: "default",
    Mode.ECO: "modifier",
}


def detect_magic_keyword(text: str) -> Mode | None:
    """Detects magic keywords in user messages and returns the corresponding Mode."""
    lower_text = text.lower()
    if "#ralph" in lower_text:
        return Mode.RALPH
    if "#ultrawork" in lower_text:
        return Mode.ULTRAWORK
    if "#autopilot" in lower_text:
        return Mode.AUTOPILOT
    if "#team" in lower_text:
        return Mode.TEAM
    if "#eco" in lower_text:
        return Mode.ECO
    return None


def select_mode(prompt: str) -> Mode:
    """Select execution mode from user input.

    Decision tree (from OMC mode-selection-guide):
    - Want autonomous execution?
      - Parallelizable into 3+ components? → team
      - Otherwise → autopilot
    - Want parallel execution? → ultrawork
    - Want persistence until verified? → ralph
    - Otherwise → standard
    """
    # Check for magic keywords first
    magic_mode = detect_magic_keyword(prompt)
    if magic_mode:
        return magic_mode

    # Mode is typically specified explicitly via CLI subcommand or skill.
    # This auto-detection is a fallback for API/WebSocket requests.
    lower = prompt.lower()

    if "autopilot" in lower or "autonomous" in lower:
        return Mode.AUTOPILOT
    if "team" in lower and any(w in lower for w in ["agents", "workers", "parallel team"]):
        return Mode.TEAM
    if "ralph" in lower or "until done" in lower or "don't stop" in lower:
        return Mode.RALPH
    if "ultrawork" in lower or "parallel" in lower:
        return Mode.ULTRAWORK

    return Mode.STANDARD
