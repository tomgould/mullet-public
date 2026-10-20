"""Dangerous command detection and approval.

Detects potentially destructive commands before execution and requires
approval. Per-session approval state with permanent allowlist support.

Ported from Hermes tools/approval.py.
"""

import logging
import re
import threading
import unicodedata

logger = logging.getLogger(__name__)

# Dangerous command patterns: (regex, description)
DANGEROUS_PATTERNS = [
    (r'\brm\s+(-[^\s]*\s+)*/', "delete in root path"),
    (r'\brm\s+-[^\s]*r', "recursive delete"),
    (r'\brm\s+--recursive\b', "recursive delete (long flag)"),
    (r'\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b', "world-writable permissions"),
    (r'\bmkfs\b', "format filesystem"),
    (r'\bdd\s+.*if=', "disk copy"),
    (r'>\s*/dev/sd', "write to block device"),
    (r'\bDROP\s+(TABLE|DATABASE)\b', "SQL DROP"),
    (r'\bDELETE\s+FROM\b(?!.*\bWHERE\b)', "SQL DELETE without WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    (r'>\s*/etc/', "overwrite system config"),
    (r'\bsystemctl\s+(stop|disable|mask)\b', "stop/disable system service"),
    (r'\bkill\s+-9\s+-1\b', "kill all processes"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    (r'\b(curl|wget)\b.*\|\s*(ba)?sh\b', "pipe remote content to shell"),
    (r'\bxargs\s+.*\brm\b', "xargs with rm"),
    (r'\bfind\b.*-exec\s+(/\S*/)?rm\b', "find -exec rm"),
    (r'\bfind\b.*-delete\b', "find -delete"),
    (r'\b(cp|mv|install)\b.*\s/etc/', "copy/move into /etc/"),
    (r'\bsed\s+-[^\s]*i.*\s/etc/', "in-place edit of system config"),
    (r'\bgit\s+push\s+.*--force\b', "force push"),
    (r'\bgit\s+reset\s+--hard\b', "hard reset"),
    (r'\bgit\s+clean\s+-[^\s]*f', "force clean untracked files"),
]

# Compiled patterns for performance
_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), desc) for p, desc in DANGEROUS_PATTERNS]


class ApprovalState:
    """Per-session approval state. Thread-safe.

    Tracks which dangerous pattern categories have been approved
    for this session. Approvals are session-scoped — they don't
    persist across restarts.
    """

    def __init__(self, allowlist: set[str] | None = None):
        """Initialize approval state with an optional permanent allowlist."""
        self._approved: set[str] = set()
        self._allowlist: set[str] = allowlist or set()  # Permanent config allowlist
        self._lock = threading.Lock()

    def is_approved(self, description: str) -> bool:
        """Check if a dangerous pattern category is already approved."""
        with self._lock:
            return description in self._approved or description in self._allowlist

    def approve(self, description: str):
        """Approve a single dangerous pattern category for this session."""
        with self._lock:
            self._approved.add(description)

    def approve_all(self, descriptions: list[str]):
        """Approve multiple dangerous pattern categories for this session."""
        with self._lock:
            self._approved.update(descriptions)


def detect_dangerous_command(command: str) -> list[tuple[str, str]]:
    """Check a command for dangerous patterns.

    Returns list of (pattern_regex, description) for all matches.
    Empty list means the command is safe.
    """
    if not command or not command.strip():
        return []

    # Normalize: strip ANSI, null bytes, fullwidth chars
    normalized = _normalize(command)

    matches = []
    for pattern, description in _COMPILED_PATTERNS:
        if pattern.search(normalized):
            matches.append((pattern.pattern, description))

    return matches


def check_and_approve(
    command: str,
    state: ApprovalState,
    approval_callback=None,
) -> tuple[bool, str]:
    """Check a command and get approval if needed.

    Returns (approved: bool, message: str).
    If the command is safe or already approved, returns (True, "").
    If denied, returns (False, reason).
    """
    matches = detect_dangerous_command(command)
    if not matches:
        return True, ""

    # Check if all matches are already approved
    unapproved = [(p, d) for p, d in matches if not state.is_approved(d)]
    if not unapproved:
        return True, ""

    descriptions = [d for _, d in unapproved]
    desc_text = ", ".join(descriptions)

    # If no callback, auto-deny
    if not approval_callback:
        logger.warning("Dangerous command blocked (no approval callback): %s", desc_text)
        return False, f"Blocked: {desc_text}. No approval mechanism available."

    # Ask for approval
    try:
        approved = approval_callback(command, descriptions)
        if approved:
            state.approve_all(descriptions)
            return True, ""
        return False, f"User denied: {desc_text}"
    except Exception as e:
        logger.warning("Approval callback failed: %s", e)
        return False, f"Approval failed: {e}"


def cli_approval_callback(command: str, descriptions: list[str]) -> bool:
    """CLI interactive approval — asks user to confirm dangerous commands."""
    try:
        print(f"\n{'='*60}")
        print("  Dangerous command detected:")
        print(f"  $ {command[:200]}")
        print("")
        for desc in descriptions:
            print(f"  - {desc}")
        print(f"{'='*60}")
        answer = input("  Allow this command? [y/N]: ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _normalize(command: str) -> str:
    """Normalize command for pattern matching."""
    # Strip null bytes
    command = command.replace("\0", "")
    # Normalize fullwidth characters (bypass attempts)
    normalized = []
    for ch in command:
        cat = unicodedata.category(ch)
        if cat.startswith("Lo") or cat.startswith("Cf"):
            continue  # Skip CJK/format chars used for obfuscation
        normalized.append(unicodedata.normalize("NFKC", ch))
    return "".join(normalized)
