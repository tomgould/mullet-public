"""Terminal spinner widgets (generic Spinner + tool-aware ToolSpinner)."""

import itertools
import threading
import time
import sys
from enum import Enum

class SpinnerStyle(Enum):
    """Available animation frames for the terminal spinner."""

    DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    LINE = ["-", "\\", "|", "/"]
    BRAILLE = ["⠁", "⠂", "⠄", "⠆", "⠇", "⠃", "⠁"]
    ARROW = ["▹", "▸", "▿", "▾", "◃", "◂", "▵", "▴"]
    BOUNCE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "⠋"]

class Spinner:
    """Thread-based terminal spinner that writes animated frames to stderr."""

    def __init__(self, style: SpinnerStyle, message: str, colorize: bool = True, stderr=sys.stderr):
        """Initialize a spinner with the given style and message."""
        self.spinner_generator = itertools.cycle(style.value)
        self.message = message
        self.colorize = colorize
        self.running = False
        self.spinner_thread = None
        self.color_code = "\033[96m" if colorize else ""
        self.reset_code = "\033[0m" if colorize else ""
        self.stderr = stderr
        self._max_len = len(message)

    def _spinner_task(self):
        while self.running:
            self.stderr.write(f"\r{self.color_code}{next(self.spinner_generator)} {self.message}{self.reset_code}")
            self.stderr.flush()
            time.sleep(0.1)

    def start(self):
        """Start the spinner animation in a background thread."""
        self.running = True
        self._max_len = max(self._max_len, len(self.message))
        self.spinner_thread = threading.Thread(target=self._spinner_task)
        self.spinner_thread.start()

    def update(self, message: str):
        """Update the spinner message while running."""
        self.message = message
        self._max_len = max(self._max_len, len(message))

    def stop(self, final_message: str | None = None):
        """Stop the spinner and optionally print a final message."""
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        self.stderr.write("\r" + " " * (self._max_len + 5) + "\r")  # Clear the line
        if final_message:
            self.stderr.write(f"{final_message}\n")
        self.stderr.flush()

    def __enter__(self):
        """Enter the runtime context and start the spinner."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and stop the spinner."""
        self.stop()

TOOL_EMOJI = {
    "read_file": "📖",
    "write_file": "✏️",
    "terminal": "⚡",
    "search": "🔍",
    "edit_file": "📝",
    "git_status": "📊",
    "think": "🤔",
    "browser": "🌐",
    "embed": "🧮",
    "index": "📥",
}

class ToolSpinner(Spinner):
    """Spinner variant that prefixes the message with a tool emoji and name."""

    def __init__(self, tool_name: str, message: str, style: SpinnerStyle = SpinnerStyle.DOTS, colorize: bool = True, stderr=sys.stderr):
        """Initialize a tool-specific spinner with an emoji prefix."""
        self.tool_name = tool_name
        emoji = TOOL_EMOJI.get(self.tool_name, "🛠️")
        super().__init__(style, f"{emoji} {self.tool_name}: {message}", colorize, stderr)

    def update(self, message: str):
        """Update the tool spinner message while preserving the emoji prefix."""
        emoji = TOOL_EMOJI.get(self.tool_name, "🛠️")
        self.message = f"{emoji} {self.tool_name}: {message}"
        self._max_len = max(self._max_len, len(self.message))
