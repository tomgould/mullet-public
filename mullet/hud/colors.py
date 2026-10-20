"""ANSI color utilities for HUD rendering."""
import re

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BRIGHT_BLUE = "\033[94m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from ``text``."""
    return _ANSI_RE.sub("", text)

def bold(t: str) -> str:
    """Wrap ``t`` in bold ANSI codes."""
    return f"{BOLD}{t}{RESET}"
def dim(t: str) -> str:
    """Wrap ``t`` in dim ANSI codes."""
    return f"{DIM}{t}{RESET}"
def red(t: str) -> str:
    """Wrap ``t`` in red ANSI codes."""
    return f"{RED}{t}{RESET}"
def green(t: str) -> str:
    """Wrap ``t`` in green ANSI codes."""
    return f"{GREEN}{t}{RESET}"
def yellow(t: str) -> str:
    """Wrap ``t`` in yellow ANSI codes."""
    return f"{YELLOW}{t}{RESET}"
def blue(t: str) -> str:
    """Wrap ``t`` in blue ANSI codes."""
    return f"{BLUE}{t}{RESET}"
def magenta(t: str) -> str:
    """Wrap ``t`` in magenta ANSI codes."""
    return f"{MAGENTA}{t}{RESET}"
def cyan(t: str) -> str:
    """Wrap ``t`` in cyan ANSI codes."""
    return f"{CYAN}{t}{RESET}"
def white(t: str) -> str:
    """Wrap ``t`` in white ANSI codes."""
    return f"{WHITE}{t}{RESET}"
def bright_blue(t: str) -> str:
    """Wrap ``t`` in bright-blue ANSI codes."""
    return f"{BRIGHT_BLUE}{t}{RESET}"

def ctx_color(pct: float) -> str:
    """Return the ANSI colour that matches a context-usage percentage (>=85 red, >=70 yellow, else green)."""
    if pct >= 85:
        return RED
    if pct >= 70:
        return YELLOW
    return GREEN

def bar(pct: float, width: int = 10) -> str:
    """Render a ``width``-cell progress bar for ``pct`` (0-100) coloured by ``ctx_color``."""
    filled = round(min(100, max(0, pct)) / 100 * width)
    color = ctx_color(pct)
    return f"{color}{chr(0x2588) * filled}{DIM}{chr(0x2591) * (width - filled)}{RESET}"
