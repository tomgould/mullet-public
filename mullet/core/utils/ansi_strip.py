"""Removes ANSI escape sequences from text strings."""
import re

def strip_ansi(text: str) -> str:
    """Removes all ANSI escape sequences from text."""
    # Pattern for CSI sequences (Control Sequence Introducer)
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    # Pattern for OSC sequences (Operating System Command)
    text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
    return text
