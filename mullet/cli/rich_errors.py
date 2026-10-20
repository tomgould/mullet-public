"""Colorful error formatting with actionable suggestions for common exceptions."""

import re

from mullet.hud.colors import RED as _RED, YELLOW as _YELLOW, CYAN as _CYAN, RESET as _RESET

def format_error(error, context=None):
    """
    Formats an error message with color and provides suggested fixes based on known patterns.

    Args:
        error (Exception): The exception object.
        context (str, optional): Additional context about where the error occurred.

    Returns:
        str: A formatted, colored error string.
    """
    error_type = type(error).__name__
    error_message = str(error)
    formatted_message = f"{_RED}Error: {error_type}{_RESET} - {error_message}"

    suggestion = ""
    if isinstance(error, ImportError):
        # Try to extract module name from error message
        match = re.search(r"No module named '([^']+)'", str(error))
        module_name = match.group(1) if match else str(error)
        suggestion = f"{_YELLOW}Suggestion: Try running `pip install {module_name}`{_RESET}"
    elif isinstance(error, FileNotFoundError):
        suggestion = f"{_YELLOW}Suggestion: Please check if the file path is correct and the file exists.{_RESET}"
    elif isinstance(error, ConnectionError):
        suggestion = f"{_YELLOW}Suggestion: Check your network connection or the target server's status.{_RESET}"
    elif isinstance(error, ValueError):
        suggestion = f"{_YELLOW}Suggestion: Review the input values. This error often indicates incorrect arguments.{_RESET}"
    elif isinstance(error, TypeError):
        suggestion = f"{_YELLOW}Suggestion: Check the types of arguments passed to functions or methods.{_RESET}"

    if context:
        formatted_message += f"\n{_CYAN}Context:{_RESET} {context}"
    if suggestion:
        formatted_message += f"\n{suggestion}"

    return formatted_message
