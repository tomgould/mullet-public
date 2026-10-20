"""Utility functions for formatting and displaying file diffs with ANSI colors."""
import difflib
from dataclasses import dataclass

# ANSI escape codes for colors
RED = "\033[31m"
GREEN = "\033[32m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class ToolDiff:
    """Represents a file change made by a tool."""
    tool_name: str
    file_path: str
    old_content: str | None
    new_content: str | None


def format_unified_diff(old_content: str, new_content: str, filename: str, colorize: bool = True, max_lines: int = 50) -> str:
    """
    Formats a unified diff with ANSI coloring and line numbers.
    Truncates diffs longer than max_lines.
    """
    old_lines = old_content.splitlines() if old_content is not None else []
    new_lines = new_content.splitlines() if new_content is not None else []

    # difflib.unified_diff already adds '\n' at the end of each line.
    diff_generator = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}", 
        tofile=f"b/{filename}",   
    )

    diff_list = list(diff_generator)

    # If no diff lines and content is identical, return empty string
    if not diff_list and old_content == new_content:
        return ""

    processed_lines = []
    old_lineno = 1
    new_lineno = 1

    for line in diff_list:
        # difflib output already contains newlines, so we don't add more.
        # We need to strip the newline from the difflib output before adding line numbers,
        # then add it back at the end of the formatted line.
        line_stripped = line.rstrip('\n')

        if line_stripped.startswith('---') or line_stripped.startswith('+++'):
            processed_lines.append(line_stripped)
        elif line_stripped.startswith('@@'):
            parts = line_stripped.split(' ')
            old_range_str = parts[1]
            new_range_str = parts[2]
            
            # Extract starting line numbers from the @@ header
            old_lineno = int(old_range_str.split(',')[0].lstrip('-'))
            new_lineno = int(new_range_str.split(',')[0].lstrip('+'))
            
            processed_lines.append(line_stripped)
        elif line_stripped.startswith('-'):
            processed_lines.append(f"{old_lineno:4d}      {line_stripped}")
            old_lineno += 1
        elif line_stripped.startswith('+'):
            processed_lines.append(f"       {new_lineno:4d}{line_stripped}")
            new_lineno += 1
        else: # Context lines
            processed_lines.append(f"{old_lineno:4d} {new_lineno:4d}{line_stripped}")
            old_lineno += 1
            new_lineno += 1

    output_lines = []
    # The truncation message itself counts as a line.
    # Adjust max_lines to account for the truncation message if it will be added.
    display_lines = processed_lines
    if len(processed_lines) > max_lines:
        output_lines.append(f"Diff for {filename} truncated (showing first {max_lines} lines out of {len(processed_lines)}).\n")
        display_lines = processed_lines[:max_lines]

    for line in display_lines:
        if colorize:
            if line.startswith('---') or line.startswith('+++'):
                output_lines.append(f"{BOLD}{line}{RESET}\n") 
            elif line.startswith('@@'):
                output_lines.append(f"{CYAN}{line}{RESET}\n") 
            elif line.strip().startswith('-'):  # pragma: no cover
                output_lines.append(f"{RED}{line}{RESET}\n")
            elif line.strip().startswith('+'):  # pragma: no cover
                output_lines.append(f"{GREEN}{line}{RESET}\n")
            else:
                output_lines.append(f"{line}\n") 
        else:
            output_lines.append(f"{line}\n") 
            
    return "".join(output_lines)


def format_tool_diff(tool_name: str, file_path: str, old_content: str | None, new_content: str | None, colorize: bool = True, max_lines: int = 50) -> str:
    """
    Formats a tool diff with a header, badge, and unified diff.
    Handles create, delete, and modify cases.
    """
    header_color = BOLD if colorize else ""
    reset_color = RESET if colorize else ""

    diff_output = format_unified_diff(old_content, new_content, file_path, colorize, max_lines)

    # Determine badge and header only if there's actual diff content or specific file change type
    if old_content is None and new_content is not None:
        badge = f"{GREEN}[NEW FILE]{RESET}" if colorize else "[NEW FILE]"
    elif old_content is not None and new_content is None:
        badge = f"{RED}[DELETED]{RESET}" if colorize else "[DELETED]"
    elif diff_output: # Only show modified if there's a diff
        badge = f"{BOLD}[MODIFIED]{RESET}" if colorize else "[MODIFIED]"
    else:
        return "" # No change, no badge, no header, no diff

    header = f"{header_color}--- {tool_name} {badge} {file_path} ---{reset_color}\n"
    
    return f"{header}{diff_output}"


class DiffRenderer:
    """Renders a list of ToolDiff objects."""

    def __init__(self, colorize: bool = True, max_lines: int = 50):
        self.colorize = colorize
        self.max_lines = max_lines

    def render(self, changes: list[ToolDiff]) -> str:
        """Renders a list of ToolDiff objects into a single string."""
        if not changes:
            return "No file changes detected."

        output_parts = []
        for change in changes:
            formatted_diff = format_tool_diff(
                change.tool_name,
                change.file_path,
                change.old_content,
                change.new_content,
                self.colorize,
                self.max_lines
            )
            if formatted_diff:
                output_parts.append(formatted_diff)
        
        if not output_parts:
            return "No file changes detected."

        return "\n".join(output_parts)
