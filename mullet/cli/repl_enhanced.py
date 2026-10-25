"""Enhanced REPL widgets — readline history, tab-completion, colored prompts."""
import json
import os
import logging

logger = logging.getLogger(__name__)

class REPLHistory:
    """Manages REPL command history with persistent storage."""
    def __init__(self, history_file=None, max_size=1000):
        if history_file is None:
            from mullet.core.paths import state_dir
            history_file = str(state_dir() / "history.json")
        self.history_file = os.path.expanduser(history_file)
        self.max_size = max_size
        self.history = self._load_history()

    def _load_history(self):
        """Load history from the persistent storage file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as _exc:
                logger.warning("repl_enhanced: Failed to load REPL history: %s", _exc)
                return []
        return []

    def _save_history(self):
        """Save current history to the persistent storage file."""
        try:
            with open(self.history_file, "w") as f:
                    json.dump(self.history[-self.max_size:], f)
        except OSError as _exc:
            logger.warning("repl_enhanced: Failed to save REPL history: %s", _exc)

    def append(self, command):
        """Append a command to the history and save."""
        self.history.append(command)
        self.history = self.history[-self.max_size:] # Truncate history after appending
        self._save_history()

    def search(self, query):
        """Search history for commands containing the query string."""
        return [cmd for cmd in self.history if query in cmd]

    def get_all(self):
        """Return the entire command history."""
        return self.history

class TabCompleter:
    """Provides tab completion for slash commands and file paths."""
    def __init__(self, slash_commands=None, base_path="/"):
        self.slash_commands = slash_commands if slash_commands is not None else []
        self.base_path = base_path

    def complete(self, text, state):
        """Provide tab completion for the given text and state index."""
        matches = []
        if text.startswith("/"):
            # Complete slash commands
            for cmd in self.slash_commands:
                if cmd.startswith(text):
                    matches.append(cmd)
        else:
            # Complete file paths
            dir_path, file_prefix = os.path.split(text)
            current_dir = os.path.join(self.base_path, dir_path) if dir_path else self.base_path

            if os.path.isdir(current_dir):
                for f in os.listdir(current_dir):
                    if f.startswith(file_prefix):
                        full_path = os.path.join(dir_path, f)
                        if os.path.isdir(os.path.join(current_dir, f)):
                            full_path += "/"
                        matches.append(full_path)
        
        if state < len(matches):
            return matches[state]
        return None

class MultiLineInput:
    """Handles multi-line input with automatic continuation for incomplete statements."""
    def __init__(self, prompt=">>> ", indent_prompt="... "):
        self.prompt = prompt
        self.indent_prompt = indent_prompt

    def _is_incomplete(self, code):
        """Check if the given code appears to be incomplete."""
        # Basic check for incomplete Python/JSON. Can be expanded.
        if code.strip().startswith(("{", "[")):
            try:
                json.loads(code)
                return False
            except json.JSONDecodeError:
                return True
        
        # Simple check for unmatched parentheses, brackets, quotes
        unmatched_parens = code.count('(') - code.count(')')
        unmatched_brackets = code.count('[') - code.count(']')
        unmatched_braces = code.count('{') - code.count('}')
        
        # Check for unclosed quotes
        unclosed_single_quote = code.count("'") % 2 != 0
        unclosed_double_quote = code.count('"') % 2 != 0

        if unmatched_parens > 0 or unmatched_brackets > 0 or unmatched_braces > 0 or unclosed_single_quote or unclosed_double_quote:
            return True
        
        if code.endswith(':'): # Likely an incomplete block
            return True
        
        return False

    def get_multiline_input(self):
        """Read multi-line input until a complete statement is entered."""
        lines = []
        while True:
            current_prompt = self.indent_prompt if lines else self.prompt
            try:
                line = input(current_prompt)
                lines.append(line)
                code = "\n".join(lines)
                if not self._is_incomplete(code) and code.strip() != "":
                    break
            except EOFError:
                break # User pressed Ctrl-D
            except KeyboardInterrupt:
                print("^C")
                return None
        return "\n".join(lines)

if __name__ == "__main__":
    # Example Usage
    history = REPLHistory()
    history.append("ls -l")
    history.append("cd /tmp")
    print("History:", history.get_all())
    print("Search 'ls':", history.search("ls"))

    completer = TabCompleter(slash_commands=["/help", "/exit", "/search"], base_path=os.getcwd())
    # You'd typically integrate this with a readline-like interface for actual tab completion
    # For demonstration:
    print("Completion for /h:", completer.complete("/h", 0))
    print("Completion for /e:", completer.complete("/e", 0))

    ml_input = MultiLineInput()
    print("Enter multi-line Python code or JSON. Press Ctrl-D or enter a complete statement to finish.")
    code_input = ml_input.get_multiline_input()
    if code_input:
        print("You entered:\n", code_input)

