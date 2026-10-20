"""User-friendly error messages with fix suggestions.

Every error the user can encounter should have a clear explanation
and actionable fix. No stack traces in normal output.
"""

import logging

logger = logging.getLogger(__name__)

ERROR_SUGGESTIONS = {
    "connection refused": {
        "message": "Cannot connect to LLM provider",
        "suggestions": [
            "Start LM Studio (port 1234) or Ollama (port 11434)",
            "Check that the provider is running: curl http://localhost:1234/v1/models",
            "Or use --base-url to specify a different endpoint",
        ],
    },
    "rate limit": {
        "message": "Provider rate limit reached",
        "suggestions": [
            "Wait a moment and try again (Mullet has automatic retry)",
            "Switch to a different provider: --provider local",
            "Set up multiple API keys for rotation",
        ],
    },
    "api key": {
        "message": "API key issue",
        "suggestions": [
            "Check your API key is set: echo $ANTHROPIC_API_KEY",
            "Regenerate the key at your provider's dashboard",
            "Or use a local LLM (no key needed): mullet --local-only",
        ],
    },
    "timeout": {
        "message": "Request timed out",
        "suggestions": [
            "The LLM may be overloaded. Try again in a moment",
            "Increase timeout: --max-iterations 120",
            "Switch to a faster model: --model mistral-nemo",
        ],
    },
    "context length": {
        "message": "Input too long for model's context window",
        "suggestions": [
            "Reduce the input size or be more specific",
            "Use a model with larger context: --model claude-sonnet-4-6",
            "Enable context compression (automatic in long conversations)",
        ],
    },
    "tool": {
        "message": "Tool execution failed",
        "suggestions": [
            "Check the tool arguments in the error above",
            "Verify file paths exist: ls /path/to/file",
            "Check permissions: the tool may need write access",
        ],
    },
}


def format_error(error: Exception) -> str:
    """Format an error with suggestions for the user."""
    err_msg = str(error).lower()

    for key, info in ERROR_SUGGESTIONS.items():
        if key in err_msg:
            lines = [f"\033[31mError:\033[0m {info['message']}"]
            lines.append(f"  Detail: {str(error)[:200]}")
            lines.append("")
            lines.append("  Suggestions:")
            for s in info["suggestions"]:
                lines.append(f"    - {s}")
            return "\n".join(lines)

    # Unknown error — generic message
    return (
        f"\033[31mError:\033[0m {str(error)[:200]}\n"
        f"\n  Run 'mullet doctor' to check your setup."
    )
