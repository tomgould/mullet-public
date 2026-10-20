"""Data structure for formatting agent execution output summaries."""
from dataclasses import dataclass
from typing import List

@dataclass
class AgentOutput:
    """Agent execution output summary."""
    response_text: str
    files_changed: List[str]
    commands_run: List[str]
    tokens_used: int
    duration_s: float
    success: bool

    def format_summary(self) -> str:
        """Format the agent output as a summary string."""
        status = "SUCCESS" if self.success else "FAILURE"
        summary = f"Agent Run {status} in {self.duration_s:.2f} seconds using {self.tokens_used} tokens.\n"
        if self.response_text:
            summary += f"Response: {self.response_text}\n"
        if self.files_changed:
            summary += "Files Changed:\n"
            for f in self.files_changed:
                summary += f"  - {f}\n"
        if self.commands_run:
            summary += "Commands Run:\n"
            for cmd in self.commands_run:
                summary += f"  - {cmd}\n"
        return summary
