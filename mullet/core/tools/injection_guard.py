"""Detects prompt injection attempts in LLM inputs using pattern matching."""
import re
from dataclasses import dataclass, field
from typing import List

@dataclass
class InjectionResult:
    """Result of a prompt injection check."""
    is_suspicious: bool = False
    confidence: float = 0.0
    patterns_matched: List[str] = field(default_factory=list)

class InjectionGuard:
    """Detects prompt injection attempts in LLM inputs using pattern matching."""
    def __init__(self):
        # Patterns to detect prompt injection
        self.patterns = {
            "role_play": re.compile(r"\b(act as|you are now|assume the role of|play the role of)\b", re.IGNORECASE),
            "system_override": re.compile(r"\b(system prompt|override all prior instructions|disregard previous commands)\b", re.IGNORECASE),
            "ignore_instructions": re.compile(r"\b(ignore (all|previous) instructions|disregard (all|everything) (before this|i told you)|forget everything (i told you|before this)|stop what you are doing (and|then)|do not follow (previous|any) instructions|you must ignore|you are to ignore|drop all previous instructions|override your instructions|you are (not|no longer) to)\b", re.IGNORECASE),
            "base64_commands": re.compile(r"(?:[A-Za-z0-9+/]{4}){8,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)") # At least 32 chars of base64 with padding
        }

    def check(self, text: str) -> InjectionResult:
        """Check the given text for prompt injection patterns."""
        is_suspicious = False
        confidence = 0.0
        matched_patterns = []

        for name, pattern in self.patterns.items():
            if pattern.search(text):
                is_suspicious = True
                matched_patterns.append(name)
                # Increase confidence based on the number of patterns matched
                confidence = min(confidence + 0.3, 1.0)

        return InjectionResult(is_suspicious, round(confidence, 2), matched_patterns)
