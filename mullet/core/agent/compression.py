"""Context compression — summarize conversation when approaching limits.

From Hermes ContextCompressor: when conversation tokens exceed a threshold,
generate a structured summary and replace old messages. Preserves the system
prompt (which contains Trove context) and recent messages.

Mullet addition: auto-save key findings to Trove memory before compressing.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from mullet.core.knowledge.manager import KnowledgeManager
    from mullet.core.providers.base import Provider

logger = logging.getLogger(__name__)

# Rough token estimation: 1 token ≈ 4 chars
CHARS_PER_TOKEN = 4

# Model context windows (tokens) — used for compression thresholds
MODEL_CONTEXT_LENGTHS = {
    "gpt-4": 128_000,
    "gpt-4o": 128_000,
    "gpt-3.5": 16_000,
    "claude-opus": 200_000,
    "claude-sonnet": 200_000,
    "claude-haiku": 200_000,
    "gemini": 1_000_000,
    "mistral": 32_000,
    "llama": 32_000,
    "default": 128_000,
}


def get_context_length(model: str = "") -> int:
    """Get context window size for a model.

    Uses comprehensive model_info database first, with local fallback.
    """
    try:
        from mullet.core.providers.model_info import get_context_length as _db_get
        length = _db_get(model)
        if length > 0:
            return length
    except ImportError:
        pass

    # Fallback to local table
    model_lower = model.lower()
    for key, length in MODEL_CONTEXT_LENGTHS.items():
        if key in model_lower:
            return length
    return MODEL_CONTEXT_LENGTHS["default"]


@dataclass
class StructuredSummary:
    """A structured summary of a conversation segment."""
    key_decisions: list[str] = field(default_factory=list)
    code_changes: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    context: str = ""

    def to_message_content(self) -> str:
        """Render the summary as a message content string."""
        parts = []
        if self.key_decisions:
            parts.append("Key Decisions:\n" + "\n".join(f"- {d}" for d in self.key_decisions))
        if self.code_changes:
            parts.append("Code Changes:\n" + "\n".join(f"- {c}" for c in self.code_changes))
        if self.open_questions:
            parts.append("Open Questions:\n" + "\n".join(f"- {q}" for q in self.open_questions))
        if self.context:
            parts.append("Context:\n" + self.context)
        
        if not parts:
            return "Previous conversation context (compressed)."
        
        return "\n\n".join(parts)


class ContextCompressor:
    """Compress conversation when approaching context limits.

    From Hermes ContextCompressor pattern:
    - threshold_percent: compress at this % of context window
    - Generates structured summary via LLM
    - Preserves system prompt (Trove context = source of truth)
    - Preserves recent messages (tail budget)
    - Auto-saves key findings to Trove memory before compressing
    """

    def __init__(
        self,
        context_length: int = 128_000,
        threshold_percent: float = 0.50,
        protect_first_n: int = 2,
        protect_last_n: int = 6,
        tail_protection: int = 5,  # New parameter
        summary_target_ratio: float = 0.20,
    ):
        self.context_length = context_length
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.tail_protection = tail_protection  # Initialize new parameter
        self.summary_target_ratio = summary_target_ratio

    def should_compress(self, messages: list[dict]) -> bool:
        """Check if conversation should be compressed."""
        total_chars = sum(len(self._msg_text(m)) for m in messages)
        estimated_tokens = total_chars // CHARS_PER_TOKEN
        threshold = int(self.context_length * self.threshold_percent)
        return estimated_tokens > threshold

    def compress(
        self,
        messages: list[dict],
        provider: "Provider | None" = None,
        knowledge: "KnowledgeManager | None" = None,
        existing_summary: StructuredSummary | None = None,
    ) -> list[dict]:
        """Compress messages, optionally persisting findings to Trove.

        Returns a new, shorter message list with a summary replacing
        the middle section.
        """
        if len(messages) <= self.protect_first_n + self.tail_protection + 1: # Adjusted for tail_protection
            return messages  # Too short to compress

        # Pre-pass: prune old tool results cheaply before expensive summarization
        # From Hermes: replace old tool outputs (>200 chars) with placeholder
        messages = self._prune_old_tool_outputs(messages)

        # Split into protected head, compressible middle, protected tail
        head = messages[:self.protect_first_n]
        # Use tail_protection for the actual tail of messages to preserve
        tail = messages[-self.tail_protection:]
        middle = messages[self.protect_first_n:-self.tail_protection]

        if not middle:
            return messages

        # Auto-save findings to Trove before compressing
        if knowledge:
            self._save_findings(middle, knowledge)

        # Generate summary
        if provider:
            new_summary = self._generate_structured_summary(middle, provider)
            if existing_summary:
                summary = self.iterative_update(existing_summary, new_summary)
            else:
                summary = new_summary
        else:
            # Fallback to simple summary, wrapped in StructuredSummary
            summary_text = self._simple_summary(middle)
            summary = StructuredSummary(context=summary_text)

        # Build compressed message list
        compressed = list(head)
        compressed.append({
            "role": "user",
            "content": f"[Context Summary]\n{summary.to_message_content()}\n[End Summary — conversation continues below]",
        })
        compressed.extend(tail)

        before_tokens = sum(len(self._msg_text(m)) for m in messages) // CHARS_PER_TOKEN
        after_tokens = sum(len(self._msg_text(m)) for m in compressed) // CHARS_PER_TOKEN
        logger.info("Compressed: %d → %d tokens (%d messages → %d)",
                     before_tokens, after_tokens, len(messages), len(compressed))

        return compressed

    def iterative_update(self, existing_summary: StructuredSummary, new_summary_segment: StructuredSummary) -> StructuredSummary:
        """
        Merges a new summary segment into an existing StructuredSummary,
        appending new information and deduplicating.
        """
        updated_summary = StructuredSummary(
            key_decisions=list(existing_summary.key_decisions),
            code_changes=list(existing_summary.code_changes),
            open_questions=list(existing_summary.open_questions),
            context=existing_summary.context,
        )

        # Append and deduplicate key decisions
        for decision in new_summary_segment.key_decisions:
            if decision not in updated_summary.key_decisions:
                updated_summary.key_decisions.append(decision)

        # Append and deduplicate code changes
        for change in new_summary_segment.code_changes:
            if change not in updated_summary.code_changes:
                updated_summary.code_changes.append(change)

        # Append and deduplicate open questions
        for question in new_summary_segment.open_questions:
            if question not in updated_summary.open_questions:
                updated_summary.open_questions.append(question)
        
        # Concatenate context, possibly with some basic deduplication or summarization if it grows too large
        if new_summary_segment.context:
            if updated_summary.context:
                # Simple concatenation for now, could be more sophisticated
                updated_summary.context = f"{updated_summary.context}\n\n{new_summary_segment.context}"
            else:
                updated_summary.context = new_summary_segment.context

        return updated_summary

    def _generate_structured_summary(self, messages: list[dict], provider: "Provider") -> StructuredSummary:
        """Use the LLM to generate a structured summary of the middle messages."""
        conversation_text = self._format_messages(messages)
        max_chars = int(len(conversation_text) * self.summary_target_ratio)

        summary_prompt = [
            {"role": "system", "content": (
                "You are an expert summarizer. Analyze the following conversation segment and extract "
                "key decisions, significant code changes, and any open questions. "
                "Format your output as a JSON object with the following keys: "
                "'key_decisions' (list of strings), "
                "'code_changes' (list of strings), "
                "'open_questions' (list of strings), and "
                "'context' (a general summary string). "
                f"Keep the 'context' summary under {max_chars} characters. "
                "Example: {'key_decisions': ['Decision A'], 'code_changes': ['File X modified'], 'open_questions': [], 'context': 'General overview.'}"
            )},
            {"role": "user", "content": conversation_text},
        ]

        try:
            max_tokens_for_summary = max(1, max_chars // CHARS_PER_TOKEN)
            response = provider.create(
                messages=summary_prompt,
                max_tokens=max_tokens_for_summary,
                temperature=0.3,
                response_format={"type": "json_object"}, # Request JSON object
            )
            summary_data = json.loads(response.content)
            return StructuredSummary(
                key_decisions=summary_data.get("key_decisions", []),
                code_changes=summary_data.get("code_changes", []),
                open_questions=summary_data.get("open_questions", []),
                context=summary_data.get("context", ""),
            )
        except Exception as e:
            logger.warning("LLM structured summary failed, using simple summary: %s", e)
            return StructuredSummary(context=self._simple_summary(messages))

    def _simple_summary(self, messages: list[dict]) -> str:
        """Generate a simple extractive summary without LLM."""
        parts = []
        tool_calls = []
        assistant_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant" and content:
                assistant_messages.append(content[:200])
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name", "?")
                    tool_calls.append(name)

        if tool_calls:
            unique_tools = list(dict.fromkeys(tool_calls))
            parts.append(f"Tools used: {', '.join(unique_tools)}")

        if assistant_messages:
            parts.append("Key points:")
            for msg in assistant_messages[-5:]:  # Last 5 assistant messages
                parts.append(f"- {msg}")

        return "\n".join(parts) if parts else "Previous conversation context (compressed)."

    def _save_findings(self, messages: list[dict], knowledge: "KnowledgeManager"):
        # This method's implementation would depend on the KnowledgeManager API
        # For now, it remains as a placeholder, as per original file.
        pass

    def _prune_old_tool_outputs(self, messages: list[dict]) -> list[dict]:
        """Replaces old, verbose tool outputs with a placeholder to save tokens."""
        pruned_messages = []
        for msg in messages:
            if msg.get("role") == "tool" and len(msg.get("content", "")) > 200:
                pruned_messages.append({
                    "role": "tool",
                    "content": "[Tool output compressed due to length]",
                    "tool_call_id": msg.get("tool_call_id", "unknown")
                })
            else:
                pruned_messages.append(msg)
        return pruned_messages

    def _msg_text(self, message: dict) -> str:
        """Extracts text content from a message for token estimation."""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle multimodal content, extract text parts
            text_parts = [part["text"] for part in content if part.get("type") == "text"]
            return " ".join(text_parts)
        return ""

    def _format_messages(self, messages: list[dict]) -> str:
        """Formats a list of messages into a readable string for summarization."""
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = self._msg_text(msg)
            if content:
                formatted_parts.append(f"{role}: {content}")
            elif msg.get("tool_calls"):
                tool_calls_str = ", ".join([tc["function"]["name"] for tc in msg["tool_calls"]])
                formatted_parts.append(f"{role} (Tool Calls): {tool_calls_str}")
        return "\n\n".join(formatted_parts)