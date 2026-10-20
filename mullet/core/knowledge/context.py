"""Context builder — formats knowledge for system prompt injection.

Takes a KnowledgeContext and the base system prompt, producing
the final prompt with project knowledge embedded.
"""

from mullet.core.knowledge.manager import KnowledgeContext


def build_system_prompt(
    base_prompt: str,
    knowledge_context: KnowledgeContext | None = None,
    project_name: str = "",
) -> str:
    """Build the full system prompt with knowledge context injected.

    Structure:
    1. Base system prompt (agent identity)
    2. Project context header (if client specified)
    3. Search results (relevant chunks)
    4. Memories (stored context from previous sessions)
    """
    parts = [base_prompt]

    if knowledge_context and knowledge_context.markdown:
        if project_name:
            parts.append(f"\n# Project: {project_name}\n")
        parts.append(knowledge_context.markdown)

    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate — 1 token per 4 characters."""
    return len(text) // 4


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...context truncated to fit token budget]"
