"""LLM citation-extraction helpers: build [1][2]-style maps from retrieved chunks."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class CitationMap:
    """Maps citation numbers to source chunks."""
    chunks: List[Dict] = field(default_factory=list)

    def get(self, num: int) -> Optional[Dict]:
        """Return the chunk for 1-indexed citation ``num``, or None if out of range."""
        if 1 <= num <= len(self.chunks):
            return self.chunks[num - 1]
        return None

    def __len__(self) -> int:
        return len(self.chunks)

def build_citation_map(chunks: List[Dict]) -> CitationMap:
    """
    Build a citation map from chunks.
    """
    return CitationMap(chunks=chunks)

def parse_citations_from_response(response: str) -> List[int]:
    """
    Extract citation numbers [1], [2] etc from LLM response text.
    """
    matches = re.findall(r'\[(\d+)\]', response)
    return [int(match) for match in matches]

def _build_context_parts(
    chunks: List[Dict],
    max_context_chars: int,
    start_index: int = 1,
    citations: bool = False,
) -> tuple[List[str], int, int]:
    """
    Build context parts from chunks within a character budget.
    """
    parts = []
    total_chars = 0
    idx = start_index

    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source_file", "Unknown")
        content = chunk["content"]

        if citations:
            part = f"[{idx}] {source}\n{content}\n"
        else:
            part = f"[Source {idx}: {source}]\n{content}\n"

        if total_chars + len(part) > max_context_chars:
            break

        parts.append(part)
        total_chars += len(part)
        idx += 1

    return parts, total_chars, idx

def build_rag_prompt(question: str, chunks: List[Dict], max_context_tokens: int = 4000) -> str:
    """
    Builds a RAG prompt with numbered source chunks.
    Format: system instruction + numbered sources + question.
    Truncates to max_context_tokens (rough estimate: 4 chars per token).
    """
    max_context_chars = max_context_tokens * 4  # Rough estimate

    context_parts, _, _ = _build_context_parts(
        chunks, max_context_chars, start_index=1, citations=False
    )

    context = "\n---\n".join(context_parts)

    prompt = f"""Based on the following documents, please answer this question:

**Question:** {question}

---

**Context:**

{context}

---

"""
    return prompt

def categorize_chunks(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Groups chunks by type: 'code', 'documentation', 'conversation', 'other'
    Based on chunk metadata.get('chunk_type')
    """
    categories: Dict[str, List[Dict]] = {
        'code': [],
        'documentation': [],
        'conversation': [],
        'other': []
    }

    for chunk in chunks:
        chunk_type = chunk.get("metadata", {}).get("chunk_type")
        if chunk_type in categories:
            categories[chunk_type].append(chunk)
        else:
            categories['other'].append(chunk)
    return categories

def build_rag_prompt_with_citations(question: str, chunks: List[Dict], max_context_tokens: int = 4000) -> str:
    """
    Like build_rag_prompt but instructs the LLM to cite sources as [1], [2], etc.
    """
    max_context_chars = max_context_tokens * 4  # Rough estimate

    context_parts, _, _ = _build_context_parts(
        chunks, max_context_chars, start_index=1, citations=True
    )

    context = "\n---\n".join(context_parts)

    prompt = f"""Based on the following documents, please answer this question:

**Question:** {question}

---

**Context:**

{context}

---

Please provide a clear answer with citations to the sources, like [1], [2], etc.
"""
    return prompt
