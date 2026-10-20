"""Detect + resolve textual references (Section, Appendix, RFC, etc.) in docs."""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Pattern

@dataclass
class Reference:
    """Represents a detected cross-document reference."""
    type: str  # e.g., "Section", "Appendix", "Table", "Figure"
    identifier: str  # e.g., "X", "A", "1", "3.2"
    position: int  # Start index of the reference in the text
    confidence: float = 1.0  # Confidence score of the detection

# Predefined regex patterns for different document types
# Patterns are compiled for efficiency and made case-insensitive.
REFERENCE_PATTERNS: Dict[str, List[Pattern[str]]] = {
    "tech_docs": [
        re.compile(r"(Section|Sec\.)\s+([A-Z0-9\.-]+)\b", re.IGNORECASE),
        re.compile(r"(Appendix|App\.)\s+([A-Z0-9\.-]+)\b", re.IGNORECASE),
        re.compile(r"(Table|Tab\.)\s+([0-9\.-]+)\b", re.IGNORECASE),
        re.compile(r"(Figure|Fig\.)\s+([0-9\.-]+)\b", re.IGNORECASE),
    ],
    "legal_docs": [
        re.compile(r"(Article)\s+([0-9\.]+)\b", re.IGNORECASE),
        re.compile(r"(Paragraph)\s+([0-9\.]+)\b", re.IGNORECASE),
        re.compile(r"(§|Section)\s+([0-9\.]+)\b", re.IGNORECASE),
        re.compile(r"(Exhibit)\s+([A-Z0-9]+)\b", re.IGNORECASE),
    ],
    "academic_docs": [
        re.compile(r"(Chapter|Ch\.)\s+([0-9\.-]+)\b", re.IGNORECASE),
        re.compile(r"(Section|Sec\.)\s+([0-9\.-]+)\b", re.IGNORECASE),
        re.compile(r"(Appendix|App\.)\s+([A-Z0-9\.-]+)\b", re.IGNORECASE),
        re.compile(r"(Table|Tab\.)\s+([0-9\.-]+)\b", re.IGNORECASE),
        re.compile(r"(Figure|Fig\.)\s+([0-9\.-]+)\b", re.IGNORECASE),
        re.compile(r"(Equation)\s+([0-9\.-]+)\b", re.IGNORECASE),
    ],
}

def detect_references(text: str, doc_type: str = "tech_docs") -> List[Reference]:
    """
    Detects cross-document references in the given text using configurable regex patterns.

    Args:
        text: The input text to scan for references.
        doc_type: The type of document to determine which set of regex patterns to use
                  ("tech_docs", "legal_docs", "academic_docs"). Defaults to "tech_docs".

    Returns:
        A list of Reference objects found in the text.
    """
    found_references: List[Reference] = []
    patterns = REFERENCE_PATTERNS.get(doc_type, [])

    for pattern in patterns:
        for match in pattern.finditer(text):
            ref_type = match.group(1)
            ref_identifier = match.group(2)
            position = match.start()
            found_references.append(
                Reference(
                    type=ref_type.replace("Sec.", "Section").replace("App.", "Appendix").replace("Tab.", "Table").replace("Fig.", "Figure").replace("Ch.", "Chapter").replace("§", "Section").title(),
                    identifier=ref_identifier,
                    position=position,
                    confidence=1.0
                )
            )
    return found_references

def resolve_reference(ref: Reference, document_index: Dict[str, str]) -> Optional[Dict[str, any]]:
    """
    Looks up the target of a given reference within a document index.

    Args:
        ref: The Reference object to resolve.
        document_index: A dictionary where keys are document identifiers (e.g., file names,
                        document IDs) and values are the full text content of those documents.

    Returns:
        A dictionary containing the 'document_id', 'match_text', and 'match_position' if
        the reference is resolved, otherwise None.
    """
    # Create a regex pattern to find the reference identifier within documents.
    # We escape the identifier to handle special regex characters.
    # This assumes the identifier itself is a direct string to be found.
    search_pattern = re.compile(re.escape(f"{ref.type} {ref.identifier}"), re.IGNORECASE)

    for doc_id, doc_content in document_index.items():
        match = search_pattern.search(doc_content)
        if match:
            return {
                "document_id": doc_id,
                "match_text": match.group(0),
                "match_position": match.start(),
                "confidence": 1.0 # Confident if found
            }
    return None
