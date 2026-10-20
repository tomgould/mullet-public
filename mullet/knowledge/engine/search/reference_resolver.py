"""Reference resolution — detect and fetch referenced content in search results."""

import logging
import re

logger = logging.getLogger(__name__)

# Patterns for cross-references in text
_REF_PATTERNS = [
    re.compile(r"see\s+(?:also\s+)?[\"']?([A-Z][^\"'\n,]{3,60})[\"']?", re.IGNORECASE),
    re.compile(r"refer\s+to\s+[\"']?([A-Z][^\"'\n,]{3,60})[\"']?", re.IGNORECASE),
    re.compile(r"(?:Appendix|Section|Chapter|Figure|Table)\s+([A-Z0-9][^,\n]{1,30})", re.IGNORECASE),
    re.compile(r"defined\s+in\s+[`\"']([^`\"'\n]{3,60})[`\"']"),
    re.compile(r"implemented\s+in\s+[`\"']([^`\"'\n]{3,60})[`\"']"),
]


def extract_references(text: str) -> list[str]:
    """Extract cross-references from text content."""
    refs = []
    for pattern in _REF_PATTERNS:
        for match in pattern.finditer(text):
            ref = match.group(1).strip()
            if ref and len(ref) > 2:
                refs.append(ref)
    return list(dict.fromkeys(refs))  # dedupe preserving order


def resolve_references(text: str, searcher=None, top_k: int = 3) -> list[dict]:
    """Find and fetch content for references mentioned in text.

    Returns list of {reference: str, results: list[dict]} for each found reference.
    """
    refs = extract_references(text)
    if not refs or not searcher:
        return []

    resolved = []
    for ref in refs[:5]:  # Limit to 5 references
        try:
            results = searcher.search(ref, top_k=top_k)
            if results:
                resolved.append({"reference": ref, "results": results})
        except Exception as _exc:
            logger.warning("search/reference_resolver: failed to search for reference '%s': %s", ref, _exc)
            pass
    return resolved
