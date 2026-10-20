"""Performs fuzzy string matching with similarity scoring."""
from difflib import SequenceMatcher

def fuzzy_match(query, candidates, threshold=0.6):
    """
    Performs fuzzy string matching between a query and a list of candidates.
    Returns a list of (candidate, similarity_ratio) tuples, sorted by similarity,
    for candidates exceeding the given threshold.
    """
    matches = []
    for candidate in candidates:
        matcher = SequenceMatcher(None, query.lower(), candidate.lower())
        ratio = matcher.ratio()
        if ratio > threshold:
            matches.append((candidate, ratio))
    return sorted(matches, key=lambda x: x[1], reverse=True)
