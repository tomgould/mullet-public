"""Expands and refines search queries using synonym thesaurus."""
class QueryRefiner:
    """Expands and refines search queries using a synonym thesaurus."""
    def __init__(self):
        self.thesaurus = {
            "search": ["find", "lookup", "discover", "retrieve"],
            "query": ["question", "request", "inquiry"],
            "document": ["file", "record", "text", "article"],
            "tool": ["utility", "instrument", "mechanism"],
            "code": ["program", "script", "software"],
        }

    def expand_query(self, query: str) -> str:
        """Expand a query by adding synonym variants for known terms."""
        words = query.lower().split()
        expanded_words = set(words)
        for word in words:
            if word in self.thesaurus:
                expanded_words.update(self.thesaurus[word])
        return " ".join(sorted(list(expanded_words)))

    def refine(self, query: str, context: str = "") -> str:
        """Refine a query by expanding it with synonyms."""
        # For the basic version, context is ignored.
        # In a more advanced version, context could be used for LLM-assisted refinement.
        return self.expand_query(query)
