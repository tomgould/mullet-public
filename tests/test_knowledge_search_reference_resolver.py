"""Tests for mullet.knowledge.engine.search.reference_resolver."""
from unittest.mock import MagicMock
import pytest

from mullet.knowledge.engine.search.reference_resolver import (
    extract_references,
    resolve_references,
)


class TestExtractReferences:
    def test_see_also_pattern(self):
        refs = extract_references("For more information, see also Authentication Guide.")
        assert any("Authentication Guide" in r for r in refs)

    def test_refer_to_pattern(self):
        refs = extract_references("Please refer to API Documentation for details.")
        assert any("API Documentation" in r for r in refs)

    def test_appendix_section_chapter_pattern(self):
        text = "As described in Section 3.1 and Appendix B and Chapter Two overview."
        refs = extract_references(text)
        assert len(refs) >= 1

    def test_defined_in_backticks(self):
        refs = extract_references("The function is defined in `mullet/core/utils.py`.")
        assert any("mullet/core/utils.py" in r for r in refs)

    def test_implemented_in_quotes(self):
        refs = extract_references('The feature is implemented in "providers/kimi.py".')
        assert any("providers/kimi.py" in r for r in refs)

    def test_deduplicates_references(self):
        text = "See Authentication Guide. Also see Authentication Guide."
        refs = extract_references(text)
        count = sum(1 for r in refs if "Authentication Guide" in r)
        assert count == 1

    def test_empty_text_returns_empty(self):
        assert extract_references("") == []

    def test_no_patterns_returns_empty(self):
        assert extract_references("This is plain text with no references.") == []

    def test_short_match_filtered_out(self):
        # Very short matches (len <= 2) should be filtered
        refs = extract_references("See AB.")
        assert not any(len(r) <= 2 for r in refs)


class TestResolveReferences:
    def test_no_refs_returns_empty(self):
        result = resolve_references("plain text", searcher=MagicMock())
        assert result == []

    def test_no_searcher_returns_empty(self):
        result = resolve_references("See also Authentication Guide.", searcher=None)
        assert result == []

    def test_successful_resolution(self):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [{"id": "c1", "content": "auth stuff"}]
        text = "For details, refer to Security Policy document."
        result = resolve_references(text, searcher=mock_searcher, top_k=3)
        assert len(result) >= 1
        assert "reference" in result[0]
        assert "results" in result[0]
        assert result[0]["results"][0]["id"] == "c1"

    def test_empty_search_results_excluded(self):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []  # no results
        text = "Refer to Configuration Manual for details."
        result = resolve_references(text, searcher=mock_searcher)
        assert result == []

    def test_search_exception_silenced(self):
        mock_searcher = MagicMock()
        mock_searcher.search.side_effect = RuntimeError("search down")
        text = "See also Error Handling Guidelines carefully."
        result = resolve_references(text, searcher=mock_searcher)
        assert result == []

    def test_limits_to_five_references(self):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [{"id": "x", "content": "y"}]
        # Create text with >5 references
        text = (
            "See also Authentication Framework module. "
            "Refer to Deployment Configuration guide. "
            "See also Monitoring Dashboard system. "
            "Refer to Error Logging Framework here. "
            "See also Database Migration tools. "
            "Refer to Performance Benchmarks report."
        )
        result = resolve_references(text, searcher=mock_searcher)
        assert len(result) <= 5
