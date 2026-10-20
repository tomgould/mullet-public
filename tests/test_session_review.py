"""Tests for scripts/session_review.py."""

import unittest
from unittest import mock

from scripts.session_review import (
    _SECTION_ORDER,
    build_review_prompt,
    parse_args,
    parse_sections,
    read_principles,
)


class TestParseArgsDefaults(unittest.TestCase):
    """Test parse_args with no arguments."""

    def test_parse_args_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.since, "HEAD~10")
        self.assertFalse(args.apply)


class TestParseArgsSinceFlag(unittest.TestCase):
    """Test parse_args with --since flag."""

    def test_parse_args_since_flag(self) -> None:
        args = parse_args(["--since", "abc123"])
        self.assertEqual(args.since, "abc123")


class TestBuildReviewPromptContainsSectionHeaders(unittest.TestCase):
    """Test build_review_prompt includes expected section headers."""

    def test_build_review_prompt_contains_section_headers(self) -> None:
        prompt = build_review_prompt(
            principles_content="principles",
            git_context="context",
            since="HEAD~10",
            prior_improvements="improvements",
        )
        for name in _SECTION_ORDER:
            self.assertIn(f"## {name}", prompt)


class TestParseSectionsAllFourKeys(unittest.TestCase):
    """Test parse_sections extracts all four expected sections."""

    def test_parse_sections_all_four_keys(self) -> None:
        response = (
            "## VIOLATIONS\nviolation 1\n\n"
            "## PRINCIPLE_UPDATES\nupdate 1\n\n"
            "## NEW_PRINCIPLES\nnew principle 1\n\n"
            "## IMPROVEMENTS\nimprovement 1\n"
        )
        sections = parse_sections(response)
        self.assertEqual(set(sections.keys()), set(_SECTION_ORDER))
        self.assertEqual(sections["VIOLATIONS"], "violation 1")
        self.assertEqual(sections["PRINCIPLE_UPDATES"], "update 1")
        self.assertEqual(sections["NEW_PRINCIPLES"], "new principle 1")
        self.assertEqual(sections["IMPROVEMENTS"], "improvement 1")


class TestReadPrinciplesMissingFileReturnsNone(unittest.TestCase):
    """Test read_principles returns None when file is missing."""

    @mock.patch("scripts.session_review.PRINCIPLES_PATH")
    def test_read_principles_missing_file_returns_none(self, mock_path: mock.MagicMock) -> None:
        mock_path.exists.return_value = False
        result = read_principles()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
