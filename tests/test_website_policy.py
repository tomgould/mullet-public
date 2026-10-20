"""Tests for website policy / robots.txt checker."""
from unittest.mock import patch
from mullet.core.tools.website_policy import RobotsTxtChecker


def test_check_returns_dict():
    checker = RobotsTxtChecker()
    with patch("urllib.robotparser.RobotFileParser") as MockRP:
        mock_rp = MockRP.return_value
        mock_rp.can_fetch.return_value = True
        mock_rp.crawl_delay.return_value = None
        result = checker.check_robots_txt("https://example.com/page")
        assert "allowed" in result
        assert result["allowed"] is True


def test_disallowed_url():
    checker = RobotsTxtChecker()
    checker._cache.clear()
    with patch("urllib.robotparser.RobotFileParser") as MockRP:
        mock_rp = MockRP.return_value
        mock_rp.can_fetch.return_value = False
        mock_rp.crawl_delay.return_value = 5
        result = checker.check_robots_txt("https://blocked.com/secret")
        assert result["allowed"] is False
        assert "/secret" in result["disallowed_paths"]


def test_error_handling():
    checker = RobotsTxtChecker()
    checker._cache.clear()
    with patch("urllib.robotparser.RobotFileParser") as MockRP:
        mock_rp = MockRP.return_value
        mock_rp.read.side_effect = Exception("Network error")
        result = checker.check_robots_txt("https://fail.com/page")
        assert result["allowed"] is False
        assert "error" in result
