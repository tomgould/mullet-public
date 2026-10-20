"""Tests for the Browse harness tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mullet.harness.tools import browser as browser_mod
from mullet.harness.tools.browser import SCHEMA, execute, register
from mullet.harness.commands.browse import handle_browse_command


@pytest.fixture(autouse=True)
def clear_sessions():
    """Reset module-level session state before every test."""
    browser_mod._SESSIONS.clear()
    yield
    browser_mod._SESSIONS.clear()


class TestSchema:
    def test_schema_valid(self):
        assert SCHEMA["name"] == "Browse"
        props = SCHEMA["input_schema"]["properties"]
        assert "action" in props
        assert "url" in props
        assert "selector" in props
        assert "value" in props
        assert "session_id" in props
        assert SCHEMA["input_schema"]["required"] == ["action"]


class TestExecute:
    def test_execute_without_playwright_returns_install_hint(self):
        with patch.object(browser_mod, "sync_playwright", None):
            result = execute("goto", url="https://example.com")

        assert "error" in result
        assert "pip install playwright" in result["error"]

    def _make_mock_playwright(self):
        """Build a mock Playwright stack returning a single page."""
        mock_page = MagicMock()
        mock_page.url = "https://example.com/"
        mock_page.title.return_value = "Example Domain"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_page.goto.return_value = mock_response
        mock_page.reload.return_value = mock_response

        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context

        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_pw.start.return_value = mock_pw

        mock_sync_playwright = MagicMock(return_value=mock_pw)
        return mock_sync_playwright, mock_page, mock_context, mock_browser, mock_pw

    def test_execute_goto(self):
        mock_sync, mock_page, *_ = self._make_mock_playwright()

        with patch(
            "mullet.harness.tools.browser.sync_playwright",
            mock_sync,
        ):
            result = execute("goto", url="https://example.com")

        assert result["url"] == "https://example.com/"
        assert result["title"] == "Example Domain"
        assert result["status_code"] == 200
        mock_page.goto.assert_called_once()

    def test_execute_extract(self):
        mock_sync, mock_page, *_ = self._make_mock_playwright()
        mock_page.inner_text.return_value = "hello world"

        with patch(
            "mullet.harness.tools.browser.sync_playwright",
            mock_sync,
        ):
            result = execute("extract", selector="h1")

        assert result["selector"] == "h1"
        assert result["text"] == "hello world"

    def test_execute_links(self):
        mock_sync, mock_page, *_ = self._make_mock_playwright()
        mock_page.eval_on_selector_all.return_value = [
            {"text": "Home", "href": "https://example.com/"},
            {"text": "About", "href": "https://example.com/about"},
        ]

        with patch(
            "mullet.harness.tools.browser.sync_playwright",
            mock_sync,
        ):
            result = execute("links")

        assert len(result["links"]) == 2
        assert result["links"][0]["text"] == "Home"

    def test_execute_screenshot_returns_path(self):
        mock_sync, mock_page, *_ = self._make_mock_playwright()

        with patch(
            "mullet.harness.tools.browser.sync_playwright",
            mock_sync,
        ):
            result = execute("screenshot")

        assert "path" in result
        assert result["path"].endswith(".png")
        assert result["bytes"] >= 0
        mock_page.screenshot.assert_called_once()

    def test_execute_unknown_action_error(self):
        result = execute("dance")
        assert "error" in result
        assert "Unknown action" in result["error"]

    def test_session_id_reuses_browser_context(self):
        mock_sync, mock_page, mock_context, mock_browser, mock_pw = self._make_mock_playwright()

        with patch(
            "mullet.harness.tools.browser.sync_playwright",
            mock_sync,
        ):
            execute("goto", url="https://a.com", session_id="sess1")
            execute("goto", url="https://b.com", session_id="sess1")

        # Browser should only be launched once for the same session_id
        mock_browser.new_context.assert_called_once()
        assert mock_page.goto.call_count == 2


class TestBrowseCommand:
    def test_browse_command_goto(self):
        with patch("mullet.harness.commands.browse.execute") as mock_execute:
            mock_execute.return_value = {
                "url": "https://example.com",
                "title": "Example",
                "status_code": 200,
            }
            result = handle_browse_command("goto https://example.com")

        assert "Navigated to https://example.com" in result
        mock_execute.assert_called_once_with("goto", url="https://example.com")

    def test_browse_command_close(self):
        mock_sync, mock_page, mock_context, mock_browser, mock_pw = TestExecute()._make_mock_playwright()

        with patch(
            "mullet.harness.tools.browser.sync_playwright",
            mock_sync,
        ):
            # Seed a session so close has something to tear down
            execute("goto", url="https://example.com", session_id="sess1")

        result = handle_browse_command("close sess1")
        assert "closed" in result.lower()
        mock_page.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_pw.stop.assert_called_once()


class TestRegister:
    def test_register_adds_tool(self):
        registry = MagicMock()
        register(registry)
        registry.register.assert_called_once()
        call = registry.register.call_args
        assert call.kwargs["name"] == "Browse"
        assert call.kwargs["toolset"] == "browser"
