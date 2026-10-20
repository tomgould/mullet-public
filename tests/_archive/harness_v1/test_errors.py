"""Tests for mullet.harness.errors."""

from __future__ import annotations

import urllib.error


from mullet.harness.errors import FriendlyError, diagnose, render_error


class TestDiagnose:
    def test_diagnose_import_error_suggests_pip_install(self):
        exc = ImportError("No module named 'pypdf'")
        err = diagnose(exc)
        assert "pypdf" in err.message
        assert any("pip install pypdf" in s for s in err.suggestions)

    def test_diagnose_missing_kimi_key(self):
        exc = ValueError("KIMI_API_KEY not set")
        err = diagnose(exc)
        assert "API key" in err.message
        assert any("KIMI_API_KEY" in s for s in err.suggestions)

    def test_diagnose_file_not_found(self):
        exc = FileNotFoundError(2, "No such file or directory", "/tmp/foo.pdf")
        err = diagnose(exc)
        assert "File not found" in err.message
        assert "/tmp/foo.pdf" in err.message

    def test_diagnose_http_401(self):
        exc = urllib.error.HTTPError(
            url="https://api.moonshot.ai/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )
        err = diagnose(exc)
        assert "Auth" in err.message
        assert any("API key" in s for s in err.suggestions)

    def test_diagnose_http_429(self):
        exc = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "30"},
            fp=None,
        )
        err = diagnose(exc)
        assert "Rate limited" in err.message
        assert any("Wait 30s" in s for s in err.suggestions)

    def test_diagnose_unknown_returns_generic_friendly(self):
        exc = RuntimeError("Something weird happened")
        err = diagnose(exc)
        assert err.message == "RuntimeError: Something weird happened"
        assert any("mullet doctor" in s for s in err.suggestions)


class TestRender:
    def test_render_error_includes_suggestions(self):
        err = FriendlyError(
            message="Oops",
            suggestions=["Try again", "Check docs"],
        )
        text = render_error(err, no_color=True)
        assert "Oops" in text
        assert "Try again" in text
        assert "Check docs" in text

    def test_render_error_no_color_plain_text(self):
        err = FriendlyError(
            message="Plain error",
            cause="Because",
            suggestions=["Fix it"],
            docs_url="https://example.com",
        )
        text = render_error(err, no_color=True)
        assert "Plain error" in text
        assert "Because" in text
        assert "Fix it" in text
        assert "https://example.com" in text
        # Ensure no ANSI escape codes
        assert "\033[" not in text
        assert "\x1b[" not in text
