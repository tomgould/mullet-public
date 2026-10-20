"""Tests for secret_redactor."""

from mullet.harness.security.secret_redactor import redact


class TestSecretRedactor:
    def test_redact_replaces_with_placeholder(self) -> None:
        text = "My key is sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa and some more text"
        redacted, matches = redact(text)
        assert "[REDACTED]" in redacted
        assert "sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in redacted
        assert "and some more text" in redacted

    def test_redact_returns_match_metadata(self) -> None:
        text = "Token: ghp_" + "b" * 36
        redacted, matches = redact(text)
        assert len(matches) == 1
        assert matches[0].pattern_name == "github_token"
        assert matches[0].start == 7
        assert matches[0].end == 7 + 4 + 36

    def test_redact_custom_replacement(self) -> None:
        text = "key: sk-" + "c" * 32
        redacted, _ = redact(text, replacement="***")
        assert "***" in redacted
        assert "[REDACTED]" not in redacted

    def test_redact_no_match_passthrough(self) -> None:
        text = "Nothing here."
        redacted, matches = redact(text)
        assert redacted == text
        assert matches == []
