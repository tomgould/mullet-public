"""Tests for secret_scanner."""


from mullet.harness.security.secret_scanner import has_secrets, scan


class TestSecretScanner:
    def test_scans_aws_key(self) -> None:
        text = "My AWS key is AKIAIOSFODNN7EXAMPLE."
        matches = scan(text)
        assert len(matches) == 1
        assert matches[0].pattern_name == "aws_access_key"
        assert matches[0].severity == "high"
        assert "AKIA" in matches[0].matched_substring

    def test_scans_github_token(self) -> None:
        text = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        matches = scan(text)
        assert len(matches) == 1
        assert matches[0].pattern_name == "github_token"
        assert matches[0].severity == "high"

    def test_scans_openai_key(self) -> None:
        text = "sk-" + "a" * 32
        matches = scan(text)
        assert len(matches) == 1
        assert matches[0].pattern_name == "openai_key"
        assert matches[0].severity == "high"

    def test_scans_anthropic_key(self) -> None:
        text = "sk-ant-api03-" + "b" * 32
        matches = scan(text)
        assert len(matches) == 1
        assert matches[0].pattern_name == "anthropic_key"
        assert matches[0].severity == "high"

    def test_scans_jwt(self) -> None:
        text = "Bearer eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY3ODkwIiw.name.eyJhbGciOiJIUzI1NiIs"
        matches = scan(text)
        assert len(matches) == 1
        assert matches[0].pattern_name == "jwt"
        assert matches[0].severity == "medium"

    def test_no_match_returns_empty(self) -> None:
        text = "Hello world, nothing secret here."
        matches = scan(text)
        assert matches == []
        assert has_secrets(text) is False

    def test_severity_classification(self) -> None:
        text = "password = 'supersecret123'"
        matches = scan(text)
        assert len(matches) == 1
        assert matches[0].pattern_name == "generic_password"
        assert matches[0].severity == "low"

    def test_has_secrets_helper(self) -> None:
        assert has_secrets("key: sk-" + "x" * 32) is True
        assert has_secrets("clean text") is False
