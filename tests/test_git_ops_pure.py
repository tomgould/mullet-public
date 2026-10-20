"""Tests for pure utility functions in mullet.knowledge.sources.git_ops."""
from __future__ import annotations

import pytest

from mullet.knowledge.sources.git_ops import GitError, _sanitise_output, _validate_name


# ---------------------------------------------------------------------------
# _sanitise_output
# ---------------------------------------------------------------------------

class TestSanitiseOutput:
    def test_plain_text_unchanged(self) -> None:
        text = "Cloning into /repos/myproject..."
        assert _sanitise_output(text) == text

    def test_redacts_ghp_token(self) -> None:
        token = "ghp_" + "A" * 36
        text = f"remote: Using token {token} for auth"
        result = _sanitise_output(text)
        assert token not in result
        assert "****" in result

    def test_redacts_github_pat_token(self) -> None:
        token = "github_pat_" + "A" * 82
        text = f"Authorization: token {token}"
        result = _sanitise_output(text)
        assert token not in result
        assert "****" in result

    def test_redacts_gho_token(self) -> None:
        token = "gho_" + "B" * 36
        result = _sanitise_output(f"token={token}")
        assert token not in result

    def test_redacts_token_in_url(self) -> None:
        url = "https://user:ghp_secrettoken1234567890123456789012@github.com/org/repo.git"
        result = _sanitise_output(url)
        assert "ghp_secrettoken" not in result
        assert "****@github.com" in result

    def test_empty_string_unchanged(self) -> None:
        assert _sanitise_output("") == ""

    def test_multiple_tokens_redacted(self) -> None:
        token1 = "ghp_" + "A" * 36
        token2 = "ghs_" + "B" * 36
        text = f"tok1={token1} tok2={token2}"
        result = _sanitise_output(text)
        assert token1 not in result
        assert token2 not in result


# ---------------------------------------------------------------------------
# _validate_name
# ---------------------------------------------------------------------------

class TestValidateName:
    def test_valid_simple_name(self) -> None:
        _validate_name("my-repo")  # Should not raise

    def test_valid_name_with_dots(self) -> None:
        _validate_name("org.project.v2")  # Should not raise

    def test_valid_alphanumeric_underscore(self) -> None:
        _validate_name("my_project_2024")  # Should not raise

    def test_rejects_path_traversal_dotdot(self) -> None:
        with pytest.raises(GitError, match="Invalid"):
            _validate_name("../../../etc/passwd")

    def test_rejects_dotdot_in_name(self) -> None:
        with pytest.raises(GitError, match="Path traversal"):
            _validate_name("repo..name")

    def test_rejects_slash_in_name(self) -> None:
        with pytest.raises(GitError, match="Invalid"):
            _validate_name("org/repo")

    def test_rejects_space_in_name(self) -> None:
        with pytest.raises(GitError, match="Invalid"):
            _validate_name("my repo")

    def test_rejects_semicolon(self) -> None:
        with pytest.raises(GitError, match="Invalid"):
            _validate_name("repo; rm -rf /")

    def test_rejects_command_injection(self) -> None:
        with pytest.raises(GitError, match="Invalid"):
            _validate_name("repo`whoami`")

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(GitError, match="Invalid"):
            _validate_name("")
