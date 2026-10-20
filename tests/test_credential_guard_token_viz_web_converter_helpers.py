"""Tests for credential_guard, token_viz, and web_converter pure helpers."""


from mullet.core.tools.credential_guard import (
    CredentialGuard,
    CREDENTIAL_FILENAMES,
    CREDENTIAL_EXTENSIONS,
    CREDENTIAL_PATTERNS,
)
from mullet.core.agent.token_viz import TokenUsageViz
from mullet.knowledge.sources.web_converter import (
    clean_markdown,
    extract_title,
)


# ---------------------------------------------------------------------------
# CREDENTIAL_FILENAMES / CREDENTIAL_EXTENSIONS
# ---------------------------------------------------------------------------

class TestCredentialConstants:
    def test_env_in_filenames(self):
        assert ".env" in CREDENTIAL_FILENAMES

    def test_credentials_json_in_filenames(self):
        assert "credentials.json" in CREDENTIAL_FILENAMES

    def test_pem_in_extensions(self):
        assert ".pem" in CREDENTIAL_EXTENSIONS

    def test_key_in_extensions(self):
        assert ".key" in CREDENTIAL_EXTENSIONS

    def test_filenames_is_frozenset(self):
        assert isinstance(CREDENTIAL_FILENAMES, frozenset)

    def test_extensions_is_frozenset(self):
        assert isinstance(CREDENTIAL_EXTENSIONS, frozenset)

    def test_patterns_is_list(self):
        assert isinstance(CREDENTIAL_PATTERNS, list)

    def test_patterns_nonempty(self):
        assert len(CREDENTIAL_PATTERNS) > 0


# ---------------------------------------------------------------------------
# CredentialGuard.is_credential_file
# ---------------------------------------------------------------------------

class TestIsCredentialFile:
    def test_dot_env_is_credential(self):
        guard = CredentialGuard()
        assert guard.is_credential_file(".env") is True

    def test_dot_pem_is_credential(self):
        guard = CredentialGuard()
        assert guard.is_credential_file("cert.pem") is True

    def test_id_rsa_is_credential(self):
        guard = CredentialGuard()
        assert guard.is_credential_file("id_rsa") is True

    def test_regular_py_not_credential(self):
        guard = CredentialGuard()
        assert guard.is_credential_file("main.py") is False

    def test_regular_txt_not_credential(self):
        guard = CredentialGuard()
        assert guard.is_credential_file("README.txt") is False

    def test_credentials_json_is_credential(self):
        guard = CredentialGuard()
        assert guard.is_credential_file("credentials.json") is True

    def test_path_with_dir_works(self):
        guard = CredentialGuard()
        assert guard.is_credential_file("/home/user/.env") is True

    def test_dot_key_extension_is_credential(self):
        guard = CredentialGuard()
        assert guard.is_credential_file("server.key") is True

    def test_service_account_json_is_credential(self):
        guard = CredentialGuard()
        assert guard.is_credential_file("service_account.json") is True


# ---------------------------------------------------------------------------
# CredentialGuard.scan_content
# ---------------------------------------------------------------------------

class TestScanContent:
    def test_safe_content_returns_empty(self):
        guard = CredentialGuard()
        result = guard.scan_content("Hello world, this is a normal message")
        assert result == []

    def test_returns_list(self):
        guard = CredentialGuard()
        assert isinstance(guard.scan_content("safe text"), list)

    def test_aws_access_key_detected(self):
        guard = CredentialGuard()
        result = guard.scan_content("key = AKIAIOSFODNN7EXAMPLE123")
        assert len(result) > 0

    def test_github_pat_detected(self):
        guard = CredentialGuard()
        result = guard.scan_content("token = ghp_" + "A" * 36)
        assert len(result) > 0

    def test_api_key_pattern_detected(self):
        guard = CredentialGuard()
        result = guard.scan_content("api_key = my_very_long_api_key_value_here_12345")
        assert len(result) > 0

    def test_password_pattern_detected(self):
        guard = CredentialGuard()
        result = guard.scan_content("password = supersecret123")
        assert len(result) > 0

    def test_pem_private_key_detected(self):
        guard = CredentialGuard()
        result = guard.scan_content("-----BEGIN RSA PRIVATE KEY-----")
        assert len(result) > 0

    def test_jwt_detected(self):
        guard = CredentialGuard()
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk"
        result = guard.scan_content(jwt)
        assert len(result) > 0

    def test_openai_key_detected(self):
        guard = CredentialGuard()
        result = guard.scan_content("sk-" + "A" * 40)
        assert len(result) > 0

    def test_empty_content_returns_empty(self):
        guard = CredentialGuard()
        assert guard.scan_content("") == []


# ---------------------------------------------------------------------------
# CredentialGuard.warn_if_credential
# ---------------------------------------------------------------------------

class TestWarnIfCredential:
    def test_returns_warning_for_credential_file(self):
        guard = CredentialGuard()
        result = guard.warn_if_credential(".env")
        assert result is not None
        assert "WARNING" in result

    def test_returns_none_for_safe_file(self):
        guard = CredentialGuard()
        result = guard.warn_if_credential("main.py")
        assert result is None

    def test_warning_contains_filename(self):
        guard = CredentialGuard()
        result = guard.warn_if_credential("id_rsa")
        assert "id_rsa" in result

    def test_returns_string_or_none(self):
        guard = CredentialGuard()
        result = guard.warn_if_credential(".env")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TokenUsageViz
# ---------------------------------------------------------------------------

class TestTokenUsageVizRecordTurn:
    def test_initial_turns_empty(self):
        viz = TokenUsageViz()
        assert len(viz.turns) == 0

    def test_record_turn_adds_entry(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert len(viz.turns) == 1

    def test_tokens_stored_correctly(self):
        viz = TokenUsageViz()
        viz.record_turn(200, 75)
        assert viz.turns[0]["tokens_in"] == 200
        assert viz.turns[0]["tokens_out"] == 75

    def test_cached_defaults_false(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert viz.turns[0]["cached"] is False

    def test_cached_true_stored(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=True)
        assert viz.turns[0]["cached"] is True

    def test_multiple_turns(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        viz.record_turn(150, 60, cached=True)
        assert len(viz.turns) == 3


# ---------------------------------------------------------------------------
# TokenUsageViz.format_table
# ---------------------------------------------------------------------------

class TestTokenUsageVizFormatTable:
    def test_empty_turns_returns_message(self):
        viz = TokenUsageViz()
        result = viz.format_table()
        assert "No token usage" in result

    def test_returns_string(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert isinstance(viz.format_table(), str)

    def test_contains_turn_number(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        result = viz.format_table()
        assert "1" in result

    def test_contains_token_counts(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        result = viz.format_table()
        assert "100" in result
        assert "50" in result

    def test_contains_cached_column(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=True)
        result = viz.format_table()
        assert "Yes" in result

    def test_multiple_turns_in_table(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        result = viz.format_table()
        assert "2" in result


# ---------------------------------------------------------------------------
# TokenUsageViz.total
# ---------------------------------------------------------------------------

class TestTokenUsageVizTotal:
    def test_returns_dict(self):
        viz = TokenUsageViz()
        assert isinstance(viz.total(), dict)

    def test_empty_turns_all_zeros(self):
        viz = TokenUsageViz()
        t = viz.total()
        assert t["total_tokens_in"] == 0
        assert t["total_tokens_out"] == 0

    def test_sums_tokens_correctly(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        t = viz.total()
        assert t["total_tokens_in"] == 300
        assert t["total_tokens_out"] == 125

    def test_combined_total(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        t = viz.total()
        assert t["total_tokens_combined"] == 150

    def test_cached_tokens_only_from_cached_turns(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75, cached=True)
        t = viz.total()
        assert t["total_cached_tokens_in"] == 200
        assert t["total_cached_tokens_out"] == 75

    def test_no_cached_turns_zero_cached(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        t = viz.total()
        assert t["total_cached_tokens_combined"] == 0

    def test_has_all_expected_keys(self):
        viz = TokenUsageViz()
        t = viz.total()
        expected_keys = {
            "total_tokens_in", "total_tokens_out", "total_tokens_combined",
            "total_cached_tokens_in", "total_cached_tokens_out", "total_cached_tokens_combined",
        }
        assert expected_keys.issubset(set(t.keys()))


# ---------------------------------------------------------------------------
# clean_markdown
# ---------------------------------------------------------------------------

class TestCleanMarkdown:
    def test_returns_none_for_empty(self):
        assert clean_markdown("") is None

    def test_returns_none_for_none(self):
        assert clean_markdown(None) is None

    def test_clean_text_returned_unchanged(self):
        text = "# Title\n\nSome body content here."
        result = clean_markdown(text)
        assert result is not None
        assert "Title" in result
        assert "body content" in result

    def test_advertisement_line_removed(self):
        md = "# Article\n\nArticle continues below\n\nReal content here."
        result = clean_markdown(md)
        assert result is not None
        assert "Article continues below" not in result

    def test_subscribe_line_removed(self):
        md = "# News\n\nSubscribe\n\nRead more here in this article content."
        result = clean_markdown(md)
        assert result is not None
        assert "Subscribe\n" not in result or "Subscribe" not in result.splitlines()

    def test_empty_link_removed(self):
        md = "Check [](http://example.com) this."
        result = clean_markdown(md)
        assert "[](http://example.com)" not in result

    def test_excessive_blank_lines_collapsed(self):
        md = "Line one\n\n\n\n\nLine two"
        result = clean_markdown(md)
        assert result is not None
        assert "\n\n\n\n" not in result

    def test_returns_string_for_valid_content(self):
        result = clean_markdown("# Hello\n\nThis is content worth keeping.")
        assert isinstance(result, str)

    def test_whitespace_only_returns_none(self):
        result = clean_markdown("   \n\n   ")
        assert result is None


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_extracts_h1_title(self):
        html = "<html><body><h1>My Article Title</h1><p>Body.</p></body></html>"
        result = extract_title(html)
        assert result == "My Article Title"

    def test_extracts_title_tag_when_no_h1(self):
        html = "<html><head><title>Page Title</title></head><body><p>Body.</p></body></html>"
        result = extract_title(html)
        assert result == "Page Title"

    def test_h1_preferred_over_title_tag(self):
        html = "<html><head><title>Meta Title</title></head><body><h1>H1 Title</h1></body></html>"
        result = extract_title(html)
        assert result == "H1 Title"

    def test_empty_html_returns_empty_string(self):
        result = extract_title("")
        assert result == ""

    def test_no_title_returns_empty(self):
        html = "<html><body><p>No title here</p></body></html>"
        result = extract_title(html)
        assert result == ""

    def test_returns_string(self):
        result = extract_title("<html></html>")
        assert isinstance(result, str)
