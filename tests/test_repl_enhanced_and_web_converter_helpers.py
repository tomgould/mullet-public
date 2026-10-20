"""Tests for mullet.cli.repl_enhanced and mullet.knowledge.sources.web_converter pure helpers."""


from mullet.cli.repl_enhanced import REPLHistory, TabCompleter, MultiLineInput
from mullet.knowledge.sources.web_converter import clean_markdown, extract_title


# ---------------------------------------------------------------------------
# REPLHistory.search
# ---------------------------------------------------------------------------

class TestREPLHistorySearch:
    def setup_method(self, tmp_path=None):
        import tempfile
        import os
        self.tmp = tempfile.mkdtemp()
        self.hist_file = os.path.join(self.tmp, "history.json")
        self.history = REPLHistory(history_file=self.hist_file)
        # Pre-populate with some entries
        self.history.history = ["ls -la", "cd /tmp", "git status", "python hello.py"]

    def test_exact_match_found(self):
        results = self.history.search("git")
        assert any("git" in r for r in results)

    def test_no_match_returns_empty(self):
        results = self.history.search("zzz_not_found_zzz")
        assert results == []

    def test_partial_match_works(self):
        results = self.history.search("python")
        assert len(results) == 1
        assert "python" in results[0]

    def test_multiple_matches(self):
        self.history.history = ["ls -l", "ls -la", "cd /tmp"]
        results = self.history.search("ls")
        assert len(results) == 2


class TestREPLHistoryGetAll:
    def setup_method(self):
        import tempfile
        import os
        self.tmp = tempfile.mkdtemp()
        self.hist_file = os.path.join(self.tmp, "history.json")
        self.history = REPLHistory(history_file=self.hist_file)

    def test_empty_history_returns_empty_list(self):
        assert self.history.get_all() == []

    def test_appended_entries_returned(self):
        self.history.history = ["cmd1", "cmd2"]
        result = self.history.get_all()
        assert "cmd1" in result
        assert "cmd2" in result

    def test_returns_list(self):
        assert isinstance(self.history.get_all(), list)


class TestREPLHistoryAppend:
    def setup_method(self):
        import tempfile
        import os
        self.tmp = tempfile.mkdtemp()
        self.hist_file = os.path.join(self.tmp, "history.json")
        self.history = REPLHistory(history_file=self.hist_file)

    def test_append_adds_to_history(self):
        self.history.append("my command")
        assert "my command" in self.history.get_all()

    def test_max_size_respected(self):
        history = REPLHistory(history_file=self.hist_file, max_size=3)
        history.history = []
        for i in range(5):
            history.append(f"cmd{i}")
        assert len(history.history) <= 3


# ---------------------------------------------------------------------------
# TabCompleter.complete
# ---------------------------------------------------------------------------

class TestTabCompleter:
    def setup_method(self):
        self.completer = TabCompleter(
            slash_commands=["/help", "/exit", "/search", "/status"],
            base_path="/",
        )

    def test_slash_command_match_state_0(self):
        result = self.completer.complete("/h", 0)
        assert result == "/help"

    def test_slash_command_no_match_returns_none(self):
        result = self.completer.complete("/zzz", 0)
        assert result is None

    def test_slash_exact_match_state_0(self):
        result = self.completer.complete("/exit", 0)
        assert result == "/exit"

    def test_slash_multiple_matches_state_1(self):
        # /s matches /search and /status
        match0 = self.completer.complete("/s", 0)
        match1 = self.completer.complete("/s", 1)
        assert match0 is not None
        assert match1 is not None
        assert match0 != match1

    def test_slash_state_beyond_matches_returns_none(self):
        # Only one command starts with /e (/exit)
        result = self.completer.complete("/e", 1)
        assert result is None


# ---------------------------------------------------------------------------
# MultiLineInput._is_incomplete
# ---------------------------------------------------------------------------

class TestMultiLineInputIsIncomplete:
    def setup_method(self):
        self.ml = MultiLineInput()

    def test_complete_json_object_not_incomplete(self):
        assert self.ml._is_incomplete('{"key": "value"}') is False

    def test_incomplete_json_object_is_incomplete(self):
        assert self.ml._is_incomplete('{"key":') is True

    def test_complete_json_array_not_incomplete(self):
        assert self.ml._is_incomplete('[1, 2, 3]') is False

    def test_incomplete_json_array_is_incomplete(self):
        assert self.ml._is_incomplete('[1, 2,') is True

    def test_open_paren_is_incomplete(self):
        assert self.ml._is_incomplete('func(') is True

    def test_balanced_parens_not_incomplete(self):
        assert self.ml._is_incomplete('func()') is False

    def test_ends_with_colon_is_incomplete(self):
        assert self.ml._is_incomplete('if True:') is True

    def test_unclosed_single_quote_is_incomplete(self):
        assert self.ml._is_incomplete("print('hello") is True

    def test_complete_statement_not_incomplete(self):
        assert self.ml._is_incomplete('x = 42') is False


# ---------------------------------------------------------------------------
# clean_markdown
# ---------------------------------------------------------------------------

class TestCleanMarkdown:
    def test_none_input_returns_none(self):
        assert clean_markdown(None) is None

    def test_empty_string_returns_none(self):
        assert clean_markdown("") is None

    def test_clean_text_returned_unchanged(self):
        text = "# Hello\n\nThis is content that should survive."
        result = clean_markdown(text)
        assert "Hello" in result
        assert "content" in result

    def test_strips_advertisement_line(self):
        md = "Some content\n\nAdvertisement\n\nMore content"
        result = clean_markdown(md)
        assert "Advertisement" not in result

    def test_strips_subscribe_line(self):
        md = "Article text\n\nSign up\n\nMore text"
        result = clean_markdown(md)
        assert "Sign up" not in result

    def test_collapses_excessive_blank_lines(self):
        md = "A\n\n\n\n\n\nB"
        result = clean_markdown(md)
        # Should not have 4+ consecutive newlines
        assert "\n\n\n\n" not in result

    def test_removes_empty_links(self):
        md = "Text [](http://example.com) more"
        result = clean_markdown(md)
        assert "[](http://example.com)" not in result

    def test_removes_empty_images(self):
        md = "Text ![](http://example.com/img.png) more"
        result = clean_markdown(md)
        assert "![](http://example.com/img.png)" not in result

    def test_strips_relative_date(self):
        md = "Article\n\n2 hours ago\n\nContent here"
        result = clean_markdown(md)
        assert "2 hours ago" not in result

    def test_strips_continue_reading(self):
        md = "Content\n\nArticle continues below\n\nMore"
        result = clean_markdown(md)
        assert "Article continues below" not in result

    def test_whitespace_only_returns_none(self):
        result = clean_markdown("   \n\n\n  ")
        assert result is None


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_h1_tag_extracted(self):
        html = "<html><body><h1>My Title</h1><p>Content</p></body></html>"
        assert extract_title(html) == "My Title"

    def test_title_tag_fallback(self):
        html = "<html><head><title>Page Title</title></head><body><p>No h1</p></body></html>"
        assert extract_title(html) == "Page Title"

    def test_h1_preferred_over_title(self):
        html = "<html><head><title>Page Title</title></head><body><h1>Real Title</h1></body></html>"
        assert extract_title(html) == "Real Title"

    def test_no_title_returns_empty(self):
        html = "<html><body><p>Just a paragraph</p></body></html>"
        assert extract_title(html) == ""

    def test_empty_html_returns_empty(self):
        result = extract_title("")
        assert result == ""

    def test_h1_strips_whitespace(self):
        html = "<html><body><h1>  Padded Title  </h1></body></html>"
        assert extract_title(html) == "Padded Title"
