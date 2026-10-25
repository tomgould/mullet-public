"""Tests for web_converter pure helpers."""


from mullet.knowledge.sources.web_converter import (
    clean_markdown,
    extract_title,
)


# ---------------------------------------------------------------------------
# clean_markdown
# ---------------------------------------------------------------------------

class TestCleanMarkdown:
    def test_empty_string_returns_none(self):
        assert clean_markdown("") is None

    def test_whitespace_only_returns_none(self):
        assert clean_markdown("   \n\n  ") is None

    def test_plain_text_returned(self):
        assert clean_markdown("Hello world") == "Hello world"

    def test_collapses_excessive_blank_lines(self):
        md = "A\n\n\n\n\nB"
        result = clean_markdown(md)
        assert "\n\n\n\n" not in result
        assert "A" in result and "B" in result

    def test_removes_advertisement_line(self):
        md = "Before\n\n   Advertisement\n\nAfter"
        result = clean_markdown(md)
        assert "Advertisement" not in result

    def test_removes_most_read_section(self):
        md = "# Article\n\nContent here\n\n## Most Read\n\nJunk 1\nJunk 2\n"
        result = clean_markdown(md)
        assert "Most Read" not in result

    def test_removes_subscribe_line(self):
        md = "Text\n\nSubscribe\n\nMore text"
        result = clean_markdown(md)
        assert "Subscribe" not in result.split()

    def test_removes_empty_links(self):
        md = "Hello [](http://example.com) world"
        result = clean_markdown(md)
        assert "[](http://example.com)" not in result

    def test_returns_stripped_text(self):
        md = "   Hello world   "
        result = clean_markdown(md)
        assert result == "Hello world"

    def test_real_content_preserved(self):
        md = "# Title\n\nThis is real content that should be kept."
        result = clean_markdown(md)
        assert "real content" in result


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_h1_extracted(self):
        html = "<html><body><h1>My Title</h1><p>content</p></body></html>"
        assert extract_title(html) == "My Title"

    def test_title_tag_fallback(self):
        html = "<html><head><title>Page Title</title></head><body><p>text</p></body></html>"
        assert extract_title(html) == "Page Title"

    def test_h1_wins_over_title_tag(self):
        html = "<html><head><title>Page Title</title></head><body><h1>H1 Title</h1></body></html>"
        assert extract_title(html) == "H1 Title"

    def test_no_title_returns_empty(self):
        html = "<html><body><p>no title here</p></body></html>"
        assert extract_title(html) == ""

    def test_empty_html_returns_empty(self):
        assert extract_title("") == ""

    def test_strips_whitespace(self):
        html = "<html><body><h1>  Padded  </h1></body></html>"
        assert extract_title(html) == "Padded"
