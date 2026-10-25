"""Tests for token_viz and web_converter pure helpers."""


from mullet.core.agent.token_viz import TokenUsageViz
from mullet.knowledge.sources.web_converter import (
    clean_markdown,
    extract_title,
)


# ---------------------------------------------------------------------------
# TokenUsageViz
# ---------------------------------------------------------------------------

class TestTokenUsageVizEmpty:
    def test_no_turns_table_message(self):
        viz = TokenUsageViz()
        table = viz.format_table()
        assert "No token usage" in table

    def test_total_all_zeros_when_empty(self):
        viz = TokenUsageViz()
        t = viz.total()
        assert t["total_tokens_in"] == 0
        assert t["total_tokens_out"] == 0
        assert t["total_tokens_combined"] == 0

    def test_total_returns_dict(self):
        viz = TokenUsageViz()
        assert isinstance(viz.total(), dict)


class TestTokenUsageVizRecordTurn:
    def test_record_single_turn(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert len(viz.turns) == 1

    def test_record_multiple_turns(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        assert len(viz.turns) == 2

    def test_cached_defaults_false(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert viz.turns[0]["cached"] is False

    def test_cached_true_stored(self):
        viz = TokenUsageViz()
        viz.record_turn(200, 75, cached=True)
        assert viz.turns[0]["cached"] is True


class TestTokenUsageVizTotal:
    def _make_viz(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75, cached=True)
        viz.record_turn(150, 60)
        return viz

    def test_total_tokens_in(self):
        assert self._make_viz().total()["total_tokens_in"] == 450

    def test_total_tokens_out(self):
        assert self._make_viz().total()["total_tokens_out"] == 185

    def test_total_combined(self):
        assert self._make_viz().total()["total_tokens_combined"] == 635

    def test_cached_tokens_in(self):
        assert self._make_viz().total()["total_cached_tokens_in"] == 200

    def test_cached_tokens_out(self):
        assert self._make_viz().total()["total_cached_tokens_out"] == 75

    def test_cached_combined(self):
        assert self._make_viz().total()["total_cached_tokens_combined"] == 275

    def test_has_all_keys(self):
        t = self._make_viz().total()
        for key in (
            "total_tokens_in", "total_tokens_out", "total_tokens_combined",
            "total_cached_tokens_in", "total_cached_tokens_out", "total_cached_tokens_combined"
        ):
            assert key in t


class TestTokenUsageVizFormatTable:
    def test_returns_string(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert isinstance(viz.format_table(), str)

    def test_has_turn_header(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert "Turn" in viz.format_table()

    def test_has_tokens_in_header(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert "Tokens In" in viz.format_table()

    def test_has_tokens_out_header(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert "Tokens Out" in viz.format_table()

    def test_has_cached_header(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert "Cached" in viz.format_table()

    def test_cached_yes_for_cached_turn(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=True)
        assert "Yes" in viz.format_table()

    def test_cached_no_for_uncached_turn(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=False)
        assert "No" in viz.format_table()

    def test_token_values_in_table(self):
        viz = TokenUsageViz()
        viz.record_turn(12345, 6789)
        table = viz.format_table()
        assert "12345" in table
        assert "6789" in table


# ---------------------------------------------------------------------------
# clean_markdown
# ---------------------------------------------------------------------------

class TestCleanMarkdown:
    def test_returns_none_for_empty(self):
        assert clean_markdown("") is None

    def test_returns_none_for_none(self):
        assert clean_markdown(None) is None

    def test_normal_text_returned(self):
        result = clean_markdown("Hello world, this is some content.")
        assert result is not None
        assert "Hello world" in result

    def test_removes_advertisement_line(self):
        text = "Content here\n\nArticle continues below\n\nMore content"
        result = clean_markdown(text)
        assert "Article continues below" not in result

    def test_removes_advertisement_marker(self):
        text = "Content\n\nAdvertisement\n\nMore"
        result = clean_markdown(text)
        assert result is not None
        assert "\nAdvertisement\n" not in result

    def test_collapses_excessive_blank_lines(self):
        text = "line1\n\n\n\n\nline2"
        result = clean_markdown(text)
        assert "\n\n\n\n" not in result

    def test_removes_empty_links(self):
        text = "Hello [](http://example.com) world, some more content here."
        result = clean_markdown(text)
        assert "[](http://example.com)" not in result

    def test_removes_empty_images(self):
        text = "Hello ![](http://example.com/img.png) world content here."
        result = clean_markdown(text)
        assert "![](http://example.com/img.png)" not in result

    def test_strips_whitespace(self):
        result = clean_markdown("   content here   ")
        assert result == result.strip()

    def test_whitespace_only_returns_none(self):
        result = clean_markdown("   \n\n   \n")
        assert result is None

    def test_subscribe_standalone_removed(self):
        text = "Real content here\n\nSign up\n\nMore content here"
        result = clean_markdown(text)
        assert result is not None
        # Sign up standalone line should be removed
        lines = result.split("\n")
        stripped = [l.strip() for l in lines]
        assert "Sign up" not in stripped


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_title_from_h1(self):
        html = "<html><body><h1>Page Title</h1><p>content</p></body></html>"
        assert extract_title(html) == "Page Title"

    def test_title_from_title_tag(self):
        html = "<html><head><title>Site Title</title></head><body></body></html>"
        assert extract_title(html) == "Site Title"

    def test_h1_takes_precedence_over_title_tag(self):
        html = "<html><head><title>Page Title</title></head><body><h1>H1 Title</h1></body></html>"
        assert extract_title(html) == "H1 Title"

    def test_no_title_returns_empty(self):
        html = "<html><body><p>Just content</p></body></html>"
        assert extract_title(html) == ""

    def test_returns_string(self):
        assert isinstance(extract_title("<html></html>"), str)

    def test_strips_whitespace(self):
        html = "<html><body><h1>  Padded Title  </h1></body></html>"
        assert extract_title(html) == "Padded Title"
