"""Tests for mullet.knowledge.sources.web_converter."""

from unittest.mock import MagicMock, patch

import pytest

from mullet.knowledge.sources import web_converter as wc


class TestRemoveJunkElements:
    """Tests for _remove_junk_elements."""

    def test_removes_script_style_nav_header_footer(self):
        html = (
            "<html><head><script>alert(1)</script><style>.x{}</style></head>"
            "<body><nav>nav</nav><header>hdr</header><main>content</main>"
            "<footer>ftr</footer></body></html>"
        )
        soup = wc.BeautifulSoup(html, "lxml")
        wc._remove_junk_elements(soup)
        text = soup.get_text(strip=True)
        assert "nav" not in text
        assert "hdr" not in text
        assert "ftr" not in text
        assert "alert" not in text
        assert ".x{}" not in text
        assert "content" in text

    def test_removes_sidebar_aside(self):
        html = (
            "<body><aside class='sidebar'><p>side</p></aside>"
            "<main>real</main></body>"
        )
        soup = wc.BeautifulSoup(html, "lxml")
        wc._remove_junk_elements(soup)
        text = soup.get_text(strip=True)
        assert "side" not in text
        assert "real" in text

    def test_removes_ad_container_by_class(self):
        html = (
            "<body><div class='ad-slot'><p>ad</p></div>"
            "<main>real</main></body>"
        )
        soup = wc.BeautifulSoup(html, "lxml")
        wc._remove_junk_elements(soup)
        text = soup.get_text(strip=True)
        assert "ad" not in text
        assert "real" in text

    def test_removes_role_navigation(self):
        html = (
            "<body><div role='navigation'><p>nav</p></div>"
            "<main>real</main></body>"
        )
        soup = wc.BeautifulSoup(html, "lxml")
        wc._remove_junk_elements(soup)
        text = soup.get_text(strip=True)
        assert "nav" not in text
        assert "real" in text


class TestHtmlToMarkdown:
    """Tests for html_to_markdown."""

    def test_content_selector_hit(self):
        words = " ".join(f"word{i}" for i in range(60))
        html = f"<article class='entry-content'><p>{words}</p></article>"
        result = wc.html_to_markdown(html)
        assert result is not None
        assert "word0" in result
        assert "word59" in result

    def test_content_selector_uses_h1_inside_content(self):
        words = " ".join(f"word{i}" for i in range(60))
        html = (
            f"<article class='entry-content'>"
            f"<h1>Content Title</h1><p>{words}</p></article>"
        )
        result = wc.html_to_markdown(html)
        assert result is not None
        assert "Content Title" in result

    def test_content_selector_uses_title_tag_when_no_h1(self):
        words = " ".join(f"word{i}" for i in range(60))
        html = (
            f"<html><head><title>Page Title</title></head>"
            f"<body><article class='entry-content'><p>{words}</p></article>"
            f"</body></html>"
        )
        result = wc.html_to_markdown(html)
        assert result is not None
        assert "Page Title" in result
        assert result.startswith("# Page Title")

    def test_trafilatura_fallback_success(self):
        words = " ".join(f"word{i}" for i in range(60))
        html = "<html><body><p>short</p></body></html>"
        with patch.object(wc, "HAS_TRAFILATURA", True):
            with patch.object(
                wc.trafilatura, "extract", return_value=f"# Title\n\n{words}"
            ):
                result = wc.html_to_markdown(html)
        assert result is not None
        assert "word0" in result

    def test_trafilatura_raises_generic_exception(self):
        # Use a div that won't match any content selector so we reach trafilatura
        html = (
            "<html><body><div><p>" + " ".join(f"word{i}" for i in range(60))
            + "</p></div></body></html>"
        )
        with patch.object(wc, "HAS_TRAFILATURA", True):
            with patch.object(
                wc.trafilatura, "extract", side_effect=RuntimeError("boom")
            ):
                with patch.object(wc.logger, "warning") as mock_warn:
                    result = wc.html_to_markdown(html)
        mock_warn.assert_called_once()
        assert result is not None
        assert "word0" in result

    def test_generic_body_fallback_no_trafilatura(self):
        # Use a plain div to avoid matching any content selector
        words = " ".join(f"word{i}" for i in range(60))
        html = f"<body><div><p>{words}</p></div></body>"
        with patch.object(wc, "HAS_TRAFILATURA", False):
            result = wc.html_to_markdown(html)
        assert result is not None
        assert "word0" in result
        assert "word59" in result

    def test_trafilatura_memoryerror_reraised(self):
        html = "<html><body><div><p>" + " ".join(f"word{i}" for i in range(60)) + "</p></div></body></html>"
        with patch.object(wc, "HAS_TRAFILATURA", True):
            with patch.object(wc.trafilatura, "extract", side_effect=MemoryError("oom")):
                with pytest.raises(MemoryError):
                    wc.html_to_markdown(html)

    def test_trafilatura_importerror_reraised(self):
        html = "<html><body><div><p>" + " ".join(f"word{i}" for i in range(60)) + "</p></div></body></html>"
        with patch.object(wc, "HAS_TRAFILATURA", True):
            with patch.object(wc.trafilatura, "extract", side_effect=ImportError("missing")):
                with pytest.raises(ImportError):
                    wc.html_to_markdown(html)

    def test_generic_fallback_removes_junk_tags(self):
        words = " ".join(f"word{i}" for i in range(60))
        html = (
            f"<body><script>alert(1)</script><nav>nav</nav>"
            f"<div><p>{words}</p></div></body>"
        )
        with patch.object(wc, "HAS_TRAFILATURA", False):
            result = wc.html_to_markdown(html)
        assert result is not None
        assert "word0" in result
        assert "alert" not in result
        assert "nav" not in result

    def test_generic_fallback_uses_title_tag(self):
        words = " ".join(f"word{i}" for i in range(60))
        html = (
            f"<html><head><title>Page Title</title></head>"
            f"<body><div><p>{words}</p></div></body></html>"
        )
        with patch.object(wc, "HAS_TRAFILATURA", False):
            result = wc.html_to_markdown(html)
        assert result is not None
        assert "Page Title" in result
        assert result.startswith("# Page Title")

    def test_generic_fallback_uses_h1_tag(self):
        words = " ".join(f"word{i}" for i in range(60))
        html = (
            f"<html><body><h1>Heading One</h1>"
            f"<div><p>{words}</p></div></body></html>"
        )
        with patch.object(wc, "HAS_TRAFILATURA", False):
            result = wc.html_to_markdown(html)
        assert result is not None
        assert "Heading One" in result
        assert result.startswith("# Heading One")

    def test_returns_none_when_no_meaningful_content(self):
        html = "<html><body><p>hi</p></body></html>"
        with patch.object(wc, "HAS_TRAFILATURA", False):
            result = wc.html_to_markdown(html)
        assert result is None


class TestExtractTitle:
    """Tests for extract_title."""

    def test_h1_present(self):
        html = "<html><body><h1>Title Here</h1></body></html>"
        assert wc.extract_title(html) == "Title Here"

    def test_title_tag_no_h1(self):
        html = "<html><head><title>Page Title</title></head><body></body></html>"
        assert wc.extract_title(html) == "Page Title"

    def test_no_title_or_h1(self):
        html = "<html><body><p>nothing</p></body></html>"
        assert wc.extract_title(html) == ""


class TestCleanMarkdown:
    """Tests for clean_markdown."""

    def test_empty_string_returns_none(self):
        assert wc.clean_markdown("") is None

    def test_removes_article_continues_below_advertisement(self):
        md = "Hello\n\nArticle continues below ADVERTISEMENT\n\nWorld"
        result = wc.clean_markdown(md)
        assert "Article continues below" not in result
        assert "Hello" in result
        assert "World" in result

    def test_removes_advertisement_continue_reading(self):
        md = "Start\n\nAdvertisement - Continue Reading Below\n\nEnd"
        result = wc.clean_markdown(md)
        assert "Advertisement" not in result
        assert "Start" in result
        assert "End" in result

    def test_removes_standalone_advertisement(self):
        md = "A\n\nAdvertisement\n\nB"
        result = wc.clean_markdown(md)
        assert "Advertisement" not in result
        assert "A" in result
        assert "B" in result

    def test_removes_relative_date_strings(self):
        for date_str in [
            "5 minutes ago",
            "just now",
            "Updated 2 days ago",
            "one hour ago",
            "an hour ago",
        ]:
            md = f"Text\n\n{date_str}\n\nMore"
            result = wc.clean_markdown(md)
            assert date_str not in result, f"Failed to remove: {date_str}"
            assert "Text" in result
            assert "More" in result

    def test_removes_sign_up_subscribe_lines(self):
        for line in ["Sign up", "Subscribe", "Follow", "Leave this field blank"]:
            md = f"A\n\n{line}\n\nB"
            result = wc.clean_markdown(md)
            assert line not in result, f"Failed to remove: {line}"
            assert "A" in result
            assert "B" in result

    def test_removes_junk_heading_sections(self):
        md = "# Real\n\n## Most Read\nfoo bar\n\n# Next"
        result = wc.clean_markdown(md)
        assert "Most Read" not in result
        assert "foo bar" not in result
        assert "Real" in result
        assert "Next" in result

    def test_removes_empty_links(self):
        md = "Text [](http://example.com) more"
        result = wc.clean_markdown(md)
        assert "[]" not in result
        assert "Text" in result
        assert "more" in result

    def test_removes_empty_images(self):
        md = "Text ![](http://example.com/img.png) more"
        result = wc.clean_markdown(md)
        assert "![]" not in result
        assert "Text" in result
        assert "more" in result

    def test_collapses_excessive_newlines(self):
        md = "A\n\n\n\n\nB"
        result = wc.clean_markdown(md)
        assert "\n\n\n\n" not in result
        assert "A" in result
        assert "B" in result

    def test_preserves_real_content(self):
        md = "# Heading\n\nThis is real content.\n\n- Item 1\n- Item 2"
        result = wc.clean_markdown(md)
        assert "# Heading" in result
        assert "This is real content." in result
        assert "- Item 1" in result
        assert "- Item 2" in result
