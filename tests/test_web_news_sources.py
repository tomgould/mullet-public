"""Tests for news-site web sources: path allow-list, sitemap seeding, document
downloads, published-date extraction, frontmatter metadata, and the
searcher's recency boost."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from mullet.knowledge.sources.base import SyncResult
from mullet.knowledge.sources.web_cache import CrawlCache
from mullet.knowledge.sources.web_converter import extract_published_date
from mullet.knowledge.sources.web_crawler import (
    WebSourceConfig,
    WebSourceConnector,
    _document_extension,
    _download_filename,
    _parse_sitemap,
    normalize_link,
)

BASE = "https://news.example.com"
PDF_BYTES = b"%PDF-1.4 fake document body"


def _response(status=200, text="", content=None, content_type="text/html; charset=utf-8", headers=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = text
    resp.content = content if content is not None else text.encode()
    resp.headers = {"content-type": content_type, **(headers or {})}
    return resp


def _page(title, published=None, links=(), nonce="", extra=""):
    """An article page; ``published`` adds JouwWeb-style microdata.

    ``nonce`` varies per request on real sites without changing the content;
    ``extra`` is appended to the article body.
    """
    meta = f'<meta itemprop="datePublished" content="{published}">' if published else ""
    anchors = "".join(f'<a href="{link}">more</a>' for link in links)
    body = " ".join(["Substantial article body text for the crawler to keep."] * 20)
    return (
        f'<html><head><title>{title}</title><script nonce="{nonce}">var x = 1;</script>'
        f"</head><body><article><h1>{title}</h1>{meta}<p>{body}</p>{extra}{anchors}"
        f"</article></body></html>"
    )


def _pdf(filename="Report One.pdf"):
    return _response(
        content=PDF_BYTES, content_type="application/pdf",
        headers={"content-disposition": f'attachment; filename="{filename}"'},
    )


def _crawl(tmp_path, config, routes, cache=None):
    """Run _crawl against a URL -> response (or HTML string) map; unknown URLs 404."""
    def get(url, headers=None):
        found = routes.get(url, _response(404))
        return _response(text=found) if isinstance(found, str) else found

    client = MagicMock()
    client.get.side_effect = get
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    cache = cache or CrawlCache(str(tmp_path))
    result = SyncResult(source_type="web")
    with patch("httpx.Client", return_value=client), patch("time.sleep"):
        WebSourceConnector(config)._crawl(tmp_path, cache, result, set())
    fetched = [c.args[0] for c in client.get.call_args_list]
    return result, fetched, cache


class TestExtractPublishedDate:
    def test_meta_article_published_time(self):
        html = '<html><head><meta property="article:published_time" content="2025-03-01T10:00:00Z"></head></html>'
        assert extract_published_date(html) == "2025-03-01T10:00:00+00:00"

    def test_json_ld_converted_to_utc(self):
        html = (
            '<script type="application/ld+json">'
            '{"@type": "NewsArticle", "datePublished": "2025-03-01T10:00:00+0100"}</script>'
        )
        assert extract_published_date(html) == "2025-03-01T09:00:00+00:00"

    def test_json_ld_graph_entry(self):
        html = (
            '<script type="application/ld+json">'
            '{"@graph": [{"@type": "WebPage"}, {"@type": "Article", "datePublished": "2024-12-25"}]}</script>'
        )
        assert extract_published_date(html) == "2024-12-25T00:00:00+00:00"

    def test_microdata(self):
        html = _page("Post", published="2026-01-18T11:28:56+00:00")
        assert extract_published_date(html) == "2026-01-18T11:28:56+00:00"

    def test_listing_page_with_several_dates_is_undated(self):
        html = (
            '<div itemprop="datePublished" content="2026-01-18">a</div>'
            '<div itemprop="datePublished" content="2025-10-07">b</div>'
        )
        assert extract_published_date(html) is None

    def test_no_date(self):
        assert extract_published_date(_page("Evergreen")) is None

    def test_unparseable_date(self):
        html = '<meta property="article:published_time" content="last Tuesday">'
        assert extract_published_date(html) is None

    def test_rfc2822_date(self):
        html = '<meta name="pubdate" content="Tue, 07 Oct 2025 09:00:00 GMT">'
        assert extract_published_date(html) == "2025-10-07T09:00:00+00:00"

    def test_odd_length_fractional_seconds(self):
        html = '<meta property="article:published_time" content="2025-03-01T10:00:00.12Z">'
        assert extract_published_date(html) == "2025-03-01T10:00:00.120000+00:00"

    def test_null_graph_does_not_raise(self):
        html = '<script type="application/ld+json">{"@graph": null, "datePublished": "2024-12-25"}</script>'
        assert extract_published_date(html) == "2024-12-25T00:00:00+00:00"


class TestCrawlerPathsAndSitemap:
    def test_about_prefixed_news_skipped_by_default(self, tmp_path):
        routes = {BASE + "/": _page("Home", links=["/about-us/1_post"])}
        _result, fetched, _ = _crawl(tmp_path, WebSourceConfig(url=BASE + "/", wait_time=0), routes)
        assert BASE + "/about-us/1_post" not in fetched

    def test_allow_paths_overrides_skip_list(self, tmp_path):
        routes = {
            BASE + "/": _page("Home", links=["/about-us/1_post"]),
            BASE + "/about-us/1_post": _page("News post"),
        }
        config = WebSourceConfig(url=BASE + "/", wait_time=0, allow_paths=[r"^/about-us/"])
        _crawl(tmp_path, config, routes)
        assert (tmp_path / "about-us_1_post.md").exists()

    def test_sitemap_seeds_unlinked_pages(self, tmp_path):
        sitemap = f"<urlset><url><loc>{BASE}/orphan</loc></url></urlset>"
        routes = {
            BASE + "/sitemap.xml": _response(text=sitemap, content_type="text/xml"),
            BASE + "/": _page("Home"),
            BASE + "/orphan": _page("Orphan"),
        }
        _crawl(tmp_path, WebSourceConfig(url=BASE + "/", wait_time=0, use_sitemap=True), routes)
        assert (tmp_path / "orphan.md").exists()

    def test_sitemap_index_followed(self, tmp_path):
        index = f"<sitemapindex><sitemap><loc>{BASE}/posts.xml</loc></sitemap></sitemapindex>"
        child = f"<urlset><url><loc>{BASE}/deep&amp;post</loc></url></urlset>"
        routes = {
            BASE + "/sitemap.xml": _response(text=index, content_type="text/xml"),
            BASE + "/posts.xml": _response(text=child, content_type="text/xml"),
            BASE + "/": _page("Home"),
            BASE + "/deep&post": _page("Deep"),
        }
        _crawl(tmp_path, WebSourceConfig(url=BASE + "/", wait_time=0, use_sitemap=True), routes)
        assert (tmp_path / "deep&post.md").exists()

    def test_missing_sitemap_still_crawls(self, tmp_path):
        routes = {BASE + "/": _page("Home")}
        _crawl(tmp_path, WebSourceConfig(url=BASE + "/", wait_time=0, use_sitemap=True), routes)
        assert (tmp_path / "index.md").exists()

    def test_parse_sitemap(self):
        locs, is_index = _parse_sitemap("<urlset><url><loc> https://a/b </loc></url></urlset>")
        assert locs == ["https://a/b"]
        assert is_index is False


class TestPublishedFrontmatter:
    def test_dated_page_gets_published_line(self, tmp_path):
        routes = {BASE + "/": _page("Home", published="2026-01-18T11:28:56+00:00")}
        _crawl(tmp_path, WebSourceConfig(url=BASE + "/", wait_time=0), routes)
        text = (tmp_path / "index.md").read_text()
        assert 'published: "2026-01-18T11:28:56+00:00"\n' in text

    def test_undated_page_has_no_published_line(self, tmp_path):
        routes = {BASE + "/": _page("Home")}
        _crawl(tmp_path, WebSourceConfig(url=BASE + "/", wait_time=0), routes)
        assert "published:" not in (tmp_path / "index.md").read_text()


class TestDocumentDownloads:
    def _config(self, **kw):
        return WebSourceConfig(url=BASE + "/", wait_time=0, download_types=["pdf"], **kw)

    def test_extensionless_download_saved(self, tmp_path):
        routes = {
            BASE + "/": _page("Home", links=["/_downloads/abc123"]),
            BASE + "/_downloads/abc123": _pdf(),
        }
        result, _fetched, _ = _crawl(tmp_path, self._config(), routes)
        files = list((tmp_path / "files").iterdir())
        assert len(files) == 1
        assert files[0].name.startswith("Report_One-") and files[0].suffix == ".pdf"
        assert files[0].read_bytes() == PDF_BYTES
        assert result.files_added == 2  # page + document

    def test_pdf_link_followed_only_when_downloading(self):
        assert normalize_link("/docs/report.pdf", "/", BASE) is None
        kept = normalize_link("/docs/report.pdf", "/", BASE, keep_extensions=frozenset({"pdf"}))
        assert kept == "/docs/report.pdf"

    def test_downloads_disabled_by_default(self, tmp_path):
        routes = {
            BASE + "/": _page("Home", links=["/_downloads/abc123"]),
            BASE + "/_downloads/abc123": _pdf(),
        }
        _crawl(tmp_path, WebSourceConfig(url=BASE + "/", wait_time=0), routes)
        assert not (tmp_path / "files").exists()

    def test_unwanted_type_not_saved(self, tmp_path):
        routes = {
            BASE + "/": _page("Home", links=["/_downloads/zip1"]),
            BASE + "/_downloads/zip1": _response(content=b"PK", content_type="application/zip"),
        }
        _crawl(tmp_path, self._config(), routes)
        assert not (tmp_path / "files").exists()

    def test_oversized_file_skipped(self, tmp_path):
        routes = {
            BASE + "/": _page("Home", links=["/_downloads/abc123"]),
            BASE + "/_downloads/abc123": _pdf(),
        }
        _crawl(tmp_path, self._config(max_file_mb=0.000001), routes)
        assert not (tmp_path / "files").exists()

    def test_saved_file_not_refetched(self, tmp_path):
        routes = {
            BASE + "/": _page("Home", links=["/_downloads/abc123"]),
            BASE + "/_downloads/abc123": _pdf(),
        }
        _result, _fetched, cache = _crawl(tmp_path, self._config(), routes)
        _result, fetched, _ = _crawl(tmp_path, self._config(), routes, cache=cache)
        assert BASE + "/_downloads/abc123" not in fetched

    def test_deleted_file_refetched_unconditionally(self, tmp_path):
        etag_pdf = _pdf()
        etag_pdf.headers["etag"] = '"v1"'
        routes = {
            BASE + "/": _page("Home", links=["/_downloads/abc123"]),
            BASE + "/_downloads/abc123": etag_pdf,
        }
        _result, _fetched, cache = _crawl(tmp_path, self._config(), routes, cache=None)
        for f in (tmp_path / "files").iterdir():
            f.unlink()

        seen_headers = {}

        def get(url, headers=None):
            seen_headers[url] = headers
            found = routes.get(url, _response(404))
            return _response(text=found) if isinstance(found, str) else found

        client = MagicMock()
        client.get.side_effect = get
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        with patch("httpx.Client", return_value=client), patch("time.sleep"):
            WebSourceConnector(self._config())._crawl(tmp_path, cache, SyncResult(source_type="web"), set())
        assert seen_headers[BASE + "/_downloads/abc123"] == {}
        assert len(list((tmp_path / "files").iterdir())) == 1

    def test_extension_prefers_disposition_then_mime_then_url(self):
        assert _document_extension("/x", {"content-disposition": 'attachment; filename="a.DOCX"'}) == "docx"
        assert _document_extension("/x", {"content-type": "application/pdf"}) == "pdf"
        assert _document_extension("/files/a.xlsx", {"content-type": "application/octet-stream"}) == "xlsx"

    def test_download_filename_stable_and_safe(self):
        headers = {"content-disposition": 'attachment; filename="NHS: Report / 2025.pdf"'}
        first = _download_filename("/_downloads/abc", headers, "pdf")
        assert first == _download_filename("/_downloads/abc", headers, "pdf")
        assert first != _download_filename("/_downloads/def", headers, "pdf")
        assert "/" not in first and ":" not in first and first.endswith(".pdf")

    def test_config_from_dict_reads_new_fields(self):
        cfg = WebSourceConfig.from_dict({
            "url": BASE, "allow_paths": ["^/a"], "use_sitemap": True,
            "download_types": ["pdf"], "max_file_mb": 5,
        })
        assert cfg.allow_paths == ["^/a"] and cfg.use_sitemap is True
        assert cfg.download_types == ["pdf"] and cfg.max_file_mb == 5


class TestFrontmatterMetadata:
    def test_reads_url_and_published(self, tmp_path):
        from mullet.knowledge.engine.ingest import read_frontmatter_metadata

        md = tmp_path / "post.md"
        md.write_text(
            '---\ntitle: "Post"\nurl: https://news.example.com/post\n'
            'published: "2026-01-18T11:28:56+00:00"\nsource: web_crawler\n---\n\nBody\n'
        )
        meta = read_frontmatter_metadata(md)
        assert meta == {
            "source_url": "https://news.example.com/post",
            "doc_title": "Post",
            "published_at": "2026-01-18T11:28:56+00:00",
            "published_ts": 1768735736,
        }

    def test_unquoted_yaml_datetime(self, tmp_path):
        from mullet.knowledge.engine.ingest import read_frontmatter_metadata

        md = tmp_path / "post.md"
        md.write_text("---\npublished: 2026-01-18\n---\nBody\n")
        assert read_frontmatter_metadata(md)["published_at"] == "2026-01-18T00:00:00+00:00"

    @pytest.mark.parametrize("name, text", [
        ("plain.md", "# No frontmatter\n"),
        ("bad.md", "---\n: : [\n---\nBody\n"),
        ("list.md", "---\n- a\n- b\n---\nBody\n"),
        ("doc.txt", '---\nurl: https://x.com\n---\n'),
    ])
    def test_nothing_to_read(self, tmp_path, name, text):
        from mullet.knowledge.engine.ingest import read_frontmatter_metadata

        path = tmp_path / name
        path.write_text(text)
        assert read_frontmatter_metadata(path) == {}

    def test_ingest_document_attaches_metadata(self, tmp_path):
        from mullet.knowledge.engine.ingest import DocumentIngestor

        md = tmp_path / "post.md"
        md.write_text('---\npublished: "2026-01-18T00:00:00+00:00"\n---\nBody\n')
        ingestor = DocumentIngestor.__new__(DocumentIngestor)
        ingestor.parse_document = MagicMock(return_value=[])
        ingestor.chunk_elements = MagicMock(return_value=[{"id": "1", "content": "x", "metadata": {}}])
        ingestor.create_embeddings = MagicMock(return_value=[[0.0]])
        chunks, _ = ingestor.ingest_document(md)
        assert chunks[0]["metadata"]["published_ts"] == 1768694400
        assert chunks[0]["metadata"]["source_file"] == "post.md"


class TestRecencyBoost:
    DAY = 86400
    NOW = 1_800_000_000

    @pytest.fixture
    def searcher(self):
        from mullet.knowledge.engine.search.searcher import HybridSearcher

        return HybridSearcher.__new__(HybridSearcher)

    def _chunk(self, cid, score, age_days=None):
        meta = {} if age_days is None else {"published_ts": self.NOW - age_days * self.DAY}
        return {"id": cid, "_rerank_score": score, "metadata": meta}

    def test_recent_outranks_older_at_equal_relevance(self, searcher):
        chunks = [self._chunk("old", 5.0, age_days=720), self._chunk("new", 5.0, age_days=10)]
        out = searcher.boost_recent_chunks(chunks, 2.0, 180, now=self.NOW)
        assert [c["id"] for c in out] == ["new", "old"]

    def test_bonus_halves_each_half_life(self, searcher):
        out = searcher.boost_recent_chunks(
            [self._chunk("a", 0.0, age_days=0), self._chunk("b", 0.0, age_days=180)],
            2.0, 180, now=self.NOW,
        )
        by_id = {c["id"]: c for c in out}
        assert by_id["a"]["_recency_boost"] == pytest.approx(2.0)
        assert by_id["b"]["_recency_boost"] == pytest.approx(1.0)

    def test_undated_chunks_untouched(self, searcher):
        chunks = [self._chunk("evergreen", 4.0)]
        out = searcher.boost_recent_chunks(chunks, 2.0, 180, now=self.NOW)
        assert out[0]["_rerank_score"] == 4.0
        assert "_recency_boost" not in out[0]

    def test_strong_relevance_beats_small_recency_bonus(self, searcher):
        chunks = [self._chunk("relevant-old", 8.0, age_days=900), self._chunk("weak-new", 3.0, age_days=1)]
        out = searcher.boost_recent_chunks(chunks, 2.0, 180, now=self.NOW)
        assert out[0]["id"] == "relevant-old"

    def test_combined_score_fallback_and_no_mutation(self, searcher):
        chunk = {"id": "c", "_combined_score": 0.5, "metadata": {"published_ts": self.NOW}}
        out = searcher.boost_recent_chunks([chunk], 1.0, 30, now=self.NOW)
        assert out[0]["_combined_score"] == pytest.approx(1.5)
        assert chunk["_combined_score"] == 0.5

    def test_future_dates_capped_at_full_boost(self, searcher):
        out = searcher.boost_recent_chunks([self._chunk("f", 0.0, age_days=-5)], 2.0, 180, now=self.NOW)
        assert out[0]["_recency_boost"] == pytest.approx(2.0)

    def test_search_config_keys_known(self):
        from mullet.knowledge.engine.client import DEFAULT_CONFIG, _KNOWN_SEARCH_KEYS

        assert {"recency_boost", "recency_half_life_days"} <= _KNOWN_SEARCH_KEYS
        assert DEFAULT_CONFIG["search"]["recency_boost"] == 0.0

    def test_recency_ts_preferred_over_published_ts(self, searcher):
        chunk = {
            "id": "pdf", "_rerank_score": 0.0,
            "metadata": {"recency_ts": self.NOW, "published_ts": self.NOW - 3650 * self.DAY},
        }
        out = searcher.boost_recent_chunks([chunk], 2.0, 180, now=self.NOW)
        assert out[0]["_recency_boost"] == pytest.approx(2.0)


def _sync(tmp_path, config, routes):
    """Run a full sync() — cache, file index and crawl log live on disk between runs."""
    def get(url, headers=None):
        found = routes.get(url, _response(404))
        return _response(text=found) if isinstance(found, str) else found

    client = MagicMock()
    client.get.side_effect = get
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    with patch("httpx.Client", return_value=client), patch("time.sleep"):
        result = WebSourceConnector(config).sync(str(tmp_path))
    return result, tmp_path / "web" / "news.example.com"


def _last_crawl(web_dir):
    return json.loads((web_dir / ".crawl_log.jsonl").read_text().splitlines()[-1])


class TestChangeDetection:
    def _cfg(self, **kw):
        return WebSourceConfig(url=BASE + "/", wait_time=0, **kw)

    def test_nonce_only_difference_is_not_a_change(self, tmp_path):
        _sync(tmp_path, self._cfg(), {BASE + "/": _page("Home", nonce="aaa")})
        result, web = _sync(tmp_path, self._cfg(), {BASE + "/": _page("Home", nonce="bbb")})
        assert result.files_added == 0 and result.files_updated == 0
        assert "updated:" not in (web / "index.md").read_text()

    def test_download_counter_is_not_a_change(self, tmp_path):
        _sync(tmp_path, self._cfg(), {BASE + "/": _page("Home", extra="<p>205 downloads</p>")})
        result, _ = _sync(tmp_path, self._cfg(), {BASE + "/": _page("Home", extra="<p>206 downloads</p>")})
        assert result.files_updated == 0

    def test_text_change_rewrites_page_with_updated_date(self, tmp_path):
        _sync(tmp_path, self._cfg(), {BASE + "/": _page("Home")})
        result, web = _sync(tmp_path, self._cfg(), {
            BASE + "/": _page("Home", extra="<p>Breaking news: a new dossier is published today.</p>"),
        })
        assert result.files_updated == 1
        assert 'updated: "' in (web / "index.md").read_text()
        assert _last_crawl(web)["changed"] == [BASE + "/"]

    def test_page_cached_before_text_hashing_is_rewritten_once_not_changed(self, tmp_path):
        web = tmp_path / "web" / "news.example.com"
        web.mkdir(parents=True)
        old = CrawlCache(str(web))
        old.update_page("/", {}, "<html>older crawl</html>", links=[])
        old.save()
        (web / "index.md").write_text("written by an older crawler")

        _sync(tmp_path, self._cfg(), {BASE + "/": _page("Home")})
        text = (web / "index.md").read_text()
        assert "updated:" not in text and "first_seen:" not in text
        cache = CrawlCache(str(web))
        cache.load()
        entry = cache.get_entry("/")
        assert entry["text_hash"].startswith("sha256:") and entry["baseline"] is True
        assert _last_crawl(web)["changed"] == []


class TestFirstSeen:
    def test_first_crawl_is_baseline(self, tmp_path):
        cfg = WebSourceConfig(url=BASE + "/", wait_time=0)
        _, web = _sync(tmp_path, cfg, {BASE + "/": _page("Home", links=["/a"]), BASE + "/a": _page("A")})
        assert "first_seen:" not in (web / "a.md").read_text()
        log = _last_crawl(web)
        assert log["baseline"] is True and log["added"] == []

    def test_page_added_later_gets_first_seen(self, tmp_path):
        cfg = WebSourceConfig(url=BASE + "/", wait_time=0)
        _sync(tmp_path, cfg, {BASE + "/": _page("Home")})
        _, web = _sync(tmp_path, cfg, {
            BASE + "/": _page("Home", links=["/new"]),
            BASE + "/new": _page("New post"),
        })
        assert 'first_seen: "' in (web / "new.md").read_text()
        assert _last_crawl(web)["added"] == [BASE + "/new"]

    def test_download_added_later_gets_first_seen_in_file_index(self, tmp_path):
        cfg = WebSourceConfig(url=BASE + "/", wait_time=0, download_types=["pdf"])
        _sync(tmp_path, cfg, {
            BASE + "/": _page("Home", links=["/_downloads/old"]),
            BASE + "/_downloads/old": _pdf("Old.pdf"),
        })
        _, web = _sync(tmp_path, cfg, {
            BASE + "/": _page("Home", links=["/_downloads/old", "/_downloads/new"]),
            BASE + "/_downloads/old": _pdf("Old.pdf"),
            BASE + "/_downloads/new": _pdf("New.pdf"),
        })
        index = json.loads((web / "files" / ".file_index.json").read_text())
        by_url = {v["url"]: v for v in index.values()}
        assert by_url[BASE + "/_downloads/old"]["first_seen"] is None
        assert by_url[BASE + "/_downloads/new"]["first_seen"]
        assert BASE + "/_downloads/new" in _last_crawl(web)["added"]


def _pdf_file(path, created):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/CreationDate": created})
    with path.open("wb") as fh:
        writer.write(fh)
    return path


class TestDocumentDates:
    def test_parse_pdf_date_formats(self):
        from mullet.knowledge.engine.ingest import parse_pdf_date

        assert parse_pdf_date("D:20260907143000+01'00'").isoformat() == "2026-09-07T13:30:00+00:00"
        assert parse_pdf_date("D:20260907143000Z").isoformat() == "2026-09-07T14:30:00+00:00"
        assert parse_pdf_date("D:2026").isoformat() == "2026-01-01T00:00:00+00:00"
        assert parse_pdf_date("20250102").date().isoformat() == "2025-01-02"
        for bad in (None, "", "garbage", "D:19800101", "D:20261399", "D:29990101"):
            assert parse_pdf_date(bad) is None

    def test_read_pdf_created_from_real_pdf(self, tmp_path):
        from mullet.knowledge.engine.ingest import read_pdf_created

        pdf = _pdf_file(tmp_path / "doc.pdf", "D:20260907120000Z")
        assert read_pdf_created(pdf).isoformat() == "2026-09-07T12:00:00+00:00"

    def test_broken_pdf_has_no_date(self, tmp_path):
        from mullet.knowledge.engine.ingest import read_pdf_created

        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf")
        assert read_pdf_created(bad) is None

    def test_file_index_supplies_url_and_first_seen(self, tmp_path):
        from mullet.knowledge.engine.ingest import document_metadata

        (tmp_path / ".file_index.json").write_text(json.dumps({
            "a.pdf": {"url": "https://news.example.com/_downloads/1", "first_seen": "2026-10-06T03:00:00+00:00"},
        }))
        (tmp_path / "a.pdf").write_bytes(b"not a real pdf")
        meta = document_metadata(tmp_path / "a.pdf")
        assert meta["source_url"] == "https://news.example.com/_downloads/1"
        assert meta["recency_source"] == "first_seen"
        assert meta["recency_ts"] == meta["first_seen_ts"] == 1791255600

    def test_published_date_wins_over_other_dates(self, tmp_path):
        from mullet.knowledge.engine.ingest import document_metadata

        md = tmp_path / "p.md"
        md.write_text(
            '---\npublished: "2025-01-01T00:00:00+00:00"\n'
            'first_seen: "2026-10-01T00:00:00+00:00"\n---\nBody\n'
        )
        meta = document_metadata(md)
        assert meta["recency_source"] == "published"
        assert meta["recency_ts"] == meta["published_ts"]

    def test_newest_of_pdf_created_and_first_seen(self, tmp_path):
        from mullet.knowledge.engine.ingest import document_metadata

        _pdf_file(tmp_path / "old.pdf", "D:20060101000000Z")
        (tmp_path / ".file_index.json").write_text(json.dumps({
            "old.pdf": {"url": "https://news.example.com/_downloads/2", "first_seen": "2026-10-06T03:00:00+00:00"},
        }))
        meta = document_metadata(tmp_path / "old.pdf")
        assert meta["pdf_created_ts"] < meta["first_seen_ts"]
        assert meta["recency_source"] == "first_seen"

    def test_pdf_created_used_when_nothing_newer(self, tmp_path):
        from mullet.knowledge.engine.ingest import document_metadata

        meta = document_metadata(_pdf_file(tmp_path / "d.pdf", "D:20260819000000Z"))
        assert meta["recency_source"] == "pdf_created"

    def test_undated_document_has_no_recency(self, tmp_path):
        from mullet.knowledge.engine.ingest import document_metadata

        md = tmp_path / "plain.md"
        md.write_text("# Evergreen page\n")
        assert "recency_ts" not in document_metadata(md)


class TestTrawlRecent:
    def _searcher(self, metadatas_by_key):
        collection = MagicMock()
        collection.get.side_effect = lambda where, include: {
            "metadatas": metadatas_by_key.get(next(iter(where)), []),
        }
        searcher = MagicMock()
        searcher.collection = collection
        return searcher

    def _run(self, searcher, **args):
        from mullet.server import mcp_knowledge as mk

        with patch.object(mk, "_load_pool", return_value=(MagicMock(), searcher)):
            return mk.handle_recent({"pool_name": "news", **args})

    def test_lists_documents_newest_first_once_each(self):
        pdf = {"source_path": "/p/a.pdf", "source_file": "files/a.pdf",
               "recency_ts": 1_788_000_000, "recency_source": "pdf_created"}
        post = {"source_path": "/p/b.md", "source_file": "b.md", "doc_title": "Newest post",
                "source_url": "https://news.example.com/b", "recency_ts": 1_790_000_000,
                "recency_source": "published"}
        out = self._run(self._searcher({"recency_ts": [pdf, dict(pdf), post]}))
        assert out.index("Newest post") < out.index("a.pdf")
        assert out.count("a.pdf") == 1
        assert "(2 dated)" in out and "PDF created" in out
        assert "https://news.example.com/b" in out

    def test_since_is_passed_to_the_index(self):
        searcher = self._searcher({})
        out = self._run(searcher, since="2026-09-01")
        where = searcher.collection.get.call_args_list[0].kwargs["where"]
        assert where == {"recency_ts": {"$gte": 1788220800}}
        assert "No dated documents" in out

    def test_falls_back_to_published_ts_for_older_indexes(self):
        post = {"source_path": "/p/b.md", "source_file": "b.md", "published_ts": 1_790_000_000}
        out = self._run(self._searcher({"published_ts": [post]}))
        assert "b.md" in out and "published" in out

    def test_bad_since_is_rejected(self):
        assert "since must be an ISO date" in self._run(self._searcher({}), since="last week")


class TestPrioritizeDataSetting:
    def test_load_pool_passes_setting_to_searcher(self):
        from mullet.server import mcp_knowledge as mk

        client = MagicMock()
        client.search_config = {"prioritize_data": False}
        client.embedding_config = {}
        client.index_path.exists.return_value = True
        previous = mk._pool_cache.current_name
        try:
            with patch("mullet.knowledge.engine.client.load_client", return_value=client), \
                    patch("mullet.knowledge.engine.search.searcher.HybridSearcher") as searcher_cls, \
                    patch.object(mk._pool_cache, "get", return_value=None), \
                    patch.object(mk._pool_cache, "put"):
                mk._load_pool("news")
        finally:
            mk._pool_cache.current_name = previous
        assert searcher_cls.call_args.kwargs["prioritize_data"] is False

    def test_setting_is_known_everywhere_it_is_read(self):
        from mullet.core.knowledge.manager import KnowledgeManager
        from mullet.knowledge.engine.client import DEFAULT_CONFIG, _KNOWN_SEARCH_KEYS

        assert "prioritize_data" in _KNOWN_SEARCH_KEYS
        assert "prioritize_data" in KnowledgeManager._SEARCHER_PARAMS
        assert DEFAULT_CONFIG["search"]["prioritize_data"] is True


class TestReindexLock:
    @pytest.fixture
    def reindex(self, tmp_path, monkeypatch):
        from tests.test_reindex_process_pool import _load_reindex

        module = _load_reindex()
        monkeypatch.setattr(module, "TRAWL_POOLS_DIR", tmp_path)
        (tmp_path / "news").mkdir()
        return module

    def test_second_holder_is_refused_until_released(self, reindex):
        with reindex._pool_lock("news") as first:
            with reindex._pool_lock("news") as second:
                assert first is True and second is False
        with reindex._pool_lock("news") as again:
            assert again is True

    def test_locked_pool_is_skipped(self, reindex):
        with reindex._pool_lock("news"):
            with patch.object(reindex, "_reindex_one_locked") as inner, \
                    patch.object(reindex, "sync_client_sources") as sync:
                reindex._reindex_one("news", MagicMock(), quiet=True, sync_sources=True)
        inner.assert_not_called()
        sync.assert_not_called()

    def test_sources_sync_inside_the_lock(self, reindex):
        with patch.object(reindex, "_reindex_one_locked") as inner, \
                patch.object(reindex, "sync_client_sources") as sync:
            reindex._reindex_one("news", MagicMock(), quiet=True, sync_sources=True)
        sync.assert_called_once_with("news")
        inner.assert_called_once()

    def test_unknown_pool_is_not_locked(self, reindex):
        with reindex._pool_lock("no-such-pool") as locked:
            assert locked is True
