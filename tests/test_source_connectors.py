"""Tests for mullet.knowledge.sources — GitHub and web source connectors."""

from mullet.knowledge.sources.base import SyncResult
from mullet.knowledge.sources.github_models import GitHubSourceConfig, RepoInfo
from mullet.knowledge.sources.web_converter import extract_title, clean_markdown
from mullet.knowledge.sources.web_cache import CrawlCache


class TestSyncResult:
    def test_creation(self):
        r = SyncResult(source_type="github", files_added=5, files_updated=2, files_removed=1)
        assert r.source_type == "github"
        assert r.files_added == 5

    def test_total_changes(self):
        r = SyncResult(source_type="github", files_added=3, files_updated=2, files_removed=1)
        assert r.total_changes == 6

    def test_success_no_errors(self):
        r = SyncResult(source_type="web")
        assert r.success is True

    def test_failure_with_errors(self):
        r = SyncResult(source_type="web", errors=["timeout"])
        assert r.success is False


class TestGitHubModels:
    def test_source_config_creation(self):
        config = GitHubSourceConfig(owner="anthropics", repo="claude-code")
        assert config.owner == "anthropics"
        assert config.repo == "claude-code"

    def test_source_config_shorthand(self):
        config = GitHubSourceConfig(repo="anthropics/claude-code")
        assert config.owner == "anthropics"
        assert config.repo == "claude-code"

    def test_source_config_from_dict(self):
        config = GitHubSourceConfig.from_dict({"owner": "foo", "repo": "bar"})
        assert config.owner == "foo"
        assert config.repo == "bar"

    def test_repo_info_from_api(self):
        data = {
            "name": "claude-code",
            "owner": {"login": "anthropics"},
            "html_url": "https://github.com/anthropics/claude-code",
            "clone_url": "https://github.com/anthropics/claude-code.git",
            "private": False,
            "description": "CLI tool",
            "default_branch": "main",
            "size": 1024,
            "fork": False,
        }
        info = RepoInfo.from_api_response(data)
        assert info.name == "claude-code"
        assert info.owner == "anthropics"
        assert info.default_branch == "main"
        assert info.size_kb == 1024
        assert info.is_fork is False


class TestWebConverter:
    def test_extract_title_from_h1(self):
        html = "<html><body><h1>My Title</h1><p>Content</p></body></html>"
        assert extract_title(html) == "My Title"

    def test_clean_markdown_removes_excess_newlines(self):
        md = "# Title\n\n\n\n\nParagraph\n\n\n\n\nEnd"
        cleaned = clean_markdown(md)
        assert "\n\n\n\n" not in cleaned


class TestCrawlCache:
    def test_update_and_check(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/test", {"etag": "abc"}, "content here", links=["/page2"])
        assert cache.page_count == 1

    def test_page_unchanged(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/test", {}, "same content")
        assert cache.is_page_unchanged("/test", "same content") is True
        assert cache.is_page_unchanged("/test", "different") is False

    def test_page_links(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/test", {}, "content", links=["/a", "/b"])
        links = cache.get_page_links("/test")
        assert links == ["/a", "/b"]

    def test_conditional_headers(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/test", {"etag": '"abc"', "last-modified": "Mon, 01 Jan 2024"}, "c")
        headers = cache.get_page_headers("/test")
        assert headers["If-None-Match"] == '"abc"'

    def test_save_and_load(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/test", {}, "content")
        cache.save()

        cache2 = CrawlCache(str(tmp_path))
        cache2.load()
        assert cache2.page_count == 1
