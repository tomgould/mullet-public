"""Tests for source connectors — GitHub and web crawler."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from mullet.core.knowledge.sources import (
    SyncResult,
    GitHubSource,
    GitHubConnector,
    WebSource,
    WebCrawler,
    _is_cloudflare_challenge,
)


# ── SyncResult ───────────────────────────────────────────────────────────────

def test_sync_result_defaults():
    r = SyncResult(source_type="test")
    assert r.total_changes == 0
    assert r.success is True


def test_sync_result_with_errors():
    r = SyncResult(source_type="test", errors=["something broke"])
    assert r.success is False


def test_sync_result_total_changes():
    r = SyncResult(source_type="test", files_added=3, files_updated=2, files_removed=1)
    assert r.total_changes == 6


# ── GitHubSource ─────────────────────────────────────────────────────────────

def test_github_source_defaults():
    s = GitHubSource(owner="octocat")
    assert s.repos is None
    assert s.include_forks is False
    assert s.branch is None


# ── GitHubConnector ──────────────────────────────────────────────────────────

@patch("mullet.core.knowledge.sources.subprocess")
def test_github_discover_repos(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps([
            {"name": "repo1", "isFork": False, "isArchived": False},
            {"name": "repo2", "isFork": True, "isArchived": False},
            {"name": "repo3", "isFork": False, "isArchived": True},
        ]),
    )

    source = GitHubSource(owner="testuser")
    connector = GitHubConnector(source)
    result = SyncResult(source_type="github")
    repos = connector._discover_repos(result)

    assert "repo1" in repos
    assert "repo2" not in repos
    assert "repo3" not in repos


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_discover_includes_forks(mock_subprocess):
    mock_subprocess.run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps([
            {"name": "repo1", "isFork": False, "isArchived": False},
            {"name": "forked", "isFork": True, "isArchived": False},
        ]),
    )

    source = GitHubSource(owner="testuser", include_forks=True)
    connector = GitHubConnector(source)
    result = SyncResult(source_type="github")
    repos = connector._discover_repos(result)
    assert "forked" in repos


def test_github_explicit_repos():
    source = GitHubSource(owner="testuser", repos=["myrepo"])
    connector = GitHubConnector(source)
    result = SyncResult(source_type="github")
    repos = connector._discover_repos(result)
    assert repos == ["myrepo"]


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_gh_not_found(mock_subprocess):
    mock_subprocess.run.side_effect = FileNotFoundError()

    source = GitHubSource(owner="testuser")
    connector = GitHubConnector(source)
    result = SyncResult(source_type="github")
    repos = connector._discover_repos(result)

    assert repos == []
    assert any("gh CLI not found" in e for e in result.errors)


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_clone(mock_subprocess, tmp_path):
    mock_subprocess.CalledProcessError = Exception

    def run_side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, list) and "clone" in cmd:
            repo_dir = Path(cmd[-1])
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir()
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_subprocess.run.side_effect = run_side_effect

    source = GitHubSource(owner="testuser", repos=["newrepo"])
    connector = GitHubConnector(source)
    result = connector.sync(tmp_path)
    assert result.files_added == 1
    assert result.success


def test_github_status():
    source = GitHubSource(owner="testuser")
    connector = GitHubConnector(source)
    status = connector.status()
    assert status["source_type"] == "github"
    assert status["owner"] == "testuser"


# ── Cloudflare detection ─────────────────────────────────────────────────────

def test_cloudflare_challenge_detected():
    html = "<html><body>Just a moment...</body></html>"
    assert _is_cloudflare_challenge(html) is True


def test_cloudflare_normal_page():
    html = "<html><body><h1>Normal page</h1></body></html>"
    assert _is_cloudflare_challenge(html) is False


def test_cloudflare_browser_verification():
    html = '<div class="challenge-platform">Verifying...</div>'
    assert _is_cloudflare_challenge(html) is True


# ── WebSource ────────────────────────────────────────────────────────────────

def test_web_source_defaults():
    s = WebSource(url="https://example.com/docs")
    assert s.stay_in_path is True
    assert s.max_pages == 100
    assert s.stealth is True


# ── WebCrawler ───────────────────────────────────────────────────────────────

def test_webcrawler_html_to_markdown():
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)

    html = "<html><body><h1>Title</h1><p>Hello world</p><script>alert(1)</script></body></html>"
    md = crawler._html_to_markdown(html)

    assert "# Title" in md
    assert "Hello world" in md
    assert "alert" not in md


def test_webcrawler_markdown_headings():
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    html = "<h1>H1</h1><h2>H2</h2><h3>H3</h3>"
    md = crawler._html_to_markdown(html)
    assert "# H1" in md
    assert "## H2" in md
    assert "### H3" in md


def test_webcrawler_markdown_code():
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    html = "<p>Use <code>pip install</code> to install.</p>"
    md = crawler._html_to_markdown(html)
    assert "`pip install`" in md


def test_webcrawler_markdown_bold_italic():
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    html = "<p><strong>bold</strong> and <em>italic</em></p>"
    md = crawler._html_to_markdown(html)
    assert "**bold**" in md
    assert "*italic*" in md


def test_webcrawler_markdown_list():
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    html = "<ul><li>One</li><li>Two</li></ul>"
    md = crawler._html_to_markdown(html)
    assert "- One" in md
    assert "- Two" in md


def test_webcrawler_html_entities():
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    html = "<p>A &amp; B &lt; C &gt; D</p>"
    md = crawler._html_to_markdown(html)
    assert "A & B < C > D" in md


def test_webcrawler_strips_nav_aside():
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    html = "<nav>Menu</nav><aside>Sidebar</aside><main>Content here</main>"
    md = crawler._html_to_markdown(html)
    assert "Menu" not in md
    assert "Sidebar" not in md
    assert "Content" in md


def test_webcrawler_extract_links():
    source = WebSource(url="https://example.com/docs")
    crawler = WebCrawler(source)

    html = '''
    <a href="/docs/getting-started">Start</a>
    <a href="/docs/api">API</a>
    <a href="https://other.com/page">External</a>
    <a href="/login">Login</a>
    <a href="/docs/image.png">Image</a>
    <a href="mailto:test@example.com">Email</a>
    '''

    links = crawler._extract_links(html, "https://example.com/docs", "https://example.com", "/docs")

    assert "https://example.com/docs/getting-started" in links
    assert "https://example.com/docs/api" in links
    assert not any("other.com" in l for l in links)
    assert not any("login" in l for l in links)
    assert not any("png" in l for l in links)
    assert not any("mailto" in l for l in links)


def test_webcrawler_extract_data_href():
    """Extract links from data-href attributes (SPA frameworks)."""
    source = WebSource(url="https://example.com/docs")
    crawler = WebCrawler(source)

    html = '<div data-href="/docs/page1">Link</div><span data-url="/docs/page2">Link</span>'
    links = crawler._extract_links(html, "https://example.com/docs", "https://example.com", "/docs")

    assert "https://example.com/docs/page1" in links
    assert "https://example.com/docs/page2" in links


def test_webcrawler_stay_in_path():
    source = WebSource(url="https://example.com/docs", stay_in_path=True)
    crawler = WebCrawler(source)

    html = '<a href="/blog/post">Blog</a><a href="/docs/page">Doc</a>'
    links = crawler._extract_links(html, "https://example.com/docs", "https://example.com", "/docs")

    assert any("/docs/page" in l for l in links)
    assert not any("/blog" in l for l in links)


def test_webcrawler_url_to_filename():
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)

    assert crawler._url_to_filename("https://example.com/", "example.com") == "index.md"
    assert crawler._url_to_filename("https://example.com/docs/api", "example.com") == "docs__api.md"


def test_webcrawler_cache(tmp_path):
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)

    cache = {"https://example.com": "abc123"}
    crawler._save_cache(tmp_path, cache)
    loaded = crawler._load_cache(tmp_path)
    assert loaded == cache


def test_webcrawler_cache_missing(tmp_path):
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    assert crawler._load_cache(tmp_path) == {}


def test_webcrawler_status():
    source = WebSource(url="https://example.com/docs")
    crawler = WebCrawler(source)
    status = crawler.status()
    assert status["source_type"] == "web"
    assert status["url"] == "https://example.com/docs"


@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_sync_basic(mock_httpx_module, tmp_path):
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
    mock_resp.text = """
    <html><body>
    <h1>Test Page</h1>
    <p>This is a test page with enough content to pass validation threshold.</p>
    <a href="/docs/page2">Page 2</a>
    </body></html>
    """
    mock_client.get.return_value = mock_resp

    source = WebSource(url="https://example.com/docs", max_pages=2, wait_time=0)
    crawler = WebCrawler(source)
    result = crawler.sync(tmp_path)

    assert result.source_type == "web"
    assert result.files_added >= 1
    assert mock_client.get.called


@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_incremental(mock_httpx_module, tmp_path):
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client

    page_html = "<html><body><p>Stable page content that should pass the minimum length threshold.</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.text = page_html
    mock_client.get.return_value = mock_resp

    source = WebSource(url="https://example.com/docs", max_pages=1, wait_time=0)

    result1 = WebCrawler(source).sync(tmp_path)
    assert result1.files_added == 1

    result2 = WebCrawler(source).sync(tmp_path)
    assert result2.files_added == 0


@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_cloudflare_blocked(mock_httpx_module, tmp_path):
    """Cloudflare challenge pages are detected and skipped."""
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.text = "<html><body>Just a moment...</body></html>"
    mock_client.get.return_value = mock_resp

    source = WebSource(url="https://protected.com", max_pages=1, wait_time=0)
    crawler = WebCrawler(source)
    result = crawler.sync(tmp_path)

    assert result.files_added == 0
    assert crawler._cloudflare_blocked > 0
    assert any("bot protection" in e for e in result.errors)


@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_stealth_headers(mock_httpx_module, tmp_path):
    """Stealth mode uses browser-like headers."""
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.text = "<html><body><p>Content long enough to pass validation check threshold.</p></body></html>"
    mock_client.get.return_value = mock_resp

    source = WebSource(url="https://example.com", max_pages=1, wait_time=0, stealth=True)
    WebCrawler(source).sync(tmp_path)

    # Check Client was created with stealth headers
    assert mock_httpx_module.Client.called
    call_kwargs = mock_httpx_module.Client.call_args
    headers = call_kwargs.kwargs.get("headers", {})
    assert "Chrome" in headers.get("User-Agent", "")


def test_webcrawler_normalize_link():
    source = WebSource(url="https://example.com/docs")
    crawler = WebCrawler(source)

    assert crawler._normalize_link("/docs/page", "https://example.com/docs", "https://example.com", "/docs") == "https://example.com/docs/page"
    assert crawler._normalize_link("https://other.com/x", "https://example.com/docs", "https://example.com", "/docs") is None
    assert crawler._normalize_link("mailto:x@y.com", "https://example.com/docs", "https://example.com", "/docs") is None
    assert crawler._normalize_link("/blog/post", "https://example.com/docs", "https://example.com", "/docs") is None  # out of path
    assert crawler._normalize_link("/docs/file.pdf", "https://example.com/docs", "https://example.com", "/docs") is None  # skip ext


def test_webcrawler_skip_non_content_paths():
    source = WebSource(url="https://example.com", stay_in_path=False)
    crawler = WebCrawler(source)
    base = "https://example.com"

    assert crawler._normalize_link("/admin/users", base, base, "/") is None
    assert crawler._normalize_link("/login", base, base, "/") is None
    assert crawler._normalize_link("/checkout/step1", base, base, "/") is None
    assert crawler._normalize_link("/cdn-cgi/challenge", base, base, "/") is None
    assert crawler._normalize_link("/docs/guide", base, base, "/") == "https://example.com/docs/guide"
