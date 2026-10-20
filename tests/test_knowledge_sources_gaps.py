"""Gap-filling tests for mullet/core/knowledge/sources.py — target ≥93%."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.knowledge.sources import (
    GitHubSource,
    GitHubConnector,
    WebSource,
    WebCrawler,
    _is_cloudflare_challenge,
)


# ── httpx import fallback (lines 25-26) ──────────────────────────────────────

def test_httpx_import_fallback():
    """When _httpx is None, sync returns an error early."""
    with patch("mullet.core.knowledge.sources._httpx", None):
        source = WebSource(url="https://example.com", max_pages=1, wait_time=0)
        crawler = WebCrawler(source)
        result = crawler.sync("/tmp/fake_web_dir")
        assert result.source_type == "web"
        assert any("httpx not installed" in e for e in result.errors)


def test_httpx_import_error_on_module_load():
    """Lines 25-26: import httpx fallback when module is loaded without httpx."""
    import builtins
    import importlib
    import sys

    mod_name = "mullet.core.knowledge.sources"
    real_import = builtins.__import__
    real_httpx = sys.modules.get("httpx")

    def fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    # Remove cached httpx so reload attempts import
    sys.modules.pop("httpx", None)
    builtins.__import__ = fake_import

    try:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        assert mod._httpx is None
    finally:
        builtins.__import__ = real_import
        if real_httpx is not None:
            sys.modules["httpx"] = real_httpx
        # Restore module with real httpx available
        mod = sys.modules.get(mod_name)
        if mod is not None:
            importlib.reload(mod)


# ── GitHubConnector gaps ─────────────────────────────────────────────────────

@patch("mullet.core.knowledge.sources.subprocess")
def test_github_sync_no_repos(mock_subprocess, tmp_path):
    """Line 98: when discover_repos returns empty, sync returns early."""
    mock_subprocess.run.side_effect = FileNotFoundError()
    source = GitHubSource(owner="testuser")
    connector = GitHubConnector(source)
    result = connector.sync(tmp_path)
    assert result.total_changes == 0
    assert not result.success


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_sync_pull_existing_repo(mock_subprocess, tmp_path):
    """Lines 106-108: existing repo with .git dir triggers _pull_repo."""
    target = tmp_path / "github" / "testuser" / "existing"
    target.mkdir(parents=True, exist_ok=True)
    (target / ".git").mkdir()

    call_count = {"rev": 0}

    def run_side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, list):
            if "rev-parse" in cmd:
                call_count["rev"] += 1
                sha = "oldsha" if call_count["rev"] == 1 else "newsha"
                return MagicMock(returncode=0, stdout=f"{sha}\n", stderr="")
            if "pull" in cmd:
                return MagicMock(returncode=0, stdout="", stderr="")
            if "diff" in cmd:
                return MagicMock(returncode=0, stdout="file1.txt\nfile2.txt\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_subprocess.run.side_effect = run_side_effect
    source = GitHubSource(owner="testuser", repos=["existing"])
    connector = GitHubConnector(source)
    result = connector.sync(tmp_path)
    assert result.files_updated == 2


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_sync_pull_no_changes(mock_subprocess, tmp_path):
    """Lines 178-202: old_head == new_head yields 0 changes."""
    target = tmp_path / "github" / "testuser" / "same"
    target.mkdir(parents=True, exist_ok=True)
    (target / ".git").mkdir()

    def run_side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, list) and "rev-parse" in cmd:
            return MagicMock(returncode=0, stdout="same_sha\n", stderr="")
        if isinstance(cmd, list) and "pull" in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_subprocess.run.side_effect = run_side_effect
    source = GitHubSource(owner="testuser", repos=["same"])
    connector = GitHubConnector(source)
    result = connector.sync(tmp_path)
    assert result.files_updated == 0


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_sync_clone_raises(mock_subprocess, tmp_path):
    """Lines 112-115: CalledProcessError during clone is caught."""
    mock_subprocess.CalledProcessError = subprocess.CalledProcessError
    mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, ["git"], stderr="fatal: repo not found")

    source = GitHubSource(owner="testuser", repos=["badrepo"])
    connector = GitHubConnector(source)
    result = connector.sync(tmp_path)
    assert not result.success
    assert any("badrepo" in e for e in result.errors)


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_discover_gh_returncode_nonzero(mock_subprocess):
    """Lines 142-143: gh returns non-zero exit code."""
    mock_subprocess.run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="some gh error",
    )
    source = GitHubSource(owner="testuser")
    connector = GitHubConnector(source)
    result = MagicMock()
    result.errors = []
    repos = connector._discover_repos(result)
    assert repos == []
    assert any("gh repo list failed" in e for e in result.errors)


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_discover_timeout(mock_subprocess):
    """Lines 158-160: subprocess.TimeoutExpired."""
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    mock_subprocess.run.side_effect = subprocess.TimeoutExpired("cmd", 30)
    source = GitHubSource(owner="testuser")
    connector = GitHubConnector(source)
    result = MagicMock()
    result.errors = []
    repos = connector._discover_repos(result)
    assert repos == []
    assert any("timed out" in e for e in result.errors)


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_discover_json_decode_error(mock_subprocess):
    """Lines 161-163: JSONDecodeError on gh output."""
    mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
    mock_subprocess.CalledProcessError = subprocess.CalledProcessError
    mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")
    source = GitHubSource(owner="testuser")
    connector = GitHubConnector(source)
    result = MagicMock()
    result.errors = []
    repos = connector._discover_repos(result)
    assert repos == []
    assert any("Failed to parse" in e for e in result.errors)


@patch("mullet.core.knowledge.sources.subprocess")
def test_github_clone_with_branch(mock_subprocess, tmp_path):
    """Line 170: branch flag appended to clone command."""
    calls = []

    def run_capture(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        calls.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_subprocess.run.side_effect = run_capture
    source = GitHubSource(owner="testuser", repos=["myrepo"], branch="main")
    connector = GitHubConnector(source)
    connector.sync(tmp_path)
    clone_cmd = [c for c in calls if "clone" in c]
    assert clone_cmd
    assert "--branch" in clone_cmd[0]
    assert "main" in clone_cmd[0]


# ── WebCrawler gaps ──────────────────────────────────────────────────────────

@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_url_already_visited(mock_httpx_module, tmp_path):
    """Line 344: duplicate URLs in queue are skipped."""
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client

    pages = {
        "https://example.com/docs": (
            "<html><body><p>Start page with enough content to pass threshold yes.</p>"
            '<a href="/docs/page1">Page1</a>'
            '<a href="/docs/page2">Page2</a>'
            "</body></html>"
        ),
        "https://example.com/docs/page1": (
            "<html><body><p>Page one content that passes the threshold here.</p>"
            '<a href="/docs/page3">Page3</a>'
            "</body></html>"
        ),
        "https://example.com/docs/page2": (
            "<html><body><p>Page two content that passes the threshold here.</p>"
            '<a href="/docs/page3">Page3</a>'
            "</body></html>"
        ),
        "https://example.com/docs/page3": (
            "<html><body><p>Page three content that passes the threshold here.</p></body></html>"
        ),
    }

    def get_side_effect(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = pages.get(url, "")
        return mock_resp

    mock_client.get.side_effect = get_side_effect

    source = WebSource(url="https://example.com/docs", max_pages=10, wait_time=0)
    crawler = WebCrawler(source)
    result = crawler.sync(tmp_path)
    # 4 unique pages added; page3 queued twice but visited only once
    assert result.files_added == 4


@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_status_403(mock_httpx_module, tmp_path):
    """Lines 350-352: 403 increments cloudflare_blocked."""
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.headers = {}
    mock_resp.text = ""
    mock_client.get.return_value = mock_resp

    source = WebSource(url="https://example.com", max_pages=1, wait_time=0)
    crawler = WebCrawler(source)
    result = crawler.sync(tmp_path)
    assert result.files_added == 0
    assert crawler._cloudflare_blocked == 1
    assert any("bot protection" in e for e in result.errors)


@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_skip_non_html(mock_httpx_module, tmp_path):
    """Lines 355-358: skip responses without text/html content-type."""
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.text = '{"foo":"bar"}'
    mock_client.get.return_value = mock_resp

    source = WebSource(url="https://example.com/api", max_pages=1, wait_time=0)
    crawler = WebCrawler(source)
    result = crawler.sync(tmp_path)
    assert result.files_added == 0


@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_short_markdown_skipped(mock_httpx_module, tmp_path):
    """Line 377: markdown shorter than 50 chars is skipped."""
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.text = "<html><body><p>hi</p></body></html>"
    mock_client.get.return_value = mock_resp

    source = WebSource(url="https://example.com", max_pages=1, wait_time=0)
    crawler = WebCrawler(source)
    result = crawler.sync(tmp_path)
    assert result.files_added == 0


@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_cache_hit_files_updated(mock_httpx_module, tmp_path):
    """Line 386: URL already in cache counts as updated when content changes."""
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client

    page_html = (
        "<html><body><p>Original content that is definitely longer than fifty characters.</p></body></html>"
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.text = page_html
    mock_client.get.return_value = mock_resp

    source = WebSource(url="https://example.com", max_pages=1, wait_time=0)
    crawler = WebCrawler(source)
    result1 = crawler.sync(tmp_path)
    assert result1.files_added == 1

    # Change content so hash differs
    mock_resp.text = (
        "<html><body><p>Modified content that is definitely longer than fifty characters now.</p></body></html>"
    )
    result2 = crawler.sync(tmp_path)
    assert result2.files_updated == 1


@patch("mullet.core.knowledge.sources.time")
@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_wait_time(mock_httpx_module, mock_time, tmp_path):
    """Line 397: time.sleep called when wait_time > 0."""
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.text = (
        "<html><body><p>Content long enough to pass validation threshold yes indeed.</p></body></html>"
    )
    mock_client.get.return_value = mock_resp

    source = WebSource(url="https://example.com", max_pages=1, wait_time=1.5)
    crawler = WebCrawler(source)
    crawler.sync(tmp_path)
    mock_time.sleep.assert_called_with(1.5)


@patch("mullet.core.knowledge.sources._httpx")
def test_webcrawler_request_exception(mock_httpx_module, tmp_path):
    """Lines 356-358: Exception during client.get is caught and recorded."""
    mock_client = MagicMock()
    mock_httpx_module.Client.return_value = mock_client
    mock_client.get.side_effect = Exception("connection refused")

    source = WebSource(url="https://example.com", max_pages=1, wait_time=0)
    crawler = WebCrawler(source)
    result = crawler.sync(tmp_path)
    assert not result.success
    assert any("connection refused" in e for e in result.errors)


def test_normalize_link_relative_urljoin():
    """Line 490: relative href resolved via urljoin."""
    source = WebSource(url="https://example.com/docs")
    crawler = WebCrawler(source)
    result = crawler._normalize_link(
        "page.html", "https://example.com/docs/", "https://example.com", "/docs"
    )
    assert result == "https://example.com/docs/page.html"


def test_normalize_link_trailing_slash_stripped():
    """Line 496: trailing slash stripped from normalized link."""
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    result = crawler._normalize_link(
        "/about/", "https://example.com", "https://example.com", "/"
    )
    assert result == "https://example.com/about"


def test_load_cache_json_decode_error(tmp_path, caplog):
    """Lines 532-533: JSON decode error in cache logs warning."""
    cache_path = tmp_path / ".crawl_cache.json"
    cache_path.write_text("not json")
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    with caplog.at_level("WARNING", logger="mullet.core.knowledge.sources"):
        cache = crawler._load_cache(tmp_path)
    assert cache == {}
    assert any("Failed to load crawl cache" in r.message for r in caplog.records)


def test_load_cache_oserror(tmp_path, caplog):
    """Lines 532-533: OSError in cache load is handled."""
    # Create a directory with the cache filename so read_text raises IsADirectoryError
    cache_path = tmp_path / ".crawl_cache.json"
    cache_path.mkdir()
    source = WebSource(url="https://example.com")
    crawler = WebCrawler(source)
    with caplog.at_level("WARNING", logger="mullet.core.knowledge.sources"):
        cache = crawler._load_cache(tmp_path)
    assert cache == {}
    assert any("Failed to load crawl cache" in r.message for r in caplog.records)
