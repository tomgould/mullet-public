"""Source connectors — GitHub repos and web crawling for the knowledge engine.

Provides two connectors that fetch external content into a local directory
structure for ingestion by the pipeline:

- GitHubConnector: clones/pulls repos via `git` CLI
- WebCrawler: crawls websites via httpx, converts HTML to Markdown

Both implement BaseConnector with sync() -> SyncResult interface.
"""

import hashlib
import json
import logging
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    import httpx as _httpx
except ImportError:
    _httpx = None

logger = logging.getLogger(__name__)


# ── Base ─────────────────────────────────────────────────────────────────────

@dataclass
class SyncResult:
    """Result of a source sync operation."""
    source_type: str
    files_added: int = 0
    files_updated: int = 0
    files_removed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        """Return the total number of file changes."""
        return self.files_added + self.files_updated + self.files_removed

    @property
    def success(self) -> bool:
        """Return whether the sync completed without errors."""
        return len(self.errors) == 0


class BaseConnector(ABC):
    """Abstract base for source connectors."""

    @abstractmethod
    def sync(self, target_dir: str | Path) -> SyncResult:
        """Sync source content into target_dir. Returns SyncResult."""
        ...

    @abstractmethod
    def status(self) -> dict:
        """Return current status info."""
        ...


# ── GitHub connector ─────────────────────────────────────────────────────────

@dataclass
class GitHubSource:
    """Configuration for a GitHub source."""
    owner: str
    repos: list[str] | None = None  # None = all repos
    include_forks: bool = False
    include_archived: bool = False
    branch: str | None = None  # None = default branch


class GitHubConnector(BaseConnector):
    """Clone and sync GitHub repositories via git CLI.

    Uses `gh` CLI for repo discovery and `git` for clone/pull.
    Supports incremental sync — only fetches changes since last pull.
    """

    def __init__(self, source: GitHubSource):
        self.source = source
        self._last_sync: float | None = None

    def sync(self, target_dir: str | Path) -> SyncResult:
        """Clone or pull repos into target_dir/github/{owner}/{repo}/."""
        result = SyncResult(source_type="github")
        target = Path(target_dir) / "github" / self.source.owner
        target.mkdir(parents=True, exist_ok=True)

        repos = self._discover_repos(result)
        if not repos:
            return result

        logger.info("Syncing %d repos for %s", len(repos), self.source.owner)

        for repo_name in repos:
            repo_dir = target / repo_name
            try:
                if repo_dir.exists() and (repo_dir / ".git").is_dir():
                    changed = self._pull_repo(repo_dir)
                    if changed:
                        result.files_updated += changed
                else:
                    self._clone_repo(repo_name, repo_dir)
                    result.files_added += 1
            except subprocess.CalledProcessError as e:
                msg = f"{self.source.owner}/{repo_name}: {e.stderr or e}"
                result.errors.append(msg)
                logger.warning("Git error: %s", msg)

        self._last_sync = time.time()
        self._save_metadata(target, repos)
        return result

    def status(self) -> dict:
        """Return current GitHub connector status."""
        return {
            "source_type": "github",
            "owner": self.source.owner,
            "last_sync": self._last_sync,
        }

    def _discover_repos(self, result: SyncResult) -> list[str]:
        """Discover repos from config or by listing from GitHub."""
        if self.source.repos:
            return list(self.source.repos)

        try:
            cmd = [
                "gh", "repo", "list", self.source.owner,
                "--json", "name,isFork,isArchived",
                "--limit", "500",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                result.errors.append(f"gh repo list failed: {proc.stderr.strip()}")
                return []

            repos_data = json.loads(proc.stdout)
            names = []
            for r in repos_data:
                if not self.source.include_forks and r.get("isFork"):
                    continue
                if not self.source.include_archived and r.get("isArchived"):
                    continue
                names.append(r["name"])
            return names

        except FileNotFoundError:
            result.errors.append("gh CLI not found. Install: https://cli.github.com/")
            return []
        except subprocess.TimeoutExpired:
            result.errors.append("gh repo list timed out")
            return []
        except json.JSONDecodeError as e:
            result.errors.append(f"Failed to parse gh output: {e}")
            return []

    def _clone_repo(self, repo_name: str, repo_dir: Path):
        """Clone a repo."""
        url = f"https://github.com/{self.source.owner}/{repo_name}.git"
        cmd = ["git", "clone", "--depth", "1"]
        if self.source.branch:
            cmd.extend(["--branch", self.source.branch])
        cmd.extend([url, str(repo_dir)])
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        logger.info("Cloned %s/%s", self.source.owner, repo_name)

    def _pull_repo(self, repo_dir: Path) -> int:
        """Pull latest changes. Returns number of changed files."""
        # Get current HEAD
        old_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir, timeout=10,
        ).stdout.strip()

        # Pull
        subprocess.run(
            ["git", "pull", "--ff-only", "--depth", "1"],
            capture_output=True, text=True, cwd=repo_dir, timeout=120, check=True,
        )

        # Count changed files
        new_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir, timeout=10,
        ).stdout.strip()

        if old_head == new_head:
            return 0

        diff = subprocess.run(
            ["git", "diff", "--name-only", old_head, new_head],
            capture_output=True, text=True, cwd=repo_dir, timeout=30,
        )
        return len(diff.stdout.strip().splitlines()) if diff.stdout.strip() else 1

    def _save_metadata(self, target: Path, repos: list[str]):
        """Save sync metadata."""
        meta = {
            "owner": self.source.owner,
            "repos": repos,
            "last_sync": time.time(),
        }
        meta_path = target / ".sync_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))


# ── Web crawler ──────────────────────────────────────────────────────────────

# Stealth user agent (mimics real Chrome browser — from ScrapePy patterns)
_STEALTH_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Stealth headers to avoid bot detection
_STEALTH_HEADERS = {
    "User-Agent": _STEALTH_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


@dataclass
class WebSource:
    """Configuration for a web source."""
    url: str
    stay_in_path: bool = True
    max_pages: int = 100
    wait_time: float = 1.0
    timeout: int = 30
    stealth: bool = True  # Use stealth headers to bypass basic bot protection


# Extensions to skip
_SKIP_EXT = re.compile(
    r"\.(pdf|doc|docx|jpg|jpeg|png|gif|svg|webp|ico|css|js|woff2?|"
    r"ttf|eot|mp[34]|wav|webm|zip|tar|gz|xml|json)$",
    re.IGNORECASE,
)

# Non-content paths (from ScrapePy NON_CONTENT_PATHS)
_SKIP_PATH = re.compile(
    r"/(?:login|log-in|signin|sign-in|signup|sign-up|register|logout)(?:/|$)"
    r"|/(?:admin|wp-admin|wp-login|dashboard|settings|account|preferences)(?:/|$)"
    r"|/(?:cart|checkout|wishlist|order|orders)(?:/|$)"
    r"|/(?:search|sitemap|feed|rss|atom|print|share|embed)(?:/|$)"
    r"|/(?:contact|privacy|cookie|gdpr|terms|conditions|accessibility)(?:/|$)"
    r"|/cdn-cgi/",
    re.IGNORECASE,
)

# Cloudflare challenge detection
_CLOUDFLARE_MARKERS = [
    "Checking if the site connection is secure",
    "Enable JavaScript and cookies to continue",
    "Just a moment...",
    "cf-browser-verification",
    "challenge-platform",
]

# Link extraction patterns (from ScrapePy — includes data-href)
_RE_HREF = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
_RE_DATA_HREF = re.compile(r'(?:data-href|data-url|data-link)=["\']([^"\']+)["\']', re.IGNORECASE)


def _is_cloudflare_challenge(html: str) -> bool:
    """Detect Cloudflare or similar bot protection challenge pages."""
    return any(marker in html for marker in _CLOUDFLARE_MARKERS)


class WebCrawler(BaseConnector):
    """BFS web crawler that converts HTML to Markdown files.

    Features (inspired by ScrapePy):
    - Stealth HTTP headers to bypass basic bot detection
    - Cloudflare challenge detection
    - data-href/data-url link extraction
    - Incremental sync via content hashing
    - Configurable path scoping

    Requires httpx (pip install httpx).
    Stores crawled pages as .md files in target_dir/web/{domain}/.
    """

    def __init__(self, source: WebSource):
        self.source = source
        self._last_sync: float | None = None
        self._cloudflare_blocked: int = 0

    def sync(self, target_dir: str | Path) -> SyncResult:
        """Crawl website and save pages as Markdown."""
        if _httpx is None:
            return SyncResult(
                source_type="web",
                errors=["httpx not installed. Run: pip install httpx"],
            )

        result = SyncResult(source_type="web")
        parsed = urlparse(self.source.url)
        domain = parsed.netloc.replace("www.", "")
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        start_path = parsed.path or "/"

        out_dir = Path(target_dir) / "web" / domain
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load cache for incremental
        cache = self._load_cache(out_dir)

        visited: set[str] = set()
        queue: list[str] = [self.source.url]
        self._cloudflare_blocked = 0

        headers = dict(_STEALTH_HEADERS) if self.source.stealth else {
            "User-Agent": "Mullet/1.0 (knowledge crawler)"
        }

        client = _httpx.Client(
            timeout=self.source.timeout,
            follow_redirects=True,
            headers=headers,
        )

        try:
            while queue and len(visited) < self.source.max_pages:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)

                try:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        if resp.status_code == 403:
                            self._cloudflare_blocked += 1
                        continue
                    content_type = resp.headers.get("content-type", "")
                    if "text/html" not in content_type:
                        continue
                except Exception as e:
                    result.errors.append(f"{url}: {e}")
                    continue

                html = resp.text

                # Detect bot protection challenge pages
                if _is_cloudflare_challenge(html):
                    self._cloudflare_blocked += 1
                    logger.warning("Cloudflare challenge detected on %s", url)
                    continue

                content_hash = hashlib.sha256(html.encode()).hexdigest()

                # Skip if unchanged
                if cache.get(url) == content_hash:
                    continue

                # Convert to markdown
                markdown = self._html_to_markdown(html)
                if not markdown or len(markdown.strip()) < 50:
                    continue

                # Save
                filename = self._url_to_filename(url, parsed.netloc)
                filepath = out_dir / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(markdown)

                if url in cache:
                    result.files_updated += 1
                else:
                    result.files_added += 1
                cache[url] = content_hash

                # Extract links (href + data-href)
                for link in self._extract_links(html, url, base_url, start_path):
                    if link not in visited:
                        queue.append(link)

                if self.source.wait_time > 0:
                    time.sleep(self.source.wait_time)

        finally:
            client.close()

        if self._cloudflare_blocked > 0:
            result.errors.append(
                f"{self._cloudflare_blocked} pages blocked by bot protection "
                "(consider using ScrapePy with StealthyFetcher for JS-rendered sites)"
            )

        self._save_cache(out_dir, cache)
        self._last_sync = time.time()

        logger.info(
            "Crawled %s: %d pages visited, %d added, %d updated, %d blocked",
            domain, len(visited), result.files_added, result.files_updated,
            self._cloudflare_blocked,
        )
        return result

    def status(self) -> dict:
        """Return current web crawler status."""
        return {
            "source_type": "web",
            "url": self.source.url,
            "last_sync": self._last_sync,
            "cloudflare_blocked": self._cloudflare_blocked,
        }

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to Markdown with structure preservation."""
        # Remove non-content blocks
        text = re.sub(r'<(script|style|nav|header|footer|aside|noscript)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Convert headings
        for i in range(1, 7):
            text = re.sub(rf'<h{i}[^>]*>(.*?)</h{i}>', rf'\n\n{"#" * i} \1\n\n', text, flags=re.DOTALL | re.IGNORECASE)
        # Convert paragraphs and divs to double newlines
        text = re.sub(r'<(?:p|div|article|section|main)[^>]*>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</(?:p|div|article|section|main)>', '\n', text, flags=re.IGNORECASE)
        # Convert line breaks
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        # Convert lists
        text = re.sub(r'<li[^>]*>', '\n- ', text, flags=re.IGNORECASE)
        # Convert code blocks
        text = re.sub(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', r'\n```\n\1\n```\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.DOTALL | re.IGNORECASE)
        # Bold/italic
        text = re.sub(r'<(?:strong|b)[^>]*>(.*?)</(?:strong|b)>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<(?:em|i)[^>]*>(.*?)</(?:em|i)>', r'*\1*', text, flags=re.DOTALL | re.IGNORECASE)
        # Strip remaining tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode entities
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'&\w+;', '', text)
        # Collapse whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def _extract_links(self, html: str, current_url: str, base_url: str, start_path: str) -> list[str]:
        """Extract links from href and data-href/data-url attributes."""
        raw_links = set()
        # Standard href links
        for match in _RE_HREF.finditer(html):
            raw_links.add(match.group(1))
        # data-href/data-url links (SPA frameworks)
        for match in _RE_DATA_HREF.finditer(html):
            raw_links.add(match.group(1))

        links = []
        for href in raw_links:
            normalized = self._normalize_link(href, current_url, base_url, start_path)
            if normalized:
                links.append(normalized)
        return links

    def _normalize_link(self, href: str, current_url: str, base_url: str, start_path: str) -> str | None:
        """Normalize a link to an absolute URL. Returns None for out-of-scope."""
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#", "data:", "blob:")):
            return None

        # Resolve
        if href.startswith("/"):
            full = base_url + href
        elif href.startswith("http"):
            full = href
        else:
            full = urljoin(current_url, href)

        # Strip fragment and query
        parsed = urlparse(full)
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean.endswith("/"):
            clean = clean.rstrip("/")

        # Same domain only
        if parsed.netloc != urlparse(base_url).netloc:
            return None

        # Path scope
        if self.source.stay_in_path and not parsed.path.startswith(start_path):
            return None

        # Skip file extensions
        if _SKIP_EXT.search(parsed.path):
            return None

        # Skip non-content paths
        if _SKIP_PATH.search(parsed.path):
            return None

        return clean

    def _url_to_filename(self, url: str, netloc: str) -> str:
        """Convert URL to a filesystem-safe filename."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            path = "index"
        safe = path.replace("/", "__")
        safe = re.sub(r'[^a-zA-Z0-9_\-.]', '_', safe)
        return f"{safe}.md"

    def _load_cache(self, out_dir: Path) -> dict[str, str]:
        """Load URL -> content hash cache."""
        cache_path = out_dir / ".crawl_cache.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except (json.JSONDecodeError, OSError) as _exc:
                logger.warning("sources: Failed to load crawl cache: %s", _exc)
        return {}

    def _save_cache(self, out_dir: Path, cache: dict[str, str]):
        """Save crawl cache."""
        cache_path = out_dir / ".crawl_cache.json"
        cache_path.write_text(json.dumps(cache, indent=2))
