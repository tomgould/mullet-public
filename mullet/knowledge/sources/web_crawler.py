"""Web source connector for Trove.

Crawls websites and converts pages to Markdown for Trove indexing.
Supports incremental crawling via ETags/content hashing.

Ported from ScrapePy — simplified to synchronous httpx-based crawling
with BFS link following, scope control, and junk content filtering.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import logging
import posixpath
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import httpx

from mullet.core.tools.url_safety import httpx_request_guard

from .base import BaseSourceConnector, SyncResult
from .web_cache import CrawlCache
from .web_converter import extract_published_date, extract_title, html_to_markdown

logger = logging.getLogger(__name__)

# File extensions to skip when crawling
SKIP_EXTENSIONS = re.compile(
    r"\.(pdf|doc|docx|jpg|jpeg|png|gif|svg|webp|ico|css|js|woff|woff2|"
    r"ttf|eot|mp4|mp3|wav|webm|zip|tar|gz|xml|json|txt)$",
    re.IGNORECASE,
)

# Non-content paths to skip
NON_CONTENT_PATHS = re.compile(
    r"/(?:login|log-in|signin|sign-in|signup|sign-up|register|logout)(?:/|$)"
    r"|/(?:settings|account|preferences|notifications)(?:/|$)"
    r"|/(?:cart|checkout|wishlist|order|orders)(?:/|$)"
    r"|/(?:admin|wp-admin|wp-login|dashboard)(?:/|$)"
    r"|/(?:share|embed|print)(?:/|$)"
    r"|^/(?:author|authors|topic|topics|tag|tags)/"
    r"|/(?:rss|feed|atom)(?:\.xml)?/?$"
    r"|^/(?:page|p)/\d+"
    r"|^/cdn-cgi/"
    r"|/(?:contact|about|privacy|cookie|gdpr|accessibility)(?:-|/|$)"
    r"|/(?:subscribe|subscriptions|newsletter|advertise)(?:-|/|$)"
    r"|/(?:search|sitemap|archive)(?:-|/|$|\.)"
    r"|/(?:terms|conditions)(?:-|/|$)",
    re.IGNORECASE,
)


@dataclass
class WebSourceConfig:
    """Configuration for a web source in config.yaml."""

    url: str = ""
    mode: str = "crawl"  # crawl, docs, swagger
    stay_in_path: bool = True
    max_pages: int = 500
    wait_time: float = 1.0
    timeout: int = 30
    # Regexes for paths to crawl even though NON_CONTENT_PATHS would skip them
    # (e.g. a news section living under /about-*)
    allow_paths: list[str] = field(default_factory=list)
    # Seed the crawl queue from /sitemap.xml — reaches pages that are only
    # linked from JS-driven pagination
    use_sitemap: bool = False
    # Linked documents to save for indexing, by extension (e.g. ["pdf", "docx"]).
    # Empty = pages only. Each file is fetched once; later crawls skip it.
    download_types: list[str] = field(default_factory=list)
    max_file_mb: float = 50.0

    @classmethod
    def from_dict(cls, data: dict) -> WebSourceConfig:
        """Create a WebSourceConfig from a dictionary."""
        return cls(
            url=data.get("url", ""),
            mode=data.get("mode", "crawl"),
            stay_in_path=data.get("stay_in_path", True),
            max_pages=data.get("max_pages", 500),
            wait_time=data.get("wait_time", 1.0),
            timeout=data.get("timeout", 30),
            allow_paths=list(data.get("allow_paths") or []),
            use_sitemap=bool(data.get("use_sitemap", False)),
            download_types=list(data.get("download_types") or []),
            max_file_mb=data.get("max_file_mb", 50.0),
        )

    @property
    def domain(self) -> str:
        """Return the domain extracted from the source URL."""
        parsed = urlparse(self.url)
        return parsed.netloc.replace("www.", "")

    @property
    def base_url(self) -> str:
        """Return the scheme and domain of the source URL."""
        parsed = urlparse(self.url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @property
    def starting_path(self) -> str:
        """Return the path component of the source URL."""
        parsed = urlparse(self.url)
        return parsed.path or "/"


def normalize_link(
    link: str, current_path: str, base_url: str,
    keep_extensions: frozenset[str] = frozenset(),
) -> str | None:
    """Normalize a link to an absolute path on the same domain.

    Returns None for external/skip links. ``keep_extensions`` (lowercase, no
    dot) exempts those file types from SKIP_EXTENSIONS so they can be downloaded.
    """
    if not link:
        return None

    if link.startswith(("mailto:", "tel:", "javascript:", "data:", "blob:", "about:")):
        return None

    clean = link.split("?")[0].split("#")[0]
    if not clean:
        return None

    skipped = SKIP_EXTENSIONS.search(clean)
    if skipped and skipped.group(1).lower() not in keep_extensions:
        return None

    if clean.startswith(base_url):
        clean = clean[len(base_url):]
    elif clean.startswith("http://") or clean.startswith("https://"):
        return None  # external

    if not clean.startswith("/"):
        current_dir = current_path.rsplit("/", 1)[0] if "/" in current_path else "/"
        clean = posixpath.join(current_dir, clean)

    clean = posixpath.normpath(clean)
    if not clean.startswith("/"):
        clean = "/" + clean

    return clean


def _is_in_scope(url_path: str, config: WebSourceConfig) -> bool:
    """Check if a URL path is within the configured scope."""
    if not config.stay_in_path:
        return True
    starting = config.starting_path
    if not starting or starting == "/":
        return True
    normalised = url_path.split("?")[0].split("#")[0]
    return normalised == starting or normalised.startswith(starting + "/")


def _extract_links(html: str) -> list[str]:
    """Quick regex-based link extraction from HTML."""
    links = set()
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
        links.add(m.group(1))
    return list(links)


def _url_to_filepath(url_path: str) -> str:
    """Convert a URL path to a safe filename."""
    path = url_path.strip("/")
    if not path:
        path = "index"
    # Replace path separators with underscores for flat storage
    path = path.replace("/", "_")
    # Remove unsafe characters
    path = re.sub(r'[<>:"|?*]', "_", path)
    if not path.endswith(".md"):
        path += ".md"
    return path


# MIME types of downloadable documents, used when neither the
# Content-Disposition filename nor the URL carries an extension
DOCUMENT_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/rtf": "rtf",
    "text/rtf": "rtf",
    "application/epub+zip": "epub",
    "text/plain": "txt",
    "text/csv": "csv",
}

# Cap on sitemap documents fetched per crawl (index + child sitemaps)
SITEMAP_MAX_FILES = 50


def _disposition_filename(headers) -> str:
    """Return the filename from a Content-Disposition header, or empty string."""
    disposition = headers.get("content-disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", disposition, re.IGNORECASE)
    return unquote(m.group(1)).strip() if m else ""


def _document_extension(url_path: str, headers) -> str:
    """Best-guess extension (lowercase, no dot) for a downloaded resource."""
    suffix = PurePosixPath(_disposition_filename(headers)).suffix
    if suffix:
        return suffix.lower().lstrip(".")
    mime = headers.get("content-type", "").split(";")[0].strip().lower()
    if mime in DOCUMENT_MIME_TYPES:
        return DOCUMENT_MIME_TYPES[mime]
    return PurePosixPath(url_path).suffix.lower().lstrip(".")


def _download_filename(url_path: str, headers, ext: str) -> str:
    """Stable local filename: the original name's stem plus a short URL hash.

    The hash keeps names unique when two downloads share a title, and stable
    across crawls so re-downloads overwrite rather than duplicate.
    """
    original = _disposition_filename(headers) or PurePosixPath(url_path).name
    stem = re.sub(r"[^\w\-]+", "_", PurePosixPath(original).stem).strip("_")[:80]
    digest = hashlib.sha1(url_path.encode("utf-8")).hexdigest()[:8]
    return f"{stem or 'download'}-{digest}.{ext}"


# Written next to downloaded documents; read by engine/ingest.py
FILE_INDEX_NAME = ".file_index.json"
CRAWL_LOG_NAME = ".crawl_log.jsonl"
CRAWL_LOG_KEEP = 365

# Lines that change without the content changing: download and view counters
# (JouwWeb download widgets show "205 downloads")
_VOLATILE_LINES = re.compile(r"^\s*[\d,.]+\s+(?:downloads?|views?)\s*$", re.IGNORECASE | re.MULTILINE)


def _text_fingerprint(*parts: str) -> str:
    """Hash of a page's converted text, ignoring counter lines."""
    text = _VOLATILE_LINES.sub("", "\n".join(parts))
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _frontmatter(title: str, url: str, published: str | None, entry: dict) -> str:
    """YAML frontmatter for a crawled page.

    ``first_seen`` is only written for pages that appeared after the first
    crawl (non-baseline); ``updated`` once the page text has changed.
    """
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    lines = ["---", f'title: "{safe_title}"', f"url: {url}"]
    if published:
        lines.append(f'published: "{published}"')
    if entry.get("first_seen") and not entry.get("baseline", True):
        lines.append(f'first_seen: "{entry["first_seen"]}"')
    if entry.get("changed_at"):
        lines.append(f'updated: "{entry["changed_at"]}"')
    lines.append("source: web_crawler")
    return "\n".join(lines) + "\n---\n\n"


def _parse_sitemap(xml: str) -> tuple[list[str], bool]:
    """Return (<loc> URLs, is_sitemap_index) for a sitemap document."""
    locs = [html_lib.unescape(m.strip()) for m in re.findall(r"<loc>(.*?)</loc>", xml, re.DOTALL)]
    return locs, "<sitemapindex" in xml


class WebSourceConnector(BaseSourceConnector):
    """Crawl a website and save pages as Markdown for Trove indexing."""

    def __init__(self, config: WebSourceConfig):
        self.config = config
        self._allow_paths = [re.compile(p) for p in config.allow_paths]
        self._download_exts = frozenset(e.lower().lstrip(".") for e in config.download_types)
        self._initial_crawl = False
        self._added: list[str] = []
        self._changed: list[str] = []

    def sync(self, target_dir: str) -> SyncResult:
        """Crawl the website and save Markdown files to target_dir/web/{domain}/.

        Downloaded documents go to target_dir/web/{domain}/files/.
        """
        result = SyncResult(source_type="web")
        target = Path(target_dir)
        web_dir = target / "web" / self.config.domain
        web_dir.mkdir(parents=True, exist_ok=True)

        # Load cache for incremental crawling
        cache = CrawlCache(str(web_dir))
        cache.load()

        existing_files = set(f.name for f in web_dir.glob("*.md"))
        existing_files.update(f.name for f in (web_dir / "files").glob("*"))

        try:
            self._crawl(web_dir, cache, result, existing_files)
        except (httpx.HTTPError, OSError) as e:
            result.errors.append(f"Crawl error: {e}")
            logger.exception("Error crawling %s", self.config.url)

        cache.save()
        self._write_file_index(web_dir / "files", cache)
        self._append_crawl_log(web_dir, result)
        return result

    def _write_file_index(self, files_dir: Path, cache: CrawlCache) -> None:
        """Write files/.file_index.json: each downloaded document's URL and first-seen date.

        Ingest reads it so documents carry their source URL and, when they
        appeared after the first crawl, the date they were first seen.
        """
        entries = cache.file_entries()
        if not entries or not files_dir.is_dir():
            return
        index = {
            e["file"]: {
                "url": self.config.base_url + url_path,
                "first_seen": None if e.get("baseline", True) else e.get("first_seen"),
            }
            for url_path, e in entries.items()
        }
        (files_dir / FILE_INDEX_NAME).write_text(
            json.dumps(index, indent=1, sort_keys=True), encoding="utf-8",
        )

    def _append_crawl_log(self, web_dir: Path, result: SyncResult) -> None:
        """Append this crawl's new and changed URLs to .crawl_log.jsonl (last N kept)."""
        record = {
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "baseline": self._initial_crawl,
            "added": self._added,
            "changed": self._changed,
            "errors": len(result.errors),
        }
        log_path = web_dir / CRAWL_LOG_NAME
        lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
        lines.append(json.dumps(record))
        log_path.write_text("\n".join(lines[-CRAWL_LOG_KEEP:]) + "\n", encoding="utf-8")
        logger.info(
            "Crawl of %s: %d new, %d changed", self.config.domain,
            len(self._added), len(self._changed),
        )

    def status(self) -> dict:
        """Return a dictionary with the connector's current status."""
        return {
            "source_type": "web",
            "url": self.config.url,
            "domain": self.config.domain,
            "mode": self.config.mode,
            "stay_in_path": self.config.stay_in_path,
            "max_pages": self.config.max_pages,
        }

    def _crawl(
        self, output_dir: Path, cache: CrawlCache, result: SyncResult,
        existing_files: set[str] | None = None,
    ) -> int:
        """BFS crawl starting from config URL. Returns pages crawled."""
        from collections import deque

        visited: set[str] = set()
        queue: deque[str] = deque()
        pages_crawled = 0
        known_files = existing_files or set()
        # On a pool's very first crawl everything is "baseline": its first-seen
        # date is the crawl date, not when it appeared on the site
        self._initial_crawl = cache.page_count == 0
        self._added = []
        self._changed = []

        # Seed with starting path
        starting_path = self.config.starting_path
        queue.append(starting_path)
        files_dir = output_dir / "files"

        with httpx.Client(
            timeout=self.config.timeout,
            follow_redirects=True,
            event_hooks={"request": [httpx_request_guard]},
        ) as client:
            if self.config.use_sitemap:
                queue.extend(self._sitemap_paths(client))

            while queue and pages_crawled < self.config.max_pages:
                url_path = queue.popleft()

                if url_path in visited:
                    continue
                visited.add(url_path)

                # Skip non-content paths, unless explicitly allowed
                if NON_CONTENT_PATHS.search(url_path) and not any(
                    p.search(url_path) for p in self._allow_paths
                ):
                    continue

                # Scope check
                if not _is_in_scope(url_path, self.config):
                    continue

                # Documents are fetched once; skip ones already on disk
                saved = cache.get_file(url_path)
                if saved and (files_dir / saved).exists():
                    continue

                full_url = self.config.base_url + url_path

                # Fetch with conditional headers — except for a document missing
                # from disk, where a 304 would never restore it
                headers = {} if saved else cache.get_page_headers(url_path)
                try:
                    response = client.get(full_url, headers=headers)
                except httpx.HTTPError as e:
                    result.errors.append(f"{url_path}: {e}")
                    continue

                # 304 Not Modified — reuse cached links
                if response.status_code == 304:
                    cache.mark_seen(url_path, self._initial_crawl)
                    for link in cache.get_page_links(url_path):
                        if link not in visited:
                            queue.append(link)
                    continue

                if response.status_code >= 400:
                    if response.status_code >= 500:
                        result.errors.append(
                            f"{url_path}: server error ({response.status_code})"
                        )
                    elif response.status_code == 429:
                        result.errors.append(f"{url_path}: rate limited (429)")
                        logger.warning("Rate limited while crawling %s", full_url)
                    else:
                        logger.debug(
                            "Skipping %s (HTTP %d)", url_path, response.status_code
                        )
                    continue

                # Check content type — non-HTML is only kept as a document
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type:
                    if self._download_exts and self._save_file(
                        url_path, response, files_dir, cache, result, known_files,
                    ):
                        pages_crawled += 1
                        if self.config.wait_time > 0:
                            time.sleep(self.config.wait_time)
                    continue

                html = response.text

                # Check content hash for unchanged pages
                if cache.is_page_unchanged(url_path, html):
                    cache.mark_seen(url_path, self._initial_crawl)
                    for link in cache.get_page_links(url_path):
                        if link not in visited:
                            queue.append(link)
                    continue

                # Extract and queue links
                raw_links = _extract_links(html)
                normalized_links = []
                for link in raw_links:
                    norm = normalize_link(
                        link, url_path, self.config.base_url,
                        keep_extensions=self._download_exts,
                    )
                    if norm and norm not in visited:
                        normalized_links.append(norm)
                        queue.append(norm)

                # Convert to Markdown
                md = html_to_markdown(html)
                if not md:
                    # Update cache even for pages with no content
                    cache.update_page(
                        url_path,
                        dict(response.headers),
                        html,
                        links=normalized_links,
                    )
                    continue

                title = extract_title(html)
                published = extract_published_date(html)
                text_hash = _text_fingerprint(title, published or "", md)
                prev_hash = cache.get_entry(url_path).get("text_hash")
                filename = _url_to_filepath(url_path)
                filepath = output_dir / filename
                cache.mark_seen(url_path, self._initial_crawl)

                # Same text as last time — the raw HTML only differed by
                # per-request noise, so don't rewrite (and re-embed) the page
                if prev_hash == text_hash and filepath.exists():
                    cache.update_page(
                        url_path, dict(response.headers), html,
                        links=normalized_links, text_hash=text_hash,
                    )
                    continue

                # A page cached before text hashing existed has no prev_hash:
                # it is written once to record the hash, not counted as changed
                changed = prev_hash is not None
                cache.update_page(
                    url_path, dict(response.headers), html,
                    links=normalized_links, text_hash=text_hash, changed=changed,
                )
                entry = cache.get_entry(url_path)
                filepath.write_text(
                    _frontmatter(title, full_url, published, entry) + md, encoding="utf-8",
                )
                if changed:
                    self._changed.append(full_url)
                elif not entry.get("baseline"):
                    self._added.append(full_url)

                if filename in known_files:
                    result.files_updated += 1
                else:
                    result.files_added += 1
                    known_files.add(filename)
                pages_crawled += 1

                logger.info(
                    "Crawled %s (%d/%d)",
                    url_path,
                    pages_crawled,
                    self.config.max_pages,
                )

                # Politeness delay
                if self.config.wait_time > 0:
                    time.sleep(self.config.wait_time)

        return pages_crawled

    def _sitemap_paths(self, client: httpx.Client) -> list[str]:
        """Collect in-scope page paths from /sitemap.xml, following sitemap indexes."""
        pending = [self.config.base_url + "/sitemap.xml"]
        fetched: set[str] = set()
        paths: list[str] = []
        while pending and len(fetched) < SITEMAP_MAX_FILES:
            sitemap_url = pending.pop(0)
            if sitemap_url in fetched:
                continue
            fetched.add(sitemap_url)
            try:
                response = client.get(sitemap_url)
            except httpx.HTTPError as e:
                logger.warning("Sitemap fetch failed for %s: %s", sitemap_url, e)
                continue
            if response.status_code >= 400:
                logger.info("No sitemap at %s (HTTP %d)", sitemap_url, response.status_code)
                continue
            locs, is_index = _parse_sitemap(response.text)
            if is_index:
                pending.extend(loc for loc in locs if loc.startswith(self.config.base_url))
                continue
            for loc in locs:
                norm = normalize_link(
                    loc, "/", self.config.base_url, keep_extensions=self._download_exts,
                )
                if norm and _is_in_scope(norm, self.config):
                    paths.append(norm)
        logger.info("Sitemap seeded %d paths for %s", len(paths), self.config.domain)
        return paths

    def _save_file(
        self, url_path: str, response: httpx.Response, files_dir: Path,
        cache: CrawlCache, result: SyncResult, known_files: set[str],
    ) -> bool:
        """Save a downloaded document if its type is wanted. Returns True if written."""
        ext = _document_extension(url_path, response.headers)
        if ext not in self._download_exts:
            return False
        content = response.content
        if len(content) > self.config.max_file_mb * 1024 * 1024:
            logger.info(
                "Skipping %s (%.1f MB > max_file_mb)", url_path, len(content) / 1048576,
            )
            return False

        filename = _download_filename(url_path, response.headers, ext)
        files_dir.mkdir(parents=True, exist_ok=True)
        (files_dir / filename).write_bytes(content)
        seen = cache.mark_seen(url_path, self._initial_crawl)
        cache.update_file(url_path, dict(response.headers), filename)
        if not seen.get("baseline"):
            self._added.append(self.config.base_url + url_path)

        if filename in known_files:
            result.files_updated += 1
        else:
            result.files_added += 1
            known_files.add(filename)
        logger.info("Downloaded %s -> files/%s", url_path, filename)
        return True
