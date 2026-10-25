"""Browser automation tool — headless web interaction.

Lightweight browser tool that fetches pages, extracts content,
follows links, and fills forms. Uses urllib for zero-dependency
operation. For full browser automation, install playwright.
"""

import re
import urllib.request
import urllib.parse
import logging

logger = logging.getLogger(__name__)

BROWSE_SCHEMA = {
    "description": "Browse a web page — fetch content, extract links, find specific elements. Use for reading documentation, checking web pages, or following links.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to browse."},
            "action": {
                "type": "string",
                "enum": ["read", "links", "forms", "extract"],
                "description": "Action: read (get text), links (list links), forms (list forms), extract (CSS-like selector).",
            },
            "selector": {"type": "string", "description": "For extract action: element to find (e.g., 'title', 'h1', 'meta:description')."},
            "max_chars": {"type": "integer", "description": "Max characters to return. Default: 10000."},
        },
        "required": ["url"],
    },
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _is_safe_url(url: str) -> bool:
    """Block SSRF: reject private/internal IP ranges."""
    from mullet.core.tools.url_safety import is_safe_url
    return is_safe_url(url)


def browse(url: str, action: str = "read", selector: str = "", max_chars: int = 10000, **_kwargs) -> str:
    """Browse a web page with various actions."""
    if not _is_safe_url(url):
        return "[error] URL blocked: internal/private addresses not allowed"
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        from mullet.core.tools.url_safety import safe_urlopen
        with safe_urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"[error] Failed to fetch {url}: {e}"

    if action == "links":
        return _extract_links(html, url)
    elif action == "forms":
        return _extract_forms(html)
    elif action == "extract":
        return _extract_element(html, selector)
    else:  # read
        text = _html_to_text(html)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[...truncated at {max_chars} chars]"
        return text.strip() or "(empty page)"


def _extract_links(html: str, base_url: str) -> str:
    """Extract all links from HTML."""
    links = re.findall(r'<a[^>]+href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE)
    results = []
    for href, text in links[:50]:
        text = re.sub(r'<[^>]+>', '', text).strip()
        if not text or not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        # Resolve relative URLs
        if not href.startswith("http"):
            href = urllib.parse.urljoin(base_url, href)
        results.append(f"- [{text[:60]}]({href})")

    return "\n".join(results) if results else "(no links found)"


def _extract_forms(html: str) -> str:
    """Extract form elements from HTML."""
    forms = re.findall(r'<form[^>]*>(.*?)</form>', html, re.DOTALL | re.IGNORECASE)
    if not forms:
        return "(no forms found)"

    results = []
    for i, form_html in enumerate(forms[:5], 1):
        inputs = re.findall(r'<input[^>]+>', form_html, re.IGNORECASE)
        fields = []
        for inp in inputs:
            name = re.search(r'name="([^"]*)"', inp)
            itype = re.search(r'type="([^"]*)"', inp)
            if name:
                fields.append(f"  - {name.group(1)} ({itype.group(1) if itype else 'text'})")
        results.append(f"Form {i}:")
        results.extend(fields if fields else ["  (no named fields)"])

    return "\n".join(results)


def _extract_element(html: str, selector: str) -> str:
    """Extract a specific element from HTML (simple selector support)."""
    if not selector:
        return "(no selector provided)"

    # Handle common selectors
    if selector == "title":
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "(no title)"

    if selector.startswith("meta:"):
        meta_name = selector[5:]
        match = re.search(
            rf'<meta[^>]+name="{meta_name}"[^>]+content="([^"]*)"',
            html, re.IGNORECASE,
        )
        if not match:
            match = re.search(
                rf'<meta[^>]+property="{meta_name}"[^>]+content="([^"]*)"',
                html, re.IGNORECASE,
            )
        return match.group(1) if match else f"(no meta:{meta_name})"

    # Tag selector (h1, h2, p, etc.)
    tag = selector.lower()
    matches = re.findall(rf'<{tag}[^>]*>(.*?)</{tag}>', html, re.DOTALL | re.IGNORECASE)
    if matches:
        texts = [re.sub(r'<[^>]+>', '', m).strip() for m in matches[:10]]
        return "\n".join(f"- {t}" for t in texts if t)

    return f"(no elements matching '{selector}')"


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<(?:p|div|br|h[1-6]|li|tr)[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def register(registry):
    """Register browser tools."""
    registry.register(
        name="browse",
        func=browse,
        schema=BROWSE_SCHEMA,
        toolset="browser",
    )
