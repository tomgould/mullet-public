"""Web search tool — uses DuckDuckGo HTML search for real results.

Falls back to the Instant Answer API if HTML parsing fails.
No API key required.
"""

import json
import re
import urllib.request
import urllib.parse

WEB_SEARCH_SCHEMA = {
    "description": "Search the web for information. Returns relevant results with titles, snippets, and URLs.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results. Default: 5.",
            },
        },
        "required": ["query"],
    },
}

FETCH_PAGE_SCHEMA = {
    "description": "Fetch a web page and return its text content (HTML stripped).",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return. Default: 10000.",
            },
        },
        "required": ["url"],
    },
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def web_search(query: str, max_results: int = 5, **_kwargs) -> str:
    """Search the web using DuckDuckGo HTML search."""
    try:
        results = _ddg_html_search(query, max_results)
        if not results:
            results = _ddg_api_search(query, max_results)

        if not results:
            return f"No results found for: {query}"

        lines = []
        for i, r in enumerate(results[:max_results], 1):
            lines.append(f"{i}. **{r['title']}**")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            if r.get("url"):
                lines.append(f"   {r['url']}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"[error] Web search failed: {e}"


def fetch_page(url: str, max_chars: int = 10000, **_kwargs) -> str:
    """Fetch a web page and return text content."""
    try:
        from mullet.core.tools.url_safety import is_safe_url
        if not is_safe_url(url):
            return f"[error] URL blocked by SSRF protection: {url}"

        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                html = resp.read().decode("utf-8", errors="replace")
                text = _html_to_text(html)
            else:
                text = resp.read().decode("utf-8", errors="replace")

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[...truncated at {max_chars} chars]"

        return text.strip() or "(empty page)"
    except Exception as e:
        return f"[error] Failed to fetch {url}: {e}"


def _ddg_html_search(query: str, max_results: int) -> list[dict]:
    """Search DuckDuckGo via HTML endpoint (returns real web results)."""
    params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
    url = f"https://html.duckduckgo.com/html/?{params}"
    req = urllib.request.Request(url, headers=_HEADERS)

    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    results = []
    # Parse result blocks — DuckDuckGo HTML uses class="result__a" for links
    # and class="result__snippet" for descriptions
    link_pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title) in enumerate(links[:max_results]):
        # Decode DuckDuckGo redirect URLs
        actual_url = href
        if "uddg=" in href:
            match = re.search(r'uddg=([^&]+)', href)
            if match:
                actual_url = urllib.parse.unquote(match.group(1))

        snippet = ""
        if i < len(snippets):
            snippet = _strip_html(snippets[i])

        results.append({
            "title": _strip_html(title).strip(),
            "url": actual_url,
            "snippet": snippet.strip(),
        })

    return results


def _ddg_api_search(query: str, max_results: int) -> list[dict]:
    """Fallback: DuckDuckGo Instant Answer API."""
    encoded = urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1})
    url = f"https://api.duckduckgo.com/?{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mullet/0.1"})

    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    results = []
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", query),
            "snippet": data["AbstractText"],
            "url": data.get("AbstractURL", ""),
        })

    for topic in data.get("RelatedTopics", [])[:max_results]:
        if isinstance(topic, dict) and "Text" in topic:
            results.append({
                "title": topic.get("Text", "")[:80],
                "snippet": topic.get("Text", ""),
                "url": topic.get("FirstURL", ""),
            })

    return results


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text, preserving structure."""
    # Remove script and style blocks
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newlines
    text = re.sub(r'<(?:p|div|br|h[1-6]|li|tr)[^>]*>', '\n', text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def register(registry):
    """Register web tools into a ToolRegistry."""
    registry.register(
        name="web_search",
        func=web_search,
        schema=WEB_SEARCH_SCHEMA,
        toolset="web",
    )
    registry.register(
        name="fetch_page",
        func=fetch_page,
        schema=FETCH_PAGE_SCHEMA,
        toolset="web",
    )
