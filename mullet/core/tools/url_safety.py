"""URL safety for tools that fetch a URL a model or a remote party supplied.

``is_safe_url`` resolves the host and refuses anything that is not a global
address. ``safe_urlopen`` applies the same check to every redirect hop. Lookup
failures refuse the URL. A host that changes its answer between the check and
the connection (DNS rebinding) is not covered.
"""

import ipaddress
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

_ALLOWED_SCHEMES = ("http", "https")


def _resolve(host: str) -> set[str]:
    """Return every address ``host`` resolves to. Raises OSError on failure."""
    return {info[4][0] for info in socket.getaddrinfo(host, None)}


def _is_global(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def is_safe_url(url: str) -> bool:
    """True only for an http(s) URL whose host resolves to global addresses."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            return False
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host or host == "localhost" or host.endswith(".localhost"):
            return False
        try:
            return _is_global(host)
        except ValueError:
            pass
        addresses = _resolve(host)
        return bool(addresses) and all(_is_global(a) for a in addresses)
    except Exception:
        return False


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not is_safe_url(newurl):
            raise urllib.error.URLError(f"redirect to a non-public address refused: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_urlopen(req: "urllib.request.Request | str", timeout: float):
    """``urlopen`` that refuses a non-public first URL or redirect target."""
    url = req if isinstance(req, str) else req.full_url
    if not is_safe_url(url):
        raise urllib.error.URLError(f"URL blocked: not a public http(s) address: {url}")
    return urllib.request.build_opener(_CheckedRedirects).open(req, timeout=timeout)


def httpx_request_guard(request) -> None:
    """httpx ``request`` event hook: refuse any request, redirect hops included,
    whose URL is not a public http(s) address. Raises ``httpx.RequestError``."""
    import httpx

    if not is_safe_url(str(request.url)):
        raise httpx.RequestError(
            f"URL blocked: not a public http(s) address: {request.url}", request=request
        )
