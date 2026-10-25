"""Every httpx request and redirect hop in the crawlers goes through the URL check."""

import inspect

import httpx
import pytest

from mullet.core.tools.url_safety import httpx_request_guard


def _client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        event_hooks={"request": [httpx_request_guard]},
    )


def test_public_request_passes():
    with _client(lambda r: httpx.Response(200, text="ok")) as c:
        assert c.get("https://example.com/").text == "ok"


def test_loopback_request_is_refused():
    with _client(lambda r: httpx.Response(200)) as c:
        with pytest.raises(httpx.RequestError, match="blocked"):
            c.get("http://127.0.0.1:8080/")


def test_redirect_to_private_address_is_refused():
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})
        return httpx.Response(200, text="secret")

    with _client(handler) as c:
        with pytest.raises(httpx.RequestError, match="blocked"):
            c.get("https://example.com/")


def test_redirect_to_name_resolving_privately_is_refused(monkeypatch):
    monkeypatch.setattr(
        "mullet.core.tools.url_safety._resolve",
        lambda host: {"10.1.1.1"} if host == "internal.test" else {"93.184.216.34"},
    )

    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(301, headers={"location": "http://internal.test/"})
        return httpx.Response(200, text="secret")

    with _client(handler) as c:
        with pytest.raises(httpx.RequestError):
            c.get("https://example.com/")


def test_both_crawlers_install_the_guard():
    from mullet.core.knowledge import sources
    from mullet.knowledge.sources import web_crawler

    assert "httpx_request_guard" in inspect.getsource(web_crawler.WebSourceConnector._crawl)
    assert "httpx_request_guard" in inspect.getsource(sources)
