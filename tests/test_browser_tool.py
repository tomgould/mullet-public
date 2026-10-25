import pytest
from unittest.mock import patch, MagicMock
import urllib.request
import urllib.parse
from mullet.core.tools.browser import browse, register, _is_safe_url, _html_to_text, _extract_forms, _extract_element, BROWSE_SCHEMA

# Mock safe_urlopen for all tests
@pytest.fixture
def mock_urlopen():
    with patch('mullet.core.tools.url_safety.safe_urlopen') as mock_urlopen:
        yield mock_urlopen

@pytest.fixture
def mock_socket_gethostbyname():
    with patch('socket.gethostbyname') as mock_gethostbyname:
        yield mock_gethostbyname

# Helper for mock response
def create_mock_response(html_content, status=200):
    mock_resp = MagicMock()
    mock_resp.read.return_value = html_content.encode('utf-8')
    mock_resp.getcode.return_value = status
    return mock_resp

def test_browse_read_action(mock_urlopen):
    mock_urlopen.return_value.__enter__.return_value = create_mock_response("<html><body><h1>Hello</h1><p>World</p></body></html>")
    result = browse(url="http://example.com", action="read")
    assert "Hello\nWorld" in result

def test_browse_links_action(mock_urlopen):
    html = """
    <html>
        <body>
            <a href="/page1">Page 1</a>
            <a href="http://external.com/page2">Page 2</a>
            <a href="javascript:void(0)">JS Link</a>
            <a href="#anchor">Anchor</a>
        </body>
    </html>
    """
    mock_urlopen.return_value.__enter__.return_value = create_mock_response(html)
    result = browse(url="http://example.com", action="links")
    assert "- [Page 1](http://example.com/page1)" in result
    assert "- [Page 2](http://external.com/page2)" in result
    assert "JS Link" not in result
    assert "Anchor" not in result

def test_browse_forms_action(mock_urlopen):
    html = """
    <html>
        <body>
            <form>
                <input type="text" name="username">
                <input type="password" name="password">
                <button type="submit">Login</button>
            </form>
            <form>
                <input type="hidden" name="csrf_token">
            </form>
        </body>
    </html>
    """
    mock_urlopen.return_value.__enter__.return_value = create_mock_response(html)
    result = browse(url="http://example.com", action="forms")
    assert "Form 1:" in result
    assert "  - username (text)" in result
    assert "  - password (password)" in result
    assert "Form 2:" in result
    assert "  - csrf_token (hidden)" in result

def test_browse_extract_title(mock_urlopen):
    mock_urlopen.return_value.__enter__.return_value = create_mock_response("<html><head><title>Test Title</title></head><body></body></html>")
    result = browse(url="http://example.com", action="extract", selector="title")
    assert result == "Test Title"

def test_browse_extract_meta_description(mock_urlopen):
    html = '<html><head><meta name="description" content="A test description"></head><body></body></html>'
    mock_urlopen.return_value.__enter__.return_value = create_mock_response(html)
    result = browse(url="http://example.com", action="extract", selector="meta:description")
    assert result == "A test description"

def test_browse_extract_h1(mock_urlopen):
    html = "<html><body><h1>Heading 1</h1><h2>Subheading</h2><h1>Another H1</h1></body></html>"
    mock_urlopen.return_value.__enter__.return_value = create_mock_response(html)
    result = browse(url="http://example.com", action="extract", selector="h1")
    assert "- Heading 1\n- Another H1" in result

def test_browse_timeout_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError('timed out')
    result = browse(url="http://example.com")
    assert "[error] Failed to fetch http://example.com: <urlopen error timed out>" in result

def test_browse_http_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.HTTPError(url="http://example.com", code=404, msg="Not Found", hdrs={}, fp=None)
    result = browse(url="http://example.com")
    assert "[error] Failed to fetch http://example.com: HTTP Error 404: Not Found" in result

def test_browse_invalid_url(mock_urlopen):
    # _is_safe_url should catch this before urlopen
    result = browse(url="http://localhost")
    assert "[error] URL blocked: internal/private addresses not allowed" in result
    mock_urlopen.assert_not_called()

def test_html_to_text_strips_scripts_and_styles():
    html = "<html><head><style>body{}</style></head><body><script>alert(1)</script><p>Text</p></body></html>"
    result = _html_to_text(html)
    assert "Text" in result
    assert "script" not in result
    assert "style" not in result

def test_max_chars_truncation(mock_urlopen):
    long_text = "A" * 20000
    mock_urlopen.return_value.__enter__.return_value = create_mock_response(f"<html><body>{long_text}</body></html>")
    result = browse(url="http://example.com", action="read", max_chars=100)
    assert len(result) <= 100 + len("\n\n[...truncated at 100 chars]")
    assert "[...truncated at 100 chars]" in result

def test_register_function():
    mock_registry = MagicMock()
    register(mock_registry)
    mock_registry.register.assert_called_once_with(
        name="browse",
        func=browse,
        schema=BROWSE_SCHEMA,
        toolset="browser",
    )

def test_is_safe_url_blocked_localhost(mock_socket_gethostbyname):
    mock_socket_gethostbyname.return_value = "127.0.0.1"
    assert not _is_safe_url("http://localhost")

def test_is_safe_url_blocked_private_ip(monkeypatch):
    monkeypatch.setattr("mullet.core.tools.url_safety._resolve", lambda host: {"192.168.1.1"})
    assert not _is_safe_url("http://internal.com")

def test_is_safe_url_allowed_public_ip(monkeypatch):
    monkeypatch.setattr("mullet.core.tools.url_safety._resolve", lambda host: {"8.8.8.8"})
    assert _is_safe_url("http://google.com")


def test_is_safe_url_exception_returns_true(monkeypatch):
    def _raise(*args, **kwargs):
        raise Exception("dns failure")

    monkeypatch.setattr("socket.gethostbyname", _raise)
    assert _is_safe_url("http://some-url.com") is True


def test_extract_forms_empty_html():
    result = _extract_forms("")
    assert "no forms found" in result


def test_extract_element_no_selector():
    result = _extract_element("<html>hi</html>", "")
    assert "no selector provided" in result


def test_extract_element_no_matching_elements():
    result = _extract_element("<html>hi</html>", "div")
    assert "no elements matching 'div'" in result


def test_extract_element_meta_property():
    html = '<html><head><meta property="og:description" content="Property Desc"></head></html>'
    result = _extract_element(html, "meta:og:description")
    assert result == "Property Desc"
