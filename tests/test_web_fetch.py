import pytest
from unittest.mock import patch, mock_open
from mullet.core.tools.web_fetch import fetch_url, HTMLStripper

# Test HTMLStripper
def test_html_stripper_basic():
    stripper = HTMLStripper()
    stripper.feed("<h1>Title</h1><p>Some text.</p>")
    assert stripper.get_data() == "TitleSome text."

def test_html_stripper_with_scripts():
    stripper = HTMLStripper()
    stripper.feed("<html><body><script>alert('hi');</script>Text</body></html>")
    assert stripper.get_data() == "Text"

def test_html_stripper_empty():
    stripper = HTMLStripper()
    stripper.feed("")
    assert stripper.get_data() == ""

def test_html_stripper_no_html():
    stripper = HTMLStripper()
    stripper.feed("Just plain text.")
    assert stripper.get_data() == "Just plain text."

# Test fetch_url
@patch('urllib.request.urlopen')
def test_fetch_url_success(mock_urlopen):
    mock_response = mock_open(read_data="<html><body>Hello World!</body></html>")
    mock_response.return_value.__enter__.return_value.read.return_value = b"<html><body>Hello World!</body></html>"
    mock_urlopen.return_value = mock_response.return_value

    result = fetch_url("http://example.com")
    assert result == "Hello World!"

@patch('urllib.request.urlopen')
def test_fetch_url_no_html_tags(mock_urlopen):
    mock_response = mock_open(read_data="Just plain text.")
    mock_response.return_value.__enter__.return_value.read.return_value = b"Just plain text."
    mock_urlopen.return_value = mock_response.return_value

    result = fetch_url("http://example.com")
    assert result == "Just plain text."

@patch('urllib.request.urlopen')
def test_fetch_url_error_handling(mock_urlopen):
    mock_urlopen.side_effect = Exception("Network error")

    with pytest.raises(IOError, match="Failed to fetch URL http://example.com: Network error"):
        fetch_url("http://example.com")

@patch('urllib.request.urlopen')
def test_fetch_url_complex_html(mock_urlopen):
    complex_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Welcome</h1>
        <p>This is a paragraph with <b>bold</b> text.</p>
        <div>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
    </body>
    </html>
    """
    mock_response = mock_open(read_data=complex_html)
    mock_response.return_value.__enter__.return_value.read.return_value = complex_html.encode('utf-8')
    mock_urlopen.return_value = mock_response.return_value

    result = fetch_url("http://example.com/complex")
    expected_text = "Test PageWelcomeThis is a paragraph with bold text.Item 1Item 2"
    # Normalize whitespace for comparison
    assert "".join(result.split()) == "".join(expected_text.split())
