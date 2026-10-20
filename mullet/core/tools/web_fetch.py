"""HTML stripping + URL fetching utility for the web_fetch agent tool."""
import urllib.request
from html.parser import HTMLParser

class HTMLStripper(HTMLParser):
    """Parse HTML and extract plain text, ignoring script tags."""

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        """Track when we enter a script tag."""
        if tag == 'script':
            self.in_script = True

    def handle_endtag(self, tag):
        """Track when we leave a script tag."""
        if tag == 'script':
            self.in_script = False

    def handle_data(self, data):
        """Collect text data outside of script tags."""
        if not self.in_script:
            self.text.append(data)

    def get_data(self):
        """Return the extracted plain text."""
        return ''.join(self.text)

def fetch_url(url):
    """
    Fetches the content of a given URL and returns the extracted text,
    stripping HTML tags.
    """
    from mullet.core.tools.url_safety import is_safe_url
    if not is_safe_url(url):
        raise OSError(f"URL blocked by SSRF protection: {url}")

    try:
        with urllib.request.urlopen(url) as response:
            html_content = response.read().decode('utf-8')
            stripper = HTMLStripper()
            stripper.feed(html_content)
            return stripper.get_data()
    except Exception as e:
        raise OSError(f"Failed to fetch URL {url}: {e}") from e

