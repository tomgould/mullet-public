"""Provides URL safety checking to prevent SSRF and unsafe schemes."""

import ipaddress
from urllib.parse import urlparse

def is_safe_url(url: str) -> bool:
    """Check if a URL is safe, preventing SSRF and blocking unsafe schemes."""
    try:
        parsed_url = urlparse(url)

        # Block unsafe schemes
        blocked_schemes = ["file", "ftp", "gopher", "tftp", "dict"]
        if parsed_url.scheme in blocked_schemes:
            return False

        # Check for private IPs and localhost to prevent SSRF
        if parsed_url.hostname:
            try:
                ip_address = ipaddress.ip_address(parsed_url.hostname)
                if ip_address.is_private or ip_address.is_loopback:
                    return False
            except ValueError:
                # Not an IP address, could be a domain name
                # Add checks for internal domains if necessary
                pass

        return True
    except Exception:
        return False

