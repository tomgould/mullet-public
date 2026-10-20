"""Check robots.txt compliance for URLs with a bounded in-memory cache."""

import threading
import urllib.robotparser
from typing import Dict, Any
from urllib.parse import urlparse

_MAX_CACHE_SIZE = 100


class RobotsTxtChecker:
    """Thread-safe LRU cache of per-domain ``robots.txt`` parsers."""

    def __init__(self):
        self._cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._lock = threading.Lock()

    def check_robots_txt(self, url: str) -> Dict[str, Any]:
        """
        Checks robots.txt compliance for a given URL.
        Caches results for the session (bounded to _MAX_CACHE_SIZE entries).
        """
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_txt_url = f"{base_url}/robots.txt"

        with self._lock:
            if base_url not in self._cache:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_txt_url)
                try:
                    rp.read()
                except Exception as e:
                    return {"allowed": False, "crawl_delay": None, "disallowed_paths": [], "error": str(e)}
                # Evict oldest entry if cache is full
                if len(self._cache) >= _MAX_CACHE_SIZE:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[base_url] = rp

            rp = self._cache[base_url]

        allowed = rp.can_fetch("MulletBot", url)
        crawl_delay = rp.crawl_delay("MulletBot")

        disallowed_paths = []
        # robotparser doesn't directly expose disallowed paths, so we'll simulate it for common cases
        # This is a simplification; a full implementation would parse rules more thoroughly.
        # For this exercise, we'll just check if the URL itself is disallowed.
        if not allowed:
            disallowed_paths.append(parsed_url.path)

        return {
            "allowed": allowed,
            "crawl_delay": crawl_delay,
            "disallowed_paths": disallowed_paths
        }

# Instantiate the checker for use
robots_checker = RobotsTxtChecker()
