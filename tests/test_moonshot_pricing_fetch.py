"""Tests for live Moonshot pricing fetch in model_registry."""

import json
import time
from unittest.mock import patch


from mullet.core.providers.model_registry import _fetch_moonshot_pricing


class TestFetchMoonshotPricing:
    def test_cache_hit(self, tmp_path, monkeypatch):
        """Fresh cache on disk should be returned with no HTTP call made."""
        monkeypatch.setattr(
            "mullet.core.providers.model_registry._MOONSHOT_PRICING_CACHE_PATH",
            tmp_path / "moonshot_pricing_cache.json",
        )
        fresh_cache = {
            "timestamp": time.time(),
            "pricing": {"kimi-k2.6": (0.001, 0.004)},
        }
        cache_path = tmp_path / "moonshot_pricing_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(fresh_cache))

        with patch("mullet.core.providers.model_registry._http_get") as mock_get:
            result = _fetch_moonshot_pricing()
            mock_get.assert_not_called()

        assert result == {"kimi-k2.6": (0.001, 0.004)}

    def test_cache_miss_calls_api(self, tmp_path, monkeypatch):
        """Expired/missing cache triggers an HTTP GET via _http_get."""
        monkeypatch.setattr(
            "mullet.core.providers.model_registry._MOONSHOT_PRICING_CACHE_PATH",
            tmp_path / "moonshot_pricing_cache.json",
        )
        stale_cache = {
            "timestamp": time.time() - 25 * 3600,  # 25 hours old
            "pricing": {"old-model": (0.001, 0.002)},
        }
        cache_path = tmp_path / "moonshot_pricing_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(stale_cache))

        api_response = {
            "data": [
                {
                    "id": "kimi-k2.6",
                    "pricing": {"input": 0.00095, "output": 0.004},
                }
            ]
        }

        with patch(
            "mullet.core.providers.model_registry._http_get",
            return_value=api_response,
        ) as mock_get:
            monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
            result = _fetch_moonshot_pricing()
            mock_get.assert_called_once()

        assert result == {"kimi-k2.6": (0.00095, 0.004)}
        # Cache should have been rewritten
        cached = json.loads(cache_path.read_text())
        assert cached["pricing"] == {"kimi-k2.6": [0.00095, 0.004]}

    def test_api_error_returns_empty(self, tmp_path, monkeypatch):
        """If _http_get raises, _fetch_moonshot_pricing returns {}."""
        monkeypatch.setattr(
            "mullet.core.providers.model_registry._MOONSHOT_PRICING_CACHE_PATH",
            tmp_path / "moonshot_pricing_cache.json",
        )
        # No cache file present

        with patch(
            "mullet.core.providers.model_registry._http_get",
            side_effect=Exception("network error"),
        ) as mock_get:
            monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
            result = _fetch_moonshot_pricing()
            mock_get.assert_called_once()

        assert result == {}
