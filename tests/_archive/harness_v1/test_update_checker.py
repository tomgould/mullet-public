"""Tests for the update checker."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mullet.harness.update_checker import (
    check_for_update,
    show_update_notification,
    _write_cache,
)


def test_check_uses_cache_when_fresh(tmp_path: Path) -> None:
    cache = tmp_path / ".update-cache"
    with patch("mullet.harness.update_checker._cache_path", return_value=cache):
        _write_cache({"latest": "99.0.0", "url": "http://example.com"})
        with patch("mullet.harness.update_checker._fetch_latest") as mock_fetch:
            info = check_for_update(current_version="1.0.0")
            mock_fetch.assert_not_called()
            assert info is not None
            assert info["latest"] == "99.0.0"


def test_check_fetches_when_cache_stale(tmp_path: Path) -> None:
    cache = tmp_path / ".update-cache"
    with patch("mullet.harness.update_checker._cache_path", return_value=cache):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"latest": "99.0.0", "url": "http://example.com", "cached_at": 0}),
            encoding="utf-8",
        )
        with patch(
            "mullet.harness.update_checker._fetch_latest",
            return_value={"latest": "100.0.0", "url": "http://example.com/v2"},
        ) as mock_fetch:
            info = check_for_update(current_version="1.0.0")
            mock_fetch.assert_called_once()
            assert info is not None
            assert info["latest"] == "100.0.0"


def test_show_update_notification_format() -> None:
    info = {
        "latest": "1.2.0",
        "url": "https://example.com",
        "current": "1.0.0",
        "behind_by": "0.2.0",
    }
    text = show_update_notification(info)
    assert "Mullet 1.2.0 available" in text
    assert "you have 1.0.0" in text
    assert "pip install -U mullet" in text


def test_check_returns_none_when_up_to_date() -> None:
    cache = Path(__file__).parent / ".tmp-cache"
    with patch("mullet.harness.update_checker._cache_path", return_value=cache):
        _write_cache({"latest": "1.0.0", "url": "http://example.com"})
        result = check_for_update(current_version="1.0.0")
        assert result is None
    if cache.exists():
        cache.unlink()
