"""Tests for TTL cache decorator."""

import time
from mullet.knowledge.engine.cache import cached, clear_all_caches


def setup_function():
    clear_all_caches()


def test_cached_returns_same_value():
    call_count = 0

    @cached(ttl=60)
    def fetch(key):
        nonlocal call_count
        call_count += 1
        return f"value:{key}"

    assert fetch("a") == "value:a"
    assert fetch("a") == "value:a"
    assert call_count == 1  # second call was cached


def test_cached_different_args_not_shared():
    call_count = 0

    @cached(ttl=60)
    def fetch(key):
        nonlocal call_count
        call_count += 1
        return f"value:{key}"

    fetch("x")
    fetch("y")
    assert call_count == 2


def test_cached_ttl_zero_disables_caching():
    call_count = 0

    @cached(ttl=0)
    def fetch(key):
        nonlocal call_count
        call_count += 1
        return key

    fetch("a")
    fetch("a")
    assert call_count == 2  # no caching when ttl=0


def test_cached_expires_after_ttl():
    call_count = 0
    _real_time = time.time

    @cached(ttl=0.05)  # 50ms TTL
    def fetch(key):
        nonlocal call_count
        call_count += 1
        return key

    fetch("a")
    time.sleep(0.1)  # wait for TTL to expire
    fetch("a")
    assert call_count == 2  # expired, called again


def test_cache_clear_removes_entries():
    @cached(ttl=60)
    def fetch(key):
        return key

    fetch("b")
    removed = fetch.cache_clear()
    assert removed >= 1


def test_cache_info_reports_counts():
    @cached(ttl=60)
    def fetch(key):
        return key

    fetch("c")
    info = fetch.cache_info()
    assert info["cached"] >= 1


def test_cached_kwargs_are_keyed():
    call_count = 0

    @cached(ttl=60)
    def fetch(query, top_k=10):
        nonlocal call_count
        call_count += 1
        return (query, top_k)

    fetch("test", top_k=5)
    fetch("test", top_k=5)
    fetch("test", top_k=10)
    assert call_count == 2  # different top_k = different cache entry


def test_clear_all_caches():
    @cached(ttl=60)
    def fetch(key):
        return key

    fetch("d")
    clear_all_caches()
    # After clear, info should show 0 from this func
    info = fetch.cache_info()
    assert info["cached"] == 0


class TestMakeKeyFallback:
    def test_circular_reference_uses_str_fallback(self):
        """Lines 61-62: json.dumps raises ValueError on circular ref → str fallback."""
        from mullet.knowledge.engine.cache import _make_key
        circular = {}
        circular["self"] = circular
        key = _make_key("prefix", (circular,), {})
        assert key.startswith("prefix:")
        assert len(key) > len("prefix:")
