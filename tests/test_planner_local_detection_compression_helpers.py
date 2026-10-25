"""Tests for local_detection cache helpers and compression constants."""


from mullet.core.providers import local_detection as ld
from mullet.core.agent.compression import (
    get_context_length,
    CHARS_PER_TOKEN,
    MODEL_CONTEXT_LENGTHS,
)


# ---------------------------------------------------------------------------
# local_detection — pure cache helpers
# ---------------------------------------------------------------------------

class TestLocalDetectionConstants:
    def test_local_ports_is_list(self):
        assert isinstance(ld.LOCAL_PORTS, list)

    def test_local_ports_nonempty(self):
        assert len(ld.LOCAL_PORTS) > 0

    def test_lm_studio_port_present(self):
        assert 1234 in ld.LOCAL_PORTS

    def test_ollama_port_present(self):
        assert 11434 in ld.LOCAL_PORTS

    def test_cache_ttl_positive(self):
        assert ld._CACHE_TTL > 0

    def test_cache_ttl_reasonable(self):
        assert ld._CACHE_TTL <= 300.0


class TestClearLocalProviderCache:
    def setup_method(self):
        ld.clear_local_provider_cache()

    def test_clears_cached_provider(self):
        ld.clear_local_provider_cache()
        assert ld._cached_local_provider is None

    def test_clears_timestamp(self):
        ld.clear_local_provider_cache()
        assert ld._cache_timestamp == 0.0

    def test_clears_refresh_task(self):
        ld.clear_local_provider_cache()
        assert ld._refresh_task is None


class TestCacheIsFresh:
    def setup_method(self):
        ld.clear_local_provider_cache()

    def test_returns_false_when_cold(self):
        assert ld._cache_is_fresh() is False

    def test_returns_bool(self):
        assert isinstance(ld._cache_is_fresh(), bool)

    def test_false_when_no_provider(self):
        ld._cached_local_provider = None
        assert ld._cache_is_fresh() is False

    def test_false_when_timestamp_is_zero(self):
        ld._cache_timestamp = 0.0
        assert ld._cache_is_fresh() is False

    def teardown_method(self):
        ld.clear_local_provider_cache()


# ---------------------------------------------------------------------------
# compression — get_context_length and constants
# ---------------------------------------------------------------------------

class TestCompressionConstants:
    def test_chars_per_token_positive(self):
        assert CHARS_PER_TOKEN > 0

    def test_chars_per_token_value(self):
        assert CHARS_PER_TOKEN == 4

    def test_model_context_lengths_is_dict(self):
        assert isinstance(MODEL_CONTEXT_LENGTHS, dict)

    def test_default_key_present(self):
        assert "default" in MODEL_CONTEXT_LENGTHS

    def test_default_context_positive(self):
        assert MODEL_CONTEXT_LENGTHS["default"] > 0

    def test_gpt4_context_large(self):
        assert MODEL_CONTEXT_LENGTHS["gpt-4"] >= 128_000


class TestGetContextLength:
    def test_returns_int(self):
        assert isinstance(get_context_length("gpt-4"), int)

    def test_positive_for_known_model(self):
        assert get_context_length("gpt-4") > 0

    def test_gpt4_context_length(self):
        assert get_context_length("gpt-4") == 128_000

    def test_gpt4o_context_length(self):
        assert get_context_length("gpt-4o") == 128_000

    def test_claude_opus_large_context(self):
        assert get_context_length("claude-opus") > 100_000

    def test_gemini_large_context(self):
        assert get_context_length("gemini") > 100_000

    def test_unknown_model_returns_positive(self):
        assert get_context_length("totally-unknown-xyz") > 0

    def test_empty_string_returns_positive(self):
        assert get_context_length("") > 0

    def test_empty_string_returns_default(self):
        assert get_context_length("") == get_context_length("truly-no-match-xyz")
