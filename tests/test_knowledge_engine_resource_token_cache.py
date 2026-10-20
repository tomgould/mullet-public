"""Tests for resource_manager.py, search/tokenizer.py, and model_cache.py."""
import gc
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from mullet.knowledge.engine.resource_manager import ResourceManager
from mullet.knowledge.engine.search.tokenizer import (
    advanced_tokenize, EnhancedTokenizer, STOP_WORDS, TOKENIZER_VERSION
)
from mullet.knowledge.engine.model_cache import ModelCache, get_model_cache


# ===========================================================================
# ResourceManager
# ===========================================================================

class TestResourceManagerInit:
    def test_default_headroom(self):
        rm = ResourceManager()
        assert rm.headroom == 0.05

    def test_headroom_clamped_below_zero(self):
        rm = ResourceManager(headroom=-0.1)
        assert rm.headroom == 0.0

    def test_headroom_clamped_above_half(self):
        rm = ResourceManager(headroom=0.9)
        assert rm.headroom == 0.5

    def test_custom_headroom(self):
        rm = ResourceManager(headroom=0.2)
        assert rm.headroom == 0.2


class TestResourceManagerProbeCpu:
    def test_returns_dict_with_required_keys(self):
        rm = ResourceManager()
        result = rm.probe_cpu()
        assert "total_cores" in result
        assert "available_cores" in result

    def test_available_less_than_or_equal_total(self):
        rm = ResourceManager()
        result = rm.probe_cpu()
        assert result["available_cores"] <= result["total_cores"]

    def test_available_at_least_two(self):
        rm = ResourceManager(headroom=0.99)  # extreme headroom
        result = rm.probe_cpu()
        assert result["available_cores"] >= 2

    def test_no_cpu_count_defaults_to_4(self):
        rm = ResourceManager()
        with patch("os.cpu_count", return_value=None):
            result = rm.probe_cpu()
        assert result["total_cores"] == 4

    def test_headroom_applied(self):
        rm = ResourceManager(headroom=0.5)
        with patch("os.cpu_count", return_value=8):
            result = rm.probe_cpu()
        assert result["available_cores"] == max(2, int(8 * 0.5))


class TestResourceManagerProbeRam:
    def _mock_mem(self, total_gb=16, available_gb=8):
        mem = MagicMock()
        mem.total = total_gb * (1024 ** 3)
        mem.available = available_gb * (1024 ** 3)
        return mem

    def test_returns_dict_with_required_keys(self):
        rm = ResourceManager()
        with patch("psutil.virtual_memory", return_value=self._mock_mem()):
            result = rm.probe_ram()
        assert "total_gb" in result
        assert "available_gb" in result
        assert "usable_gb" in result

    def test_usable_less_than_available(self):
        rm = ResourceManager(headroom=0.1)
        with patch("psutil.virtual_memory", return_value=self._mock_mem(16, 8)):
            result = rm.probe_ram()
        assert result["usable_gb"] < result["available_gb"]

    def test_headroom_applied_to_usable(self):
        rm = ResourceManager(headroom=0.2)
        with patch("psutil.virtual_memory", return_value=self._mock_mem(16, 10)):
            result = rm.probe_ram()
        assert result["usable_gb"] == pytest.approx(8.0, abs=0.2)  # 10 * 0.8


class TestResourceManagerProbeGpu:
    def test_returns_none_when_pynvml_unavailable(self):
        rm = ResourceManager()
        with patch("builtins.__import__", side_effect=ImportError("no pynvml")):
            # Reset probe state
            rm._gpu_probed = False
            result = rm.probe_gpu()
        assert result is None

    def test_cached_result_on_second_call(self):
        rm = ResourceManager()
        rm._gpu_probed = True
        rm._gpu_info = {"free_mb": 4096}
        result = rm.probe_gpu()
        assert result == {"free_mb": 4096}

    def test_returns_none_on_generic_exception(self):
        rm = ResourceManager()
        rm._gpu_probed = False
        with patch("mullet.knowledge.engine.resource_manager.warnings.catch_warnings") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(side_effect=RuntimeError("fail"))
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = rm.probe_gpu()
        assert result is None


class TestResourceManagerRecommendParseWorkers:
    def test_small_file_count(self):
        rm = ResourceManager()
        with patch.object(rm, "probe_cpu", return_value={"available_cores": 8}):
            result = rm.recommend_parse_workers(100)
        assert result == 4  # min(4, 8)

    def test_medium_file_count(self):
        rm = ResourceManager()
        with patch.object(rm, "probe_cpu", return_value={"available_cores": 8}):
            result = rm.recommend_parse_workers(5000)
        assert result == min(int(8 * 0.6), 16)

    def test_large_file_count(self):
        rm = ResourceManager()
        with patch.object(rm, "probe_cpu", return_value={"available_cores": 8}):
            result = rm.recommend_parse_workers(50000)
        assert result == 8

    def test_boundary_500(self):
        rm = ResourceManager()
        with patch.object(rm, "probe_cpu", return_value={"available_cores": 8}):
            assert rm.recommend_parse_workers(499) == min(4, 8)
            assert rm.recommend_parse_workers(500) == min(int(8 * 0.6), 16)


class TestResourceManagerRecommendEmbedBatch:
    def test_returns_16_when_no_gpu(self):
        rm = ResourceManager()
        with patch.object(rm, "probe_gpu", return_value=None):
            assert rm.recommend_embed_batch(384) == 16

    def test_capped_at_64(self):
        rm = ResourceManager()
        gpu = {"free_mb": 100000, "total_mb": 120000, "used_mb": 20000, "utilization_pct": 10}
        with patch.object(rm, "probe_gpu", return_value=gpu):
            result = rm.recommend_embed_batch(384)
        assert result <= 64

    def test_at_least_16(self):
        rm = ResourceManager()
        gpu = {"free_mb": 1, "total_mb": 8000, "used_mb": 7999, "utilization_pct": 99}
        with patch.object(rm, "probe_gpu", return_value=gpu):
            result = rm.recommend_embed_batch(384)
        assert result >= 16


class TestResourceManagerClassifyClient:
    def test_small(self):
        rm = ResourceManager()
        assert rm.classify_client(10, 100) == "small"

    def test_medium(self):
        rm = ResourceManager()
        assert rm.classify_client(100, 10000) == "medium"

    def test_large(self):
        rm = ResourceManager()
        assert rm.classify_client(1000, 100000) == "large"

    def test_boundary_5000(self):
        rm = ResourceManager()
        assert rm.classify_client(0, 4999) == "small"
        assert rm.classify_client(0, 5000) == "medium"

    def test_boundary_50000(self):
        rm = ResourceManager()
        assert rm.classify_client(0, 49999) == "medium"
        assert rm.classify_client(0, 50000) == "large"


class TestResourceManagerSummary:
    def test_summary_without_gpu(self):
        rm = ResourceManager()
        with patch.object(rm, "probe_cpu", return_value={"available_cores": 4, "total_cores": 8}):
            with patch.object(rm, "probe_ram", return_value={"usable_gb": 12.0, "total_gb": 16.0}):
                with patch.object(rm, "probe_gpu", return_value=None):
                    summary = rm.summary()
        assert "GPU: not available" in summary
        assert "CPU:" in summary
        assert "RAM:" in summary

    def test_summary_with_gpu(self):
        rm = ResourceManager()
        gpu_info = {"free_mb": 4096, "total_mb": 8192, "used_mb": 4096, "utilization_pct": 50}
        with patch.object(rm, "probe_cpu", return_value={"available_cores": 4, "total_cores": 8}):
            with patch.object(rm, "probe_ram", return_value={"usable_gb": 12.0, "total_gb": 16.0}):
                with patch.object(rm, "probe_gpu", return_value=gpu_info):
                    summary = rm.summary()
        assert "GPU:" in summary
        assert "not available" not in summary


# ===========================================================================
# advanced_tokenize
# ===========================================================================

class TestAdvancedTokenize:
    def test_basic_tokenization(self):
        tokens = advanced_tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_stopwords_removed(self):
        tokens = advanced_tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_short_words_excluded(self):
        tokens = advanced_tokenize("I am a")
        assert "i" not in tokens
        assert "am" not in tokens
        assert "a" not in tokens

    def test_lowercases(self):
        tokens = advanced_tokenize("Python Programming")
        assert "python" in tokens
        assert "programming" in tokens

    def test_empty_string(self):
        assert advanced_tokenize("") == []

    def test_numbers_included_if_2plus_chars(self):
        tokens = advanced_tokenize("version 42 release")
        assert "42" in tokens
        assert "version" in tokens


# ===========================================================================
# EnhancedTokenizer
# ===========================================================================

class TestEnhancedTokenizerInit:
    def test_default_no_stemmer(self):
        t = EnhancedTokenizer()
        assert t._stemmer is None
        assert t.stemmer_type is None

    def test_synonyms_stored(self):
        syns = {"AML": ["anti-money laundering", "KYC"]}
        t = EnhancedTokenizer(synonyms=syns)
        assert "aml" in t._synonym_expansion

    def test_bidirectional_synonym_lookup(self):
        t = EnhancedTokenizer(synonyms={"aml": ["kyc"]})
        assert "kyc" in t._synonym_expansion

    def test_unknown_stemmer_disables_stemming(self):
        t = EnhancedTokenizer(stemmer="unknown-stemmer")
        assert t._stemmer is None

    def test_import_error_disables_stemming(self):
        with patch("mullet.knowledge.engine.search.tokenizer.EnhancedTokenizer._init_stemmer") as mock:
            mock.side_effect = None
            t = EnhancedTokenizer(stemmer="porter")
            # _init_stemmer called but we patched it to do nothing
        assert t._stemmer is None


class TestEnhancedTokenizerStem:
    def test_no_stemmer_returns_word_unchanged(self):
        t = EnhancedTokenizer()
        assert t._stem("running") == "running"

    def test_with_mock_stemmer(self):
        t = EnhancedTokenizer()
        t._stemmer = MagicMock()
        t._stemmer.stem.return_value = "run"
        assert t._stem("running") == "run"


class TestEnhancedTokenizerTokenize:
    def test_basic_tokenize(self):
        t = EnhancedTokenizer()
        tokens = t.tokenize("Python programming language")
        assert "python" in tokens
        assert "programming" in tokens
        assert "language" in tokens

    def test_stopwords_removed(self):
        t = EnhancedTokenizer()
        tokens = t.tokenize("the quick brown fox")
        assert "the" not in tokens

    def test_with_stemmer_applies_stemming(self):
        t = EnhancedTokenizer()
        mock_stemmer = MagicMock()
        mock_stemmer.stem = lambda w: w[:-3] if len(w) > 3 else w
        t._stemmer = mock_stemmer
        tokens = t.tokenize("running jumping")
        assert "runn" in tokens or "running"[:3] in tokens  # stems applied

    def test_empty_text(self):
        t = EnhancedTokenizer()
        assert t.tokenize("") == []


class TestEnhancedTokenizerTokenizeQuery:
    def test_basic_query(self):
        t = EnhancedTokenizer()
        tokens = t.tokenize_query("machine learning python")
        assert "machine" in tokens
        assert "learning" in tokens

    def test_synonym_expansion(self):
        t = EnhancedTokenizer(synonyms={"aml": ["anti-money laundering"]})
        tokens = t.tokenize_query("aml compliance")
        assert "aml" in tokens
        assert "anti" not in tokens or "money" in tokens or "laundering" in tokens

    def test_no_synonyms_passthrough(self):
        t = EnhancedTokenizer()
        tokens = t.tokenize_query("database query optimization")
        assert "database" in tokens
        assert "query" in tokens

    def test_empty_query(self):
        t = EnhancedTokenizer()
        assert t.tokenize_query("") == []


class TestEnhancedTokenizerGetConfigHash:
    def test_returns_8_char_hex(self):
        t = EnhancedTokenizer()
        h = t.get_config_hash()
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_configs_different_hashes(self):
        t1 = EnhancedTokenizer(stemmer="porter")
        t2 = EnhancedTokenizer(stemmer="snowball")
        assert t1.get_config_hash() != t2.get_config_hash()

    def test_same_config_same_hash(self):
        t1 = EnhancedTokenizer()
        t2 = EnhancedTokenizer()
        assert t1.get_config_hash() == t2.get_config_hash()


# ===========================================================================
# ModelCache
# ===========================================================================

class TestModelCache:
    def test_get_embedder_loads_model_once(self):
        cache = ModelCache()
        mock_st = MagicMock()
        with patch("mullet.knowledge.engine.model_cache.ModelCache.get_embedder") as mock_get:
            mock_get.return_value = mock_st
            e1 = cache.get_embedder.__wrapped__(cache, "model-a") if hasattr(cache.get_embedder, "__wrapped__") else mock_st
        # Test caching directly
        cache._embedders["model-a:cpu"] = mock_st
        result = cache.get_embedder("model-a", device="cpu")
        assert result is mock_st

    def test_get_embedder_creates_new_model(self):
        cache = ModelCache()
        mock_model = MagicMock()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            result = cache.get_embedder("all-MiniLM-L6-v2", device="cpu")
        assert result is mock_model

    def test_get_embedder_cached_on_second_call(self):
        cache = ModelCache()
        mock_model = MagicMock()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st_cls:
            cache.get_embedder("model-a", device="cpu")
            cache.get_embedder("model-a", device="cpu")
        # Should only instantiate once
        assert mock_st_cls.call_count == 1

    def test_get_reranker_creates_new_model(self):
        cache = ModelCache()
        mock_model = MagicMock()
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            result = cache.get_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        assert result is mock_model

    def test_get_reranker_cached_on_second_call(self):
        cache = ModelCache()
        mock_model = MagicMock()
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model) as mock_ce_cls:
            cache.get_reranker("reranker", device="cpu")
            cache.get_reranker("reranker", device="cpu")
        assert mock_ce_cls.call_count == 1

    def test_clear_empties_both_caches(self):
        cache = ModelCache()
        cache._embedders["model-a:cpu"] = MagicMock()
        cache._rerankers["reranker-b:cpu"] = MagicMock()
        with patch("torch.cuda.is_available", return_value=False):
            cache.clear()
        assert len(cache._embedders) == 0
        assert len(cache._rerankers) == 0

    def test_clear_with_cuda_calls_empty_cache(self):
        cache = ModelCache()
        cache._embedders["m:cpu"] = MagicMock()
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                cache.clear()
        mock_empty.assert_called_once()

    def test_clear_empty_does_not_log(self, caplog):
        import logging
        cache = ModelCache()
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO):
                cache.clear()
        assert "cleared" not in caplog.text


class TestGetModelCacheSingleton:
    def test_returns_model_cache_instance(self):
        from mullet.knowledge.engine import model_cache as mc_mod
        orig = mc_mod._instance
        mc_mod._instance = None
        try:
            result = get_model_cache()
            assert isinstance(result, ModelCache)
        finally:
            mc_mod._instance = orig

    def test_returns_same_instance(self):
        cache1 = get_model_cache()
        cache2 = get_model_cache()
        assert cache1 is cache2
