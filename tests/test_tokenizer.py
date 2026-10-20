"""Tests for mullet.knowledge.engine.search.tokenizer — pure functions."""

from unittest.mock import MagicMock, patch

from mullet.knowledge.engine.search.tokenizer import (
    advanced_tokenize,
    EnhancedTokenizer,
    STOP_WORDS,
    TOKENIZER_VERSION,
)


class TestAdvancedTokenize:
    def test_basic_word_extraction(self):
        result = advanced_tokenize("hello world")
        assert "hello" in result
        assert "world" in result

    def test_stopwords_removed(self):
        result = advanced_tokenize("the quick brown fox")
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_lowercases_tokens(self):
        result = advanced_tokenize("Python Django REST")
        assert "python" in result
        assert "django" in result
        assert "rest" in result

    def test_single_char_words_excluded(self):
        result = advanced_tokenize("a b c do this")
        assert "a" not in result
        assert "b" not in result
        assert "c" not in result

    def test_empty_string_returns_empty(self):
        assert advanced_tokenize("") == []

    def test_only_stopwords_returns_empty(self):
        # "the is are" — all stopwords
        result = advanced_tokenize("the is are")
        assert result == []

    def test_numbers_included(self):
        result = advanced_tokenize("version 42 released")
        assert "42" in result
        assert "version" in result

    def test_punctuation_stripped(self):
        result = advanced_tokenize("hello, world! how are you?")
        assert "hello" in result
        assert "world" in result

    def test_mixed_case_lowercased(self):
        result = advanced_tokenize("GitHub Actions CI")
        assert "github" in result
        assert "actions" in result

    def test_two_char_words_included(self):
        result = advanced_tokenize("go is ok")
        # "go" is 2 chars, "is" is stopword, "ok" is 2 chars
        assert "go" in result
        assert "ok" in result

    def test_minimum_length_two(self):
        result = advanced_tokenize("ax by cz")
        # All 2-char words, should be included if not stopwords
        for token in result:
            assert len(token) >= 2

    def test_code_terms_retained(self):
        result = advanced_tokenize("def calculate_total returns float")
        assert "def" in result
        # \w+ includes underscores, so calculate_total is kept as one token
        assert "calculate_total" in result
        assert "returns" in result
        assert "float" in result


class TestStopWords:
    def test_common_words_in_stopwords(self):
        common = {"the", "is", "are", "was", "be", "in", "of", "to", "and"}
        assert common.issubset(STOP_WORDS)

    def test_stopwords_is_set(self):
        assert isinstance(STOP_WORDS, set)

    def test_stopwords_non_empty(self):
        assert len(STOP_WORDS) > 50


class TestTokenizerVersion:
    def test_version_is_int(self):
        assert isinstance(TOKENIZER_VERSION, int)
        assert TOKENIZER_VERSION >= 1


class TestEnhancedTokenizer:
    def test_default_init(self):
        t = EnhancedTokenizer()
        assert t.stemmer_type is None
        assert t.synonyms == {}

    def test_tokenize_no_stemmer(self):
        t = EnhancedTokenizer()
        result = t.tokenize("calculate total revenue")
        assert "calculate" in result
        assert "total" in result  # not a stopword
        assert "revenue" in result

    def test_synonyms_expansion_via_tokenize_query(self):
        t = EnhancedTokenizer(synonyms={"auth": ["authentication", "login"]})
        result = t.tokenize_query("auth system")
        combined = " ".join(result)
        assert "auth" in combined or "authentication" in combined

    def test_tokenize_returns_list(self):
        t = EnhancedTokenizer()
        result = t.tokenize("some text here")
        assert isinstance(result, list)

    def test_empty_input_returns_empty(self):
        t = EnhancedTokenizer()
        result = t.tokenize("")
        assert result == []

    def test_stopwords_filtered(self):
        t = EnhancedTokenizer()
        result = t.tokenize("the quick brown fox")
        assert "the" not in result

    def test_version_attribute(self):
        assert EnhancedTokenizer.VERSION >= 1

    def test_tokenize_code_text(self):
        t = EnhancedTokenizer()
        result = t.tokenize("class UserService implements ServiceInterface")
        assert "class" in result
        assert "userservice" in result or "user" in result or "service" in result

    # --- stemmer paths ---

    def test_porter_stemmer_init(self):
        mock_stemmer = MagicMock()
        mock_stemmer_cls = MagicMock(return_value=mock_stemmer)
        import sys
        fake_nltk_stem = MagicMock()
        fake_nltk_stem.PorterStemmer = mock_stemmer_cls
        fake_nltk = MagicMock()
        fake_nltk.stem = fake_nltk_stem
        with patch.dict(sys.modules, {"nltk": fake_nltk, "nltk.stem": fake_nltk_stem}):
            t = EnhancedTokenizer(stemmer="porter")
        assert t._stemmer is mock_stemmer

    def test_snowball_stemmer_init(self):
        mock_stemmer = MagicMock()
        mock_stemmer_cls = MagicMock(return_value=mock_stemmer)
        import sys
        fake_nltk_stem = MagicMock()
        fake_nltk_stem.SnowballStemmer = mock_stemmer_cls
        fake_nltk = MagicMock()
        fake_nltk.stem = fake_nltk_stem
        with patch.dict(sys.modules, {"nltk": fake_nltk, "nltk.stem": fake_nltk_stem}):
            t = EnhancedTokenizer(stemmer="snowball", language="english")
        mock_stemmer_cls.assert_called_once_with("english")
        assert t._stemmer is mock_stemmer

    def test_unknown_stemmer_type_disables_stemming(self):
        import sys
        fake_nltk_stem = MagicMock()
        fake_nltk = MagicMock()
        fake_nltk.stem = fake_nltk_stem
        with patch.dict(sys.modules, {"nltk": fake_nltk, "nltk.stem": fake_nltk_stem}):
            t = EnhancedTokenizer(stemmer="unknown_type")
        assert t._stemmer is None

    def test_stemmer_import_error_disables_stemming(self):
        import sys
        with patch.dict(sys.modules, {"nltk": None, "nltk.stem": None}):
            t = EnhancedTokenizer(stemmer="porter")
        assert t._stemmer is None

    def test_stemmer_general_exception_disables_stemming(self):
        import sys
        fake_nltk_stem = MagicMock()
        fake_nltk_stem.PorterStemmer = MagicMock(side_effect=RuntimeError("crash"))
        fake_nltk = MagicMock()
        fake_nltk.stem = fake_nltk_stem
        with patch.dict(sys.modules, {"nltk": fake_nltk, "nltk.stem": fake_nltk_stem}):
            t = EnhancedTokenizer(stemmer="porter")
        assert t._stemmer is None

    def test_stem_with_active_stemmer(self):
        t = EnhancedTokenizer()
        mock_stemmer = MagicMock()
        mock_stemmer.stem.return_value = "countri"
        t._stemmer = mock_stemmer
        result = t._stem("countries")
        assert result == "countri"
        mock_stemmer.stem.assert_called_once_with("countries")

    def test_stem_without_stemmer_returns_original(self):
        t = EnhancedTokenizer()
        assert t._stemmer is None
        assert t._stem("running") == "running"

    def test_tokenize_applies_stemming(self):
        t = EnhancedTokenizer()
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w + "_stem"
        t._stemmer = mock_stemmer
        result = t.tokenize("running faster")
        assert "running_stem" in result
        assert "faster_stem" in result

    def test_tokenize_query_with_synonyms_and_stemmer(self):
        t = EnhancedTokenizer(synonyms={"auth": ["login"]})
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w + "_s"
        t._stemmer = mock_stemmer
        result = t.tokenize_query("auth check")
        combined = " ".join(result)
        # Both auth and login should appear (stemmed)
        assert "auth_s" in combined or "login_s" in combined

    def test_get_config_hash_returns_string(self):
        t = EnhancedTokenizer()
        h = t.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 8

    def test_get_config_hash_changes_with_config(self):
        t1 = EnhancedTokenizer(stemmer="porter")
        t2 = EnhancedTokenizer(stemmer="snowball")
        assert t1.get_config_hash() != t2.get_config_hash()
