"""Tests for mullet.knowledge.engine.search.tokenizer."""
from __future__ import annotations

from mullet.knowledge.engine.search.tokenizer import (
    STOP_WORDS,
    TOKENIZER_VERSION,
    EnhancedTokenizer,
    advanced_tokenize,
)


# ---------------------------------------------------------------------------
# advanced_tokenize
# ---------------------------------------------------------------------------

class TestAdvancedTokenize:
    def test_basic_words_returned(self) -> None:
        result = advanced_tokenize("authentication and authorization")
        assert "authentication" in result
        assert "authorization" in result

    def test_stopwords_removed(self) -> None:
        result = advanced_tokenize("the user and the password")
        assert "the" not in result
        assert "and" not in result

    def test_single_char_words_excluded(self) -> None:
        result = advanced_tokenize("a b c do it")
        assert "a" not in result
        assert "b" not in result
        assert "c" not in result

    def test_lowercases_output(self) -> None:
        result = advanced_tokenize("Django REST Framework")
        assert all(t == t.lower() for t in result)
        assert "django" in result

    def test_empty_string_returns_empty_list(self) -> None:
        assert advanced_tokenize("") == []

    def test_only_stopwords_returns_empty(self) -> None:
        result = advanced_tokenize("the and or but if")
        assert result == []

    def test_numbers_included(self) -> None:
        result = advanced_tokenize("version 42 of the API")
        assert "42" in result
        assert "version" in result

    def test_punctuation_stripped(self) -> None:
        result = advanced_tokenize("Hello, world! How are you?")
        assert "hello" in result
        assert "world" in result
        # stopwords filtered
        assert "are" not in result

    def test_two_char_words_included(self) -> None:
        result = advanced_tokenize("go to it")
        # 'go' is 2 chars and not a stopword
        assert "go" in result

    def test_returns_list(self) -> None:
        assert isinstance(advanced_tokenize("test input"), list)

    def test_stop_words_constant_non_empty(self) -> None:
        assert len(STOP_WORDS) > 50

    def test_tokenizer_version_is_int(self) -> None:
        assert isinstance(TOKENIZER_VERSION, int)
        assert TOKENIZER_VERSION >= 1


# ---------------------------------------------------------------------------
# EnhancedTokenizer — no stemming
# ---------------------------------------------------------------------------

class TestEnhancedTokenizerNoStemming:
    def test_tokenize_basic(self) -> None:
        tok = EnhancedTokenizer()
        result = tok.tokenize("user authentication system")
        assert "user" in result
        assert "authentication" in result
        assert "system" in result

    def test_tokenize_removes_stopwords(self) -> None:
        tok = EnhancedTokenizer()
        result = tok.tokenize("the quick brown fox")
        assert "the" not in result

    def test_tokenize_query_no_expansion(self) -> None:
        tok = EnhancedTokenizer()
        result = tok.tokenize_query("authentication module")
        assert "authentication" in result
        assert "module" in result

    def test_tokenize_query_empty(self) -> None:
        tok = EnhancedTokenizer()
        assert tok.tokenize_query("") == []

    def test_get_config_hash_returns_string(self) -> None:
        tok = EnhancedTokenizer()
        h = tok.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 8

    def test_config_hash_changes_with_synonyms(self) -> None:
        tok1 = EnhancedTokenizer()
        tok2 = EnhancedTokenizer(synonyms={"aml": ["anti money laundering"]})
        assert tok1.get_config_hash() != tok2.get_config_hash()

    def test_config_hash_stable(self) -> None:
        tok = EnhancedTokenizer()
        assert tok.get_config_hash() == tok.get_config_hash()


# ---------------------------------------------------------------------------
# EnhancedTokenizer — synonym expansion
# ---------------------------------------------------------------------------

class TestEnhancedTokenizerSynonyms:
    def test_canonical_expands_to_synonym(self) -> None:
        tok = EnhancedTokenizer(synonyms={"aml": ["anti money laundering"]})
        result = tok.tokenize_query("aml compliance")
        # "aml" should expand; "anti", "laundering" included; "money" is stopword
        assert "aml" in result
        assert "laundering" in result
        assert "compliance" in result

    def test_synonym_maps_back_to_canonical(self) -> None:
        tok = EnhancedTokenizer(synonyms={"kyc": ["know your customer"]})
        # Querying with "know" won't trigger (it's just a word), but "kyc" should expand
        result = tok.tokenize_query("kyc checks")
        assert "kyc" in result

    def test_multi_word_synonym_split(self) -> None:
        tok = EnhancedTokenizer(synonyms={"cdd": ["customer due diligence"]})
        result = tok.tokenize_query("cdd requirements")
        assert "cdd" in result
        assert "customer" in result
        assert "diligence" in result

    def test_non_synonym_word_passes_through(self) -> None:
        tok = EnhancedTokenizer(synonyms={"aml": ["anti money laundering"]})
        result = tok.tokenize_query("database migration")
        assert "database" in result
        assert "migration" in result

    def test_empty_synonyms_dict(self) -> None:
        tok = EnhancedTokenizer(synonyms={})
        result = tok.tokenize_query("test query words")
        assert "test" in result
        assert "query" in result


# ---------------------------------------------------------------------------
# EnhancedTokenizer — invalid stemmer falls back gracefully
# ---------------------------------------------------------------------------

class TestEnhancedTokenizerBadStemmer:
    def test_unknown_stemmer_type_falls_back(self) -> None:
        tok = EnhancedTokenizer(stemmer="nonexistent_stemmer")
        # Should not raise; stemmer should be None
        assert tok._stemmer is None

    def test_tokenize_works_after_bad_stemmer(self) -> None:
        tok = EnhancedTokenizer(stemmer="nonexistent_stemmer")
        result = tok.tokenize("database connection pooling")
        assert "database" in result
        assert "connection" in result
        assert "pooling" in result
