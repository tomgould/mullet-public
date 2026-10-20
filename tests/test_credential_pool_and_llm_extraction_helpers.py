"""Tests for credential_pool and llm_extraction pure helpers."""

import time

from mullet.core.providers.credential_pool import Credential, CredentialPool
from mullet.knowledge.engine.llm_extraction import (
    CitationMap,
    build_citation_map,
    parse_citations_from_response,
    build_rag_prompt,
    categorize_chunks,
    build_rag_prompt_with_citations,
)


# ---------------------------------------------------------------------------
# Credential
# ---------------------------------------------------------------------------

class TestCredential:
    def test_api_key_stored(self):
        c = Credential(api_key="sk-abc")
        assert c.api_key == "sk-abc"

    def test_is_available_default(self):
        c = Credential(api_key="key")
        assert c.is_available is True

    def test_exhausted_is_unavailable(self):
        c = Credential(api_key="key", exhausted_until=time.time() + 3600)
        assert c.is_available is False

    def test_expired_exhaustion_is_available(self):
        c = Credential(api_key="key", exhausted_until=time.time() - 1)
        assert c.is_available is True

    def test_cooldown_remaining_zero_when_not_exhausted(self):
        c = Credential(api_key="key")
        assert c.cooldown_remaining == 0

    def test_cooldown_remaining_positive_when_exhausted(self):
        c = Credential(api_key="key", exhausted_until=time.time() + 3600)
        assert c.cooldown_remaining > 0

    def test_defaults(self):
        c = Credential(api_key="key")
        assert c.total_requests == 0
        assert c.consecutive_failures == 0


# ---------------------------------------------------------------------------
# CredentialPool
# ---------------------------------------------------------------------------

class TestCredentialPool:
    def test_empty_pool_get_next_returns_none(self):
        pool = CredentialPool()
        assert pool.get_next() is None

    def test_add_and_get_next(self):
        pool = CredentialPool()
        pool.add("key1")
        cred = pool.get_next()
        assert cred is not None
        assert cred.api_key == "key1"

    def test_get_next_increments_request_count(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.get_next()
        assert pool._credentials[0].total_requests == 1

    def test_round_robin_cycles(self):
        pool = CredentialPool(strategy="round_robin")
        pool.add("key1")
        pool.add("key2")
        first = pool.get_next()
        second = pool.get_next()
        assert first.api_key != second.api_key

    def test_least_used_picks_least_requests(self):
        pool = CredentialPool(strategy="least_used")
        pool.add("key1")
        pool.add("key2")
        pool._credentials[0].total_requests = 10
        cred = pool.get_next()
        assert cred.api_key == "key2"

    def test_mark_exhausted_rate_limit(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1", error_code=429)
        assert not pool._credentials[0].is_available

    def test_mark_exhausted_increments_failures(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1")
        assert pool._credentials[0].consecutive_failures == 1

    def test_mark_success_resets_failures(self):
        pool = CredentialPool()
        pool.add("key1")
        pool._credentials[0].consecutive_failures = 3
        pool.mark_success("key1")
        assert pool._credentials[0].consecutive_failures == 0

    def test_mark_success_tracks_tokens(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_success("key1", tokens=1000)
        assert pool._credentials[0].total_tokens == 1000

    def test_available_count(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.add("key2")
        assert pool.available_count == 2

    def test_total_count(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.add("key2")
        assert pool.total_count == 2

    def test_status_returns_list(self):
        pool = CredentialPool()
        pool.add("key1")
        status = pool.status()
        assert isinstance(status, list)
        assert len(status) == 1

    def test_status_fields(self):
        pool = CredentialPool()
        pool.add("key1", name="my_key")
        status = pool.status()
        assert "name" in status[0]
        assert "available" in status[0]
        assert "requests" in status[0]

    def test_exhausted_key_not_returned(self):
        pool = CredentialPool()
        pool.add("key1")
        pool.mark_exhausted("key1")
        assert pool.get_next() is None


# ---------------------------------------------------------------------------
# CitationMap
# ---------------------------------------------------------------------------

class TestCitationMap:
    def test_get_valid_index(self):
        cm = CitationMap(chunks=[{"id": 1}, {"id": 2}])
        assert cm.get(1) == {"id": 1}
        assert cm.get(2) == {"id": 2}

    def test_get_zero_returns_none(self):
        cm = CitationMap(chunks=[{"id": 1}])
        assert cm.get(0) is None

    def test_get_out_of_range_returns_none(self):
        cm = CitationMap(chunks=[{"id": 1}])
        assert cm.get(5) is None

    def test_len(self):
        cm = CitationMap(chunks=[{"a": 1}, {"b": 2}])
        assert len(cm) == 2

    def test_empty_map(self):
        cm = CitationMap()
        assert len(cm) == 0
        assert cm.get(1) is None


# ---------------------------------------------------------------------------
# build_citation_map
# ---------------------------------------------------------------------------

class TestBuildCitationMap:
    def test_returns_citation_map(self):
        result = build_citation_map([{"id": 1}])
        assert isinstance(result, CitationMap)

    def test_chunks_stored(self):
        chunks = [{"id": 1}, {"id": 2}]
        result = build_citation_map(chunks)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# parse_citations_from_response
# ---------------------------------------------------------------------------

class TestParseCitationsFromResponse:
    def test_extracts_single(self):
        result = parse_citations_from_response("See [1] for details.")
        assert result == [1]

    def test_extracts_multiple(self):
        result = parse_citations_from_response("See [1] and [3] and [2].")
        assert set(result) == {1, 2, 3}

    def test_no_citations_returns_empty(self):
        result = parse_citations_from_response("No citations here.")
        assert result == []

    def test_returns_list_of_ints(self):
        result = parse_citations_from_response("[1][2]")
        assert all(isinstance(n, int) for n in result)


# ---------------------------------------------------------------------------
# categorize_chunks
# ---------------------------------------------------------------------------

class TestCategorizeChunks:
    def _make_chunk(self, chunk_type: str) -> dict:
        return {"content": "text", "metadata": {"chunk_type": chunk_type}}

    def test_code_categorized(self):
        chunks = [self._make_chunk("code")]
        result = categorize_chunks(chunks)
        assert len(result["code"]) == 1

    def test_documentation_categorized(self):
        chunks = [self._make_chunk("documentation")]
        result = categorize_chunks(chunks)
        assert len(result["documentation"]) == 1

    def test_unknown_type_goes_to_other(self):
        chunks = [self._make_chunk("unknown_type")]
        result = categorize_chunks(chunks)
        assert len(result["other"]) == 1

    def test_empty_chunks_all_zero(self):
        result = categorize_chunks([])
        assert all(len(v) == 0 for v in result.values())

    def test_returns_dict_with_expected_keys(self):
        result = categorize_chunks([])
        assert "code" in result
        assert "documentation" in result
        assert "conversation" in result
        assert "other" in result


# ---------------------------------------------------------------------------
# build_rag_prompt
# ---------------------------------------------------------------------------

class TestBuildRagPrompt:
    def _chunk(self, content: str, source: str = "doc.md") -> dict:
        return {"content": content, "metadata": {"source_file": source}}

    def test_question_in_prompt(self):
        result = build_rag_prompt("What is X?", [self._chunk("X is Y.")])
        assert "What is X?" in result

    def test_content_in_prompt(self):
        result = build_rag_prompt("q", [self._chunk("important content")])
        assert "important content" in result

    def test_empty_chunks_returns_prompt(self):
        result = build_rag_prompt("What is X?", [])
        assert "What is X?" in result

    def test_returns_string(self):
        assert isinstance(build_rag_prompt("q", []), str)


# ---------------------------------------------------------------------------
# build_rag_prompt_with_citations
# ---------------------------------------------------------------------------

class TestBuildRagPromptWithCitations:
    def _chunk(self, content: str) -> dict:
        return {"content": content, "metadata": {"source_file": "doc.md"}}

    def test_question_in_prompt(self):
        result = build_rag_prompt_with_citations("What?", [self._chunk("answer")])
        assert "What?" in result

    def test_cite_instructions_in_prompt(self):
        result = build_rag_prompt_with_citations("q", [self._chunk("c")])
        assert "citation" in result.lower() or "[1]" in result

    def test_returns_string(self):
        assert isinstance(build_rag_prompt_with_citations("q", []), str)
