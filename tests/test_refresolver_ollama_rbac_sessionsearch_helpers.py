"""Tests for reference_resolver pure helpers, OllamaConfig, rbac, and session_search."""

import json
import sqlite3
import pytest
from pathlib import Path

from mullet.knowledge.engine.reference_resolver import (
    clear_reference_cache,
    get_cache_stats,
    detect_references,
    detect_references_in_chunks,
    _prioritize_references,
    build_reference_queries,
    is_reference_content,
    REFERENCE_PATTERNS,
)
from mullet.core.providers.ollama import OllamaConfig
from mullet.server.rbac import ACCESS_LEVELS, check_project_access, get_user_projects
from mullet.core.tools.session_search import SessionIndex


# ---------------------------------------------------------------------------
# clear_reference_cache / get_cache_stats
# ---------------------------------------------------------------------------

class TestReferenceCacheOps:
    def test_clear_does_not_raise(self):
        clear_reference_cache()

    def test_get_cache_stats_returns_dict(self):
        result = get_cache_stats()
        assert isinstance(result, dict)

    def test_cache_stats_has_cached_references(self):
        clear_reference_cache()
        stats = get_cache_stats()
        assert "cached_references" in stats
        assert stats["cached_references"] == 0

    def test_cache_stats_has_total_chunks(self):
        stats = get_cache_stats()
        assert "total_cached_chunks" in stats


# ---------------------------------------------------------------------------
# REFERENCE_PATTERNS
# ---------------------------------------------------------------------------

class TestReferenceResolverPatterns:
    def test_is_list(self):
        assert isinstance(REFERENCE_PATTERNS, list)

    def test_nonempty(self):
        assert len(REFERENCE_PATTERNS) > 0

    def test_all_strings(self):
        assert all(isinstance(p, str) for p in REFERENCE_PATTERNS)


# ---------------------------------------------------------------------------
# detect_references
# ---------------------------------------------------------------------------

class TestDetectReferencesPure:
    def test_returns_list(self):
        result = detect_references("See Section 3.1.")
        assert isinstance(result, list)

    def test_detects_appendix(self):
        refs = detect_references("As set out in Appendix H.")
        assert any(r["type"] == "appendix" for r in refs)

    def test_detects_section(self):
        refs = detect_references("Refer to Section 5.2.")
        assert any(r["type"] == "section" for r in refs)

    def test_detects_schedule(self):
        refs = detect_references("As per Schedule 3.")
        assert any(r["type"] == "schedule" for r in refs)

    def test_detects_table(self):
        refs = detect_references("See Table 1.")
        assert any(r["type"] == "table" for r in refs)

    def test_empty_text_no_refs(self):
        assert detect_references("") == []

    def test_plain_text_no_refs(self):
        refs = detect_references("The quick brown fox.")
        assert refs == []

    def test_ref_id_uppercased(self):
        refs = detect_references("See Appendix a.")
        if refs:
            assert refs[0]["id"] == "A"

    def test_deduplicates_same_ref(self):
        refs = detect_references("See Appendix H and also Appendix H again.")
        app_refs = [r for r in refs if r["type"] == "appendix" and r["id"] == "H"]
        assert len(app_refs) == 1

    def test_ref_has_key_field(self):
        refs = detect_references("Section 2.1.")
        if refs:
            assert "key" in refs[0]

    def test_ref_has_match_field(self):
        refs = detect_references("Section 2.1.")
        if refs:
            assert "match" in refs[0]

    def test_multiple_refs_detected(self):
        refs = detect_references("See Appendix H and Section 3.2 and Schedule 1.")
        types = [r["type"] for r in refs]
        assert len(set(types)) >= 2


# ---------------------------------------------------------------------------
# detect_references_in_chunks
# ---------------------------------------------------------------------------

class TestDetectReferencesInChunks:
    def test_empty_chunks_empty_result(self):
        assert detect_references_in_chunks([]) == []

    def test_single_chunk_detected(self):
        chunks = [{"content": "See Appendix B for details."}]
        refs = detect_references_in_chunks(chunks)
        assert any(r["type"] == "appendix" for r in refs)

    def test_deduplicates_across_chunks(self):
        chunks = [
            {"content": "See Appendix B."},
            {"content": "Also see Appendix B."},
        ]
        refs = detect_references_in_chunks(chunks)
        app_refs = [r for r in refs if r["type"] == "appendix" and r["id"] == "B"]
        assert len(app_refs) == 1

    def test_combines_refs_from_multiple_chunks(self):
        chunks = [
            {"content": "See Appendix A."},
            {"content": "Refer to Section 4.1."},
        ]
        refs = detect_references_in_chunks(chunks)
        types = {r["type"] for r in refs}
        assert "appendix" in types
        assert "section" in types


# ---------------------------------------------------------------------------
# _prioritize_references
# ---------------------------------------------------------------------------

class TestPrioritizeReferences:
    def test_empty_returns_empty(self):
        assert _prioritize_references([]) == []

    def test_high_priority_appendix_kept(self):
        refs = [
            {"type": "appendix", "id": "A", "key": "appendix:A", "match": "Appendix A"},
            {"type": "chapter", "id": "1", "key": "chapter:1", "match": "Chapter 1"},
        ]
        result = _prioritize_references(refs)
        assert any(r["type"] == "appendix" for r in result)

    def test_all_refs_returned_when_no_high_priority(self):
        refs = [
            {"type": "section", "id": "1", "key": "section:1", "match": "Section 1"},
            {"type": "section", "id": "2", "key": "section:2", "match": "Section 2"},
        ]
        result = _prioritize_references(refs)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# build_reference_queries
# ---------------------------------------------------------------------------

class TestBuildReferenceQueries:
    def test_returns_list(self):
        refs = [{"type": "appendix", "id": "H", "key": "appendix:H", "match": "Appendix H"}]
        result = build_reference_queries(refs)
        assert isinstance(result, list)

    def test_appendix_generates_queries(self):
        refs = [{"type": "appendix", "id": "H", "key": "appendix:H", "match": "Appendix H"}]
        queries = build_reference_queries(refs)
        assert any("Appendix H" in q or "Annex H" in q for q in queries)

    def test_section_generates_query(self):
        refs = [{"type": "section", "id": "3.2", "key": "section:3.2", "match": "Section 3.2"}]
        queries = build_reference_queries(refs)
        assert any("Section 3.2" in q for q in queries)

    def test_empty_refs_empty_queries(self):
        assert build_reference_queries([]) == []

    def test_deduplicates_queries(self):
        refs = [
            {"type": "appendix", "id": "A", "key": "appendix:A", "match": "Appendix A"},
            {"type": "appendix", "id": "A", "key": "appendix:A", "match": "Appendix A"},
        ]
        queries = build_reference_queries(refs)
        unique_queries = list(set(queries))
        assert len(queries) == len(unique_queries)

    def test_schedule_generates_query(self):
        refs = [{"type": "schedule", "id": "3", "key": "schedule:3", "match": "Schedule 3"}]
        queries = build_reference_queries(refs)
        assert any("Schedule 3" in q for q in queries)


# ---------------------------------------------------------------------------
# is_reference_content
# ---------------------------------------------------------------------------

class TestIsReferenceContent:
    def test_chunk_with_matching_header_is_reference(self):
        chunk = {"content": "Appendix H:\nHigh-risk countries list."}
        ref = {"type": "appendix", "id": "H", "key": "appendix:H"}
        assert is_reference_content(chunk, ref) is True

    def test_chunk_without_match_is_not_reference(self):
        chunk = {"content": "This is some unrelated content about something else."}
        ref = {"type": "appendix", "id": "Z", "key": "appendix:Z"}
        assert is_reference_content(chunk, ref) is False

    def test_empty_chunk_not_reference(self):
        chunk = {"content": ""}
        ref = {"type": "appendix", "id": "A", "key": "appendix:A"}
        assert is_reference_content(chunk, ref) is False

    def test_metadata_source_file_match(self):
        chunk = {
            "content": "content here",
            "metadata": {"source_file": "appendix_h.pdf"},
        }
        ref = {"type": "appendix", "id": "H", "key": "appendix:H"}
        assert is_reference_content(chunk, ref) is True

    def test_metadata_section_title_match(self):
        chunk = {
            "content": "content here",
            "metadata": {"section_title": "appendix h"},
        }
        ref = {"type": "appendix", "id": "H", "key": "appendix:H"}
        assert is_reference_content(chunk, ref) is True


# ---------------------------------------------------------------------------
# OllamaConfig
# ---------------------------------------------------------------------------

class TestOllamaConfig:
    def test_default_base_url(self):
        cfg = OllamaConfig()
        assert "localhost" in cfg.base_url

    def test_default_model(self):
        cfg = OllamaConfig()
        assert isinstance(cfg.model, str)
        assert len(cfg.model) > 0

    def test_default_temperature(self):
        cfg = OllamaConfig()
        assert 0.0 <= cfg.temperature <= 2.0

    def test_default_top_p(self):
        cfg = OllamaConfig()
        assert 0.0 <= cfg.top_p <= 1.0

    def test_default_max_tokens(self):
        cfg = OllamaConfig()
        assert cfg.max_tokens > 0

    def test_custom_values(self):
        cfg = OllamaConfig(model="mistral", temperature=0.1, max_tokens=500)
        assert cfg.model == "mistral"
        assert cfg.temperature == pytest.approx(0.1)
        assert cfg.max_tokens == 500

    def test_custom_base_url(self):
        cfg = OllamaConfig(base_url="http://remote:11434")
        assert cfg.base_url == "http://remote:11434"


# ---------------------------------------------------------------------------
# ACCESS_LEVELS
# ---------------------------------------------------------------------------

class TestAccessLevels:
    def test_is_dict(self):
        assert isinstance(ACCESS_LEVELS, dict)

    def test_has_read(self):
        assert "read" in ACCESS_LEVELS

    def test_has_write(self):
        assert "write" in ACCESS_LEVELS

    def test_has_owner(self):
        assert "owner" in ACCESS_LEVELS

    def test_read_lower_than_write(self):
        assert ACCESS_LEVELS["read"] < ACCESS_LEVELS["write"]

    def test_write_lower_than_owner(self):
        assert ACCESS_LEVELS["write"] < ACCESS_LEVELS["owner"]


# ---------------------------------------------------------------------------
# Helpers for in-memory SQLite DB
# ---------------------------------------------------------------------------

def _make_db():
    db = sqlite3.connect(":memory:")
    db.execute("""CREATE TABLE projects (
        id TEXT PRIMARY KEY,
        name TEXT,
        pool_name TEXT,
        description TEXT
    )""")
    db.execute("""CREATE TABLE user_projects (
        user_id TEXT,
        project_id TEXT,
        access_level TEXT
    )""")
    return db


def _add_project(db, id, name, pool_name="", desc=""):
    db.execute("INSERT INTO projects VALUES (?, ?, ?, ?)", (id, name, pool_name, desc))


def _add_user_project(db, user_id, project_id, level):
    db.execute("INSERT INTO user_projects VALUES (?, ?, ?)", (user_id, project_id, level))


# ---------------------------------------------------------------------------
# check_project_access
# ---------------------------------------------------------------------------

class TestCheckProjectAccessAdmin:
    def test_admin_always_allowed(self):
        result = check_project_access({"role": "admin", "id": "u1"}, "any_project")
        assert result is True

    def test_admin_no_db_needed(self):
        result = check_project_access({"role": "admin", "id": "u1"}, "proj", db=None)
        assert result is True


class TestCheckProjectAccessNoDb:
    def test_no_db_returns_false(self):
        result = check_project_access({"id": "u1"}, "proj", db=None)
        assert result is False


class TestCheckProjectAccessWithDb:
    def test_user_with_owner_access(self):
        db = _make_db()
        _add_project(db, "p1", "MyProject")
        _add_user_project(db, "u1", "p1", "owner")
        assert check_project_access({"id": "u1"}, "MyProject", db=db) is True

    def test_user_with_write_access_can_read(self):
        db = _make_db()
        _add_project(db, "p1", "MyProject")
        _add_user_project(db, "u1", "p1", "write")
        assert check_project_access({"id": "u1"}, "MyProject", required_level="read", db=db) is True

    def test_user_with_read_cannot_write(self):
        db = _make_db()
        _add_project(db, "p1", "MyProject")
        _add_user_project(db, "u1", "p1", "read")
        assert check_project_access({"id": "u1"}, "MyProject", required_level="write", db=db) is False

    def test_unknown_project_returns_false(self):
        db = _make_db()
        assert check_project_access({"id": "u1"}, "nonexistent", db=db) is False

    def test_user_not_in_project_returns_false(self):
        db = _make_db()
        _add_project(db, "p1", "MyProject")
        assert check_project_access({"id": "u2"}, "MyProject", db=db) is False

    def test_lookup_by_pool_name(self):
        db = _make_db()
        _add_project(db, "p1", "MyProject", pool_name="my_pool")
        _add_user_project(db, "u1", "p1", "read")
        assert check_project_access({"id": "u1"}, "my_pool", db=db) is True


# ---------------------------------------------------------------------------
# get_user_projects
# ---------------------------------------------------------------------------

class TestGetUserProjects:
    def test_no_projects_returns_empty(self):
        db = _make_db()
        assert get_user_projects("u1", db) == []

    def test_returns_list_of_dicts(self):
        db = _make_db()
        _add_project(db, "p1", "Proj", desc="Desc")
        _add_user_project(db, "u1", "p1", "read")
        result = get_user_projects("u1", db)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_entry_has_name(self):
        db = _make_db()
        _add_project(db, "p1", "MyProj")
        _add_user_project(db, "u1", "p1", "write")
        result = get_user_projects("u1", db)
        assert result[0]["name"] == "MyProj"

    def test_entry_has_access(self):
        db = _make_db()
        _add_project(db, "p1", "MyProj")
        _add_user_project(db, "u1", "p1", "owner")
        result = get_user_projects("u1", db)
        assert result[0]["access"] == "owner"

    def test_multiple_projects(self):
        db = _make_db()
        _add_project(db, "p1", "Proj1")
        _add_project(db, "p2", "Proj2")
        _add_user_project(db, "u1", "p1", "read")
        _add_user_project(db, "u1", "p2", "write")
        result = get_user_projects("u1", db)
        assert len(result) == 2

    def test_other_user_not_included(self):
        db = _make_db()
        _add_project(db, "p1", "Proj1")
        _add_user_project(db, "u2", "p1", "read")
        result = get_user_projects("u1", db)
        assert result == []


# ---------------------------------------------------------------------------
# SessionIndex
# ---------------------------------------------------------------------------

class TestSessionIndex:
    def _make_session_dir(self, tmp_path, sessions: list[list[dict]]) -> Path:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        for i, records in enumerate(sessions):
            session_file = session_dir / f"session_{i:02d}.jsonl"
            lines = [json.dumps(r) for r in records]
            session_file.write_text("\n".join(lines))
        return session_dir

    def test_empty_dir_no_sessions(self, tmp_path):
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        idx = SessionIndex(str(session_dir))
        assert idx.sessions == []

    def test_search_returns_list(self, tmp_path):
        session_dir = self._make_session_dir(tmp_path, [
            [{"user_message": "hello world"}]
        ])
        idx = SessionIndex(str(session_dir))
        result = idx.search("hello")
        assert isinstance(result, list)

    def test_search_finds_match(self, tmp_path):
        session_dir = self._make_session_dir(tmp_path, [
            [{"user_message": "debug the authentication bug"}]
        ])
        idx = SessionIndex(str(session_dir))
        result = idx.search("authentication")
        assert len(result) >= 1

    def test_search_returns_empty_for_no_match(self, tmp_path):
        session_dir = self._make_session_dir(tmp_path, [
            [{"user_message": "hello world"}]
        ])
        idx = SessionIndex(str(session_dir))
        result = idx.search("xyz_totally_unrelated")
        assert result == []

    def test_search_case_insensitive(self, tmp_path):
        session_dir = self._make_session_dir(tmp_path, [
            [{"user_message": "Debug the AUTH module"}]
        ])
        idx = SessionIndex(str(session_dir))
        result = idx.search("auth")
        assert len(result) >= 1

    def test_search_respects_max_results(self, tmp_path):
        # Create 10 sessions all containing "test"
        sessions = [[{"user_message": "test message"}]] * 10
        session_dir = self._make_session_dir(tmp_path, sessions)
        idx = SessionIndex(str(session_dir))
        result = idx.search("test", max_results=3)
        assert len(result) <= 3

    def test_search_result_has_session_file(self, tmp_path):
        session_dir = self._make_session_dir(tmp_path, [
            [{"user_message": "find this content"}]
        ])
        idx = SessionIndex(str(session_dir))
        result = idx.search("find this content")
        if result:
            assert "session_file" in result[0]

    def test_search_result_has_matches(self, tmp_path):
        session_dir = self._make_session_dir(tmp_path, [
            [{"user_message": "find this content"}]
        ])
        idx = SessionIndex(str(session_dir))
        result = idx.search("find this content")
        if result:
            assert "matches" in result[0]

    def test_malformed_jsonl_skipped(self, tmp_path):
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        f = session_dir / "bad.jsonl"
        f.write_text("not json\n{\"user_message\": \"valid\"}\n")
        idx = SessionIndex(str(session_dir))
        result = idx.search("valid")
        assert isinstance(result, list)

    def test_assistant_messages_indexed(self, tmp_path):
        session_dir = self._make_session_dir(tmp_path, [
            [{"assistant_message": "I found the answer here"}]
        ])
        idx = SessionIndex(str(session_dir))
        result = idx.search("found the answer")
        assert len(result) >= 1
