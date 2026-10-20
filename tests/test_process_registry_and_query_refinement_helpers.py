"""Tests for ProcessRegistry and QueryRefiner pure helpers."""

from mullet.core.tools.process_registry import ProcessRegistry
from mullet.core.knowledge.query_refinement import QueryRefiner


class TestProcessRegistryInit:
    def test_list_active_empty_initially(self):
        assert ProcessRegistry().list_active() == []

    def test_returns_list(self):
        assert isinstance(ProcessRegistry().list_active(), list)


class TestProcessRegistryRegister:
    def test_register_adds_to_active(self):
        pr = ProcessRegistry()
        pr.register(1234, "worker-1")
        assert len(pr.list_active()) == 1

    def test_registered_pid_in_active(self):
        pr = ProcessRegistry()
        pr.register(1234, "worker-1")
        assert 1234 in [item[0] for item in pr.list_active()]

    def test_registered_name_in_active(self):
        pr = ProcessRegistry()
        pr.register(1234, "worker-1")
        assert "worker-1" in [item[1] for item in pr.list_active()]

    def test_multiple_registrations(self):
        pr = ProcessRegistry()
        pr.register(1234, "a")
        pr.register(5678, "b")
        assert len(pr.list_active()) == 2

    def test_register_overwrites_same_pid(self):
        pr = ProcessRegistry()
        pr.register(1234, "old")
        pr.register(1234, "new")
        names = [item[1] for item in pr.list_active()]
        assert "new" in names
        assert len(pr.list_active()) == 1


class TestProcessRegistryUnregister:
    def test_unregister_removes_process(self):
        pr = ProcessRegistry()
        pr.register(1234, "worker")
        pr.unregister(1234)
        assert len(pr.list_active()) == 0

    def test_unregister_unknown_pid_no_error(self):
        pr = ProcessRegistry()
        pr.unregister(9999)

    def test_unregister_only_removes_specified(self):
        pr = ProcessRegistry()
        pr.register(1234, "a")
        pr.register(5678, "b")
        pr.unregister(1234)
        pids = [item[0] for item in pr.list_active()]
        assert 5678 in pids
        assert 1234 not in pids


class TestProcessRegistryCleanup:
    def test_cleanup_empties_all(self):
        pr = ProcessRegistry()
        pr.register(1, "a")
        pr.register(2, "b")
        pr.cleanup_all()
        assert pr.list_active() == []

    def test_cleanup_empty_registry_no_error(self):
        ProcessRegistry().cleanup_all()

    def test_can_register_after_cleanup(self):
        pr = ProcessRegistry()
        pr.register(1, "a")
        pr.cleanup_all()
        pr.register(2, "b")
        assert len(pr.list_active()) == 1


class TestQueryRefinerInit:
    def test_thesaurus_nonempty(self):
        assert len(QueryRefiner().thesaurus) > 0

    def test_search_in_thesaurus(self):
        assert "search" in QueryRefiner().thesaurus


class TestExpandQuery:
    def test_returns_string(self):
        assert isinstance(QueryRefiner().expand_query("search"), str)

    def test_known_word_expanded(self):
        result = QueryRefiner().expand_query("search")
        assert any(w in result for w in ("find", "lookup", "discover", "retrieve"))

    def test_original_word_preserved(self):
        assert "search" in QueryRefiner().expand_query("search")

    def test_unknown_word_unchanged(self):
        assert "xyzzy" in QueryRefiner().expand_query("xyzzy")

    def test_multiple_known_words_expanded(self):
        result = QueryRefiner().expand_query("search document")
        assert any(w in result for w in ("find", "discover"))
        assert any(w in result for w in ("file", "record"))

    def test_empty_query_returns_string(self):
        assert isinstance(QueryRefiner().expand_query(""), str)

    def test_result_is_sorted(self):
        words = QueryRefiner().expand_query("search").split()
        assert words == sorted(words)


class TestRefine:
    def test_refine_returns_string(self):
        assert isinstance(QueryRefiner().refine("search document"), str)

    def test_refine_expands_query(self):
        result = QueryRefiner().refine("search")
        assert any(w in result for w in ("find", "discover"))

    def test_refine_with_context_same_as_without(self):
        qr = QueryRefiner()
        assert qr.refine("search") == qr.refine("search", context="ctx")

    def test_refine_unknown_word(self):
        assert "xyzzy" in QueryRefiner().refine("xyzzy")
