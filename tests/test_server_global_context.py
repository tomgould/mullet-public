"""Unit tests for mullet/server/global_context.py."""

import json
import sys
import time
from unittest.mock import MagicMock

import pytest

from mullet.server.global_context import GlobalContextManager, GlobalSource


# ---------------------------------------------------------------------------
# GlobalSource dataclass
# ---------------------------------------------------------------------------


class TestGlobalSource:
    def test_defaults(self):
        s = GlobalSource(id="gs_1", type="github_org")
        assert s.id == "gs_1"
        assert s.type == "github_org"
        assert s.config == {}
        assert s.schedule == "0 2 * * *"
        assert s.last_sync == 0
        assert s.status == "pending"
        assert s.error_message == ""

    def test_custom_values(self):
        s = GlobalSource(
            id="gs_2",
            type="web_crawl",
            config={"urls": ["https://example.com"]},
            schedule="0 0 * * 0",
            last_sync=12345.0,
            status="ready",
            error_message="oops",
        )
        assert s.config == {"urls": ["https://example.com"]}
        assert s.schedule == "0 0 * * 0"
        assert s.last_sync == 12345.0
        assert s.status == "ready"
        assert s.error_message == "oops"

    def test_config_is_isolated_per_instance(self):
        s1 = GlobalSource(id="a", type="t")
        s2 = GlobalSource(id="b", type="t")
        s1.config["x"] = 1
        assert "x" not in s2.config

    def test_field_types(self):
        s = GlobalSource(id="x", type="y")
        assert isinstance(s.config, dict)
        assert isinstance(s.schedule, str)
        assert isinstance(s.status, str)
        assert isinstance(s.error_message, str)


# ---------------------------------------------------------------------------
# add_source
# ---------------------------------------------------------------------------


class TestAddSource:
    def test_no_db_creates_source_adds_to_sources_returns_it(self):
        mgr = GlobalContextManager()
        source = mgr.add_source("github_org", {"org": "acme"})
        assert source.id in mgr._sources
        assert mgr._sources[source.id] is source
        assert source.type == "github_org"
        assert source.config == {"org": "acme"}
        assert source.schedule == "0 2 * * *"

    def test_with_db_calls_save_to_db(self, monkeypatch):
        mgr = GlobalContextManager(db=MagicMock())
        saved = []
        monkeypatch.setattr(mgr, "_save_to_db", lambda s: saved.append(s))
        source = mgr.add_source("confluence", {"url": "https://wiki"})
        assert len(saved) == 1
        assert saved[0] is source

    def test_source_id_format(self, monkeypatch):
        mgr = GlobalContextManager()
        monkeypatch.setattr(time, "time", lambda: 1234567890)
        source = mgr.add_source("jira", {"url": "https://jira"})
        assert source.id == "gs_1234567890_jira"

    def test_custom_schedule(self):
        mgr = GlobalContextManager()
        source = mgr.add_source("web_crawl", {"urls": []}, schedule="0 4 * * *")
        assert source.schedule == "0 4 * * *"


# ---------------------------------------------------------------------------
# remove_source
# ---------------------------------------------------------------------------


class TestRemoveSource:
    def test_removes_existing_source(self):
        mgr = GlobalContextManager()
        s = mgr.add_source("github_org", {"org": "acme"})
        mgr.remove_source(s.id)
        assert s.id not in mgr._sources

    def test_missing_source_id_no_error(self):
        mgr = GlobalContextManager()
        mgr.remove_source("nonexistent")
        assert "nonexistent" not in mgr._sources

    def test_with_db_executes_delete_and_commit(self):
        db = MagicMock()
        mgr = GlobalContextManager(db=db)
        s = mgr.add_source("github_org", {"org": "acme"})
        db.reset_mock()
        mgr.remove_source(s.id)
        db.execute.assert_called_once_with(
            "DELETE FROM global_sources WHERE id = ?", (s.id,)
        )
        db.commit.assert_called_once()

    def test_remove_without_db_does_not_crash(self):
        mgr = GlobalContextManager()
        s = mgr.add_source("github_org", {"org": "acme"})
        mgr.remove_source(s.id)
        assert s.id not in mgr._sources


# ---------------------------------------------------------------------------
# list_sources
# ---------------------------------------------------------------------------


class TestListSources:
    def test_empty_returns_empty_list(self):
        mgr = GlobalContextManager()
        assert mgr.list_sources() == []

    def test_one_source_returns_dict_with_correct_keys(self):
        mgr = GlobalContextManager()
        s = mgr.add_source("github_org", {"org": "acme"}, schedule="0 3 * * *")
        s.status = "ready"
        s.last_sync = 100.0
        result = mgr.list_sources()
        assert len(result) == 1
        assert set(result[0].keys()) == {
            "id",
            "type",
            "config",
            "schedule",
            "status",
            "last_sync",
        }
        assert result[0]["id"] == s.id
        assert result[0]["type"] == "github_org"
        assert result[0]["config"] == {"org": "acme"}
        assert result[0]["schedule"] == "0 3 * * *"
        assert result[0]["status"] == "ready"
        assert result[0]["last_sync"] == 100.0

    def test_multiple_sources_returns_all(self):
        mgr = GlobalContextManager()
        s1 = mgr.add_source("github_org", {"org": "acme"})
        s2 = mgr.add_source("jira", {"url": "https://jira"})
        result = mgr.list_sources()
        assert len(result) == 2
        ids = {r["id"] for r in result}
        assert ids == {s1.id, s2.id}

    def test_reflects_mutations(self):
        mgr = GlobalContextManager()
        s = mgr.add_source("web_crawl", {"urls": []})
        s.status = "error"
        s.error_message = "boom"
        result = mgr.list_sources()
        assert result[0]["status"] == "error"


# ---------------------------------------------------------------------------
# sync_source
# ---------------------------------------------------------------------------


class TestSyncSource:
    def test_unknown_source_id_returns_false(self):
        mgr = GlobalContextManager()
        assert mgr.sync_source("nope") is False

    def test_github_org_calls_sync_github(self, monkeypatch):
        mgr = GlobalContextManager()
        calls = []
        monkeypatch.setattr(mgr, "_sync_github", lambda s: calls.append(s))
        s = mgr.add_source("github_org", {"org": "acme"})
        result = mgr.sync_source(s.id)
        assert result is True
        assert len(calls) == 1
        assert calls[0] is s
        assert s.status == "ready"
        assert s.last_sync > 0
        assert s.error_message == ""

    def test_web_crawl_calls_sync_web_crawl(self, monkeypatch):
        mgr = GlobalContextManager()
        calls = []
        monkeypatch.setattr(mgr, "_sync_web_crawl", lambda s: calls.append(s))
        s = mgr.add_source("web_crawl", {"urls": ["https://a.com"]})
        result = mgr.sync_source(s.id)
        assert result is True
        assert len(calls) == 1
        assert calls[0] is s
        assert s.status == "ready"
        assert s.last_sync > 0

    def test_confluence_dispatches_to_sync_web_crawl(self, monkeypatch):
        mgr = GlobalContextManager()
        calls = []
        monkeypatch.setattr(mgr, "_sync_web_crawl", lambda s: calls.append(s))
        s = mgr.add_source("confluence", {"url": "https://wiki"})
        result = mgr.sync_source(s.id)
        assert result is True
        assert len(calls) == 1
        assert s.status == "ready"

    def test_jira_dispatches_to_sync_web_crawl(self, monkeypatch):
        mgr = GlobalContextManager()
        calls = []
        monkeypatch.setattr(mgr, "_sync_web_crawl", lambda s: calls.append(s))
        s = mgr.add_source("jira", {"url": "https://jira"})
        result = mgr.sync_source(s.id)
        assert result is True
        assert len(calls) == 1
        assert s.status == "ready"

    def test_exception_during_sync_status_error_and_returns_false(self, monkeypatch):
        mgr = GlobalContextManager()
        monkeypatch.setattr(mgr, "_sync_github", lambda s: (_ for _ in ()).throw(RuntimeError("kaboom")))
        s = mgr.add_source("github_org", {"org": "acme"})
        result = mgr.sync_source(s.id)
        assert result is False
        assert s.status == "error"
        assert s.error_message == "kaboom"

    def test_sets_status_syncing_before_sync(self, monkeypatch):
        mgr = GlobalContextManager()
        statuses = []
        original_sync = lambda s: statuses.append(s.status)
        monkeypatch.setattr(mgr, "_sync_github", original_sync)
        s = mgr.add_source("github_org", {"org": "acme"})
        mgr.sync_source(s.id)
        assert statuses == ["syncing"]

    def test_exception_truncates_long_message(self, monkeypatch):
        mgr = GlobalContextManager()
        long_msg = "x" * 1000
        monkeypatch.setattr(
            mgr, "_sync_web_crawl", lambda s: (_ for _ in ()).throw(RuntimeError(long_msg))
        )
        s = mgr.add_source("web_crawl", {"urls": []})
        mgr.sync_source(s.id)
        assert len(s.error_message) == 500
        assert s.error_message == long_msg[:500]

    def test_unknown_type_does_not_call_any_sync(self, monkeypatch):
        mgr = GlobalContextManager()
        gh_calls = []
        wc_calls = []
        monkeypatch.setattr(mgr, "_sync_github", lambda s: gh_calls.append(s))
        monkeypatch.setattr(mgr, "_sync_web_crawl", lambda s: wc_calls.append(s))
        s = mgr.add_source("unknown_type", {})
        result = mgr.sync_source(s.id)
        assert result is True
        assert gh_calls == []
        assert wc_calls == []
        assert s.status == "ready"


# ---------------------------------------------------------------------------
# build_global_context
# ---------------------------------------------------------------------------


class TestBuildGlobalContext:
    def test_no_sources_returns_empty(self):
        mgr = GlobalContextManager()
        assert mgr.build_global_context("query") == ""

    def test_all_non_ready_returns_empty(self):
        mgr = GlobalContextManager()
        mgr.add_source("github_org", {"org": "acme"})
        assert mgr.build_global_context("query") == ""

    def test_one_ready_source_with_org_key(self):
        mgr = GlobalContextManager()
        s = mgr.add_source("github_org", {"org": "acme"})
        s.status = "ready"
        result = mgr.build_global_context("query")
        assert "## Global Context" in result
        assert "github_org: acme" in result

    def test_one_ready_source_with_url_key(self):
        mgr = GlobalContextManager()
        s = mgr.add_source("web_crawl", {"url": "https://example.com"})
        s.status = "ready"
        result = mgr.build_global_context("query")
        assert "web_crawl: https://example.com" in result

    def test_ready_source_with_neither_org_nor_url(self):
        mgr = GlobalContextManager()
        s = mgr.add_source("jira", {"token": "abc"})
        s.status = "ready"
        result = mgr.build_global_context("query")
        assert "jira: ?" in result

    def test_mix_of_ready_and_non_ready_only_includes_ready(self):
        mgr = GlobalContextManager()
        ready = mgr.add_source("github_org", {"org": "acme"})
        ready.status = "ready"
        not_ready = mgr.add_source("web_crawl", {"url": "https://x.com"})
        not_ready.status = "pending"
        result = mgr.build_global_context("query")
        assert "acme" in result
        assert "x.com" not in result

    def test_max_tokens_param_accepted(self):
        mgr = GlobalContextManager()
        s = mgr.add_source("github_org", {"org": "acme"})
        s.status = "ready"
        result = mgr.build_global_context("query", max_tokens=500)
        assert "Global Context" in result


# ---------------------------------------------------------------------------
# _sync_github
# ---------------------------------------------------------------------------


class TestSyncGithub:
    def test_import_error_logs_warning_no_exception(self, caplog, monkeypatch):
        mgr = GlobalContextManager()
        monkeypatch.setitem(sys.modules, "core.sources.github_connector", None)
        s = GlobalSource(id="gs_1", type="github_org", config={"org": "acme"})
        with caplog.at_level("WARNING"):
            mgr._sync_github(s)
        assert "Trove GitHub connector not available" in caplog.text

    def test_success_mocks_connector(self, monkeypatch):
        mgr = GlobalContextManager()
        mock_connector_cls = MagicMock()
        mock_connector = MagicMock()
        mock_connector_cls.return_value = mock_connector
        fake_module = MagicMock()
        fake_module.GitHubSourceConnector = mock_connector_cls
        monkeypatch.setitem(sys.modules, "core.sources.github_connector", fake_module)
        s = GlobalSource(id="gs_1", type="github_org", config={"org": "acme"})
        mgr._sync_github(s)
        mock_connector_cls.assert_called_once_with({"org": "acme"})
        mock_connector.sync.assert_called_once_with(target_dir="/tmp/mullet_global/gs_1")

    def test_success_connector_instance_created(self, monkeypatch):
        mgr = GlobalContextManager()
        mock_connector_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.GitHubSourceConnector = mock_connector_cls
        monkeypatch.setitem(sys.modules, "core.sources.github_connector", fake_module)
        s = GlobalSource(id="gs_2", type="github_org", config={"org": "corp"})
        mgr._sync_github(s)
        assert mock_connector_cls.called


# ---------------------------------------------------------------------------
# _sync_web_crawl
# ---------------------------------------------------------------------------


class TestSyncWebCrawl:
    def test_import_error_logs_warning_no_exception(self, caplog, monkeypatch):
        mgr = GlobalContextManager()
        monkeypatch.setitem(sys.modules, "core.sources.web_crawler", None)
        s = GlobalSource(id="gs_1", type="web_crawl", config={"urls": ["https://a.com"]})
        with caplog.at_level("WARNING"):
            mgr._sync_web_crawl(s)
        assert "Trove web crawler not available" in caplog.text

    def test_success_mocks_crawler_and_calls_crawl_per_url(self, monkeypatch):
        mgr = GlobalContextManager()
        mock_crawler_cls = MagicMock()
        mock_crawler = MagicMock()
        mock_crawler_cls.return_value = mock_crawler
        fake_module = MagicMock()
        fake_module.WebCrawler = mock_crawler_cls
        monkeypatch.setitem(sys.modules, "core.sources.web_crawler", fake_module)
        s = GlobalSource(
            id="gs_1",
            type="web_crawl",
            config={"urls": ["https://a.com", "https://b.com"], "depth": 5},
        )
        mgr._sync_web_crawl(s)
        mock_crawler_cls.assert_called_once_with()
        assert mock_crawler.crawl.call_count == 2
        mock_crawler.crawl.assert_any_call("https://a.com", max_depth=5)
        mock_crawler.crawl.assert_any_call("https://b.com", max_depth=5)

    def test_default_depth_when_not_in_config(self, monkeypatch):
        mgr = GlobalContextManager()
        mock_crawler = MagicMock()
        mock_crawler_cls = MagicMock(return_value=mock_crawler)
        fake_module = MagicMock()
        fake_module.WebCrawler = mock_crawler_cls
        monkeypatch.setitem(sys.modules, "core.sources.web_crawler", fake_module)
        s = GlobalSource(
            id="gs_1", type="web_crawl", config={"urls": ["https://c.com"]}
        )
        mgr._sync_web_crawl(s)
        mock_crawler.crawl.assert_called_once_with("https://c.com", max_depth=3)

    def test_empty_urls_list_no_crawl_calls(self, monkeypatch):
        mgr = GlobalContextManager()
        mock_crawler = MagicMock()
        mock_crawler_cls = MagicMock(return_value=mock_crawler)
        fake_module = MagicMock()
        fake_module.WebCrawler = mock_crawler_cls
        monkeypatch.setitem(sys.modules, "core.sources.web_crawler", fake_module)
        s = GlobalSource(id="gs_1", type="web_crawl", config={"urls": []})
        mgr._sync_web_crawl(s)
        mock_crawler_cls.assert_called_once_with()
        mock_crawler.crawl.assert_not_called()


# ---------------------------------------------------------------------------
# _save_to_db
# ---------------------------------------------------------------------------


class TestSaveToDb:
    def test_no_db_returns_immediately(self):
        mgr = GlobalContextManager()
        s = GlobalSource(id="gs_1", type="github_org", config={"org": "acme"})
        mgr._save_to_db(s)
        # No exception raised; nothing to assert.

    def test_db_success_executes_create_table_insert_and_commit(self):
        db = MagicMock()
        mgr = GlobalContextManager(db=db)
        s = GlobalSource(
            id="gs_1",
            type="github_org",
            config={"org": "acme"},
            schedule="0 2 * * *",
            last_sync=100.0,
            status="ready",
        )
        mgr._save_to_db(s)
        assert db.execute.call_count == 2
        create_call = db.execute.call_args_list[0]
        assert "CREATE TABLE IF NOT EXISTS global_sources" in create_call[0][0]
        insert_call = db.execute.call_args_list[1]
        assert "INSERT OR REPLACE INTO global_sources" in insert_call[0][0]
        assert insert_call[0][1] == (
            "gs_1",
            "github_org",
            json.dumps({"org": "acme"}),
            "0 2 * * *",
            100.0,
            "ready",
        )
        db.commit.assert_called_once()

    def test_db_exception_logs_warning_no_exception_raised(self, caplog):
        db = MagicMock()
        db.execute.side_effect = RuntimeError("db locked")
        mgr = GlobalContextManager(db=db)
        s = GlobalSource(id="gs_1", type="github_org", config={})
        with caplog.at_level("WARNING"):
            mgr._save_to_db(s)
        assert "Failed to save source to DB: db locked" in caplog.text

    def test_db_success_for_different_source_types(self):
        db = MagicMock()
        mgr = GlobalContextManager(db=db)
        s = GlobalSource(
            id="gs_2",
            type="confluence",
            config={"url": "https://wiki"},
            schedule="0 5 * * *",
            last_sync=0,
            status="pending",
        )
        mgr._save_to_db(s)
        insert_call = db.execute.call_args_list[1]
        assert insert_call[0][1][1] == "confluence"
        assert insert_call[0][1][2] == json.dumps({"url": "https://wiki"})


# ---------------------------------------------------------------------------
# Integration-style / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_add_and_remove_cycle(self):
        mgr = GlobalContextManager()
        s = mgr.add_source("github_org", {"org": "acme"})
        assert len(mgr.list_sources()) == 1
        mgr.remove_source(s.id)
        assert len(mgr.list_sources()) == 0

    def test_sync_updates_last_sync_monotonically(self, monkeypatch):
        mgr = GlobalContextManager()
        monkeypatch.setattr(mgr, "_sync_github", lambda s: None)
        s = mgr.add_source("github_org", {"org": "acme"})
        before = time.time()
        mgr.sync_source(s.id)
        after = time.time()
        assert before <= s.last_sync <= after

    def test_build_global_context_preserves_order(self):
        mgr = GlobalContextManager()
        s1 = mgr.add_source("github_org", {"org": "acme"})
        s1.status = "ready"
        s2 = mgr.add_source("jira", {"url": "https://jira"})
        s2.status = "ready"
        result = mgr.build_global_context("q")
        lines = result.split("\n")
        assert lines[0] == "## Global Context"
        assert "acme" in lines[1]
        assert "jira" in lines[2]

    def test_multiple_syncs_on_same_source(self, monkeypatch):
        mgr = GlobalContextManager()
        monkeypatch.setattr(mgr, "_sync_github", lambda s: None)
        s = mgr.add_source("github_org", {"org": "acme"})
        assert mgr.sync_source(s.id) is True
        first_sync = s.last_sync
        monkeypatch.setattr(time, "time", lambda: first_sync + 10)
        assert mgr.sync_source(s.id) is True
        assert s.last_sync == first_sync + 10
        assert s.status == "ready"

    def test_config_with_nested_dict(self):
        mgr = GlobalContextManager()
        nested = {"org": "acme", "opts": {"private": True}}
        s = mgr.add_source("github_org", nested)
        assert s.config["opts"]["private"] is True

    def test_db_remove_commits_even_after_prior_error(self):
        db = MagicMock()
        mgr = GlobalContextManager(db=db)
        s = mgr.add_source("github_org", {"org": "acme"})
        db.execute.side_effect = None
        mgr.remove_source(s.id)
        db.commit.assert_called()
