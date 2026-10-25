"""Tests for mullet.server.global_context and mullet.core.agent.error_log helpers."""


from mullet.server.global_context import GlobalSource, GlobalContextManager
from mullet.core.agent.error_log import get_error_log_path


# ---------------------------------------------------------------------------
# GlobalSource dataclass
# ---------------------------------------------------------------------------

class TestGlobalSource:
    def test_stores_id_and_type(self):
        s = GlobalSource(id="gs_1", type="github_org")
        assert s.id == "gs_1"
        assert s.type == "github_org"

    def test_default_schedule(self):
        s = GlobalSource(id="x", type="web_crawl")
        assert s.schedule == "0 2 * * *"

    def test_default_status_pending(self):
        s = GlobalSource(id="x", type="jira")
        assert s.status == "pending"

    def test_default_last_sync_zero(self):
        s = GlobalSource(id="x", type="confluence")
        assert s.last_sync == 0

    def test_default_error_message_empty(self):
        s = GlobalSource(id="x", type="web_crawl")
        assert s.error_message == ""

    def test_custom_config(self):
        s = GlobalSource(id="x", type="github_org", config={"org": "myorg"})
        assert s.config["org"] == "myorg"


# ---------------------------------------------------------------------------
# GlobalContextManager.add_source
# ---------------------------------------------------------------------------

class TestGlobalContextManagerAddSource:
    def setup_method(self):
        self.mgr = GlobalContextManager()

    def test_add_source_returns_global_source(self):
        src = self.mgr.add_source("github_org", {"org": "acme"})
        assert isinstance(src, GlobalSource)

    def test_add_source_type_stored(self):
        src = self.mgr.add_source("web_crawl", {"url": "https://example.com"})
        assert src.type == "web_crawl"

    def test_add_source_config_stored(self):
        src = self.mgr.add_source("github_org", {"org": "myorg"})
        assert src.config["org"] == "myorg"

    def test_add_source_initial_status_pending(self):
        src = self.mgr.add_source("jira", {})
        assert src.status == "pending"

    def test_add_source_appears_in_list(self):
        self.mgr.add_source("github_org", {"org": "test"})
        sources = self.mgr.list_sources()
        assert len(sources) == 1

    def test_multiple_sources_stored(self):
        self.mgr.add_source("github_org", {})
        self.mgr.add_source("web_crawl", {})
        assert len(self.mgr.list_sources()) == 2

    def test_custom_schedule_stored(self):
        src = self.mgr.add_source("confluence", {}, schedule="0 4 * * 1")
        assert src.schedule == "0 4 * * 1"


# ---------------------------------------------------------------------------
# GlobalContextManager.remove_source
# ---------------------------------------------------------------------------

class TestGlobalContextManagerRemoveSource:
    def setup_method(self):
        self.mgr = GlobalContextManager()

    def test_remove_existing_source(self):
        src = self.mgr.add_source("github_org", {})
        self.mgr.remove_source(src.id)
        assert len(self.mgr.list_sources()) == 0

    def test_remove_nonexistent_no_crash(self):
        self.mgr.remove_source("no-such-id")  # Should not raise

    def test_remove_one_of_two(self):
        s1 = self.mgr.add_source("github_org", {})
        self.mgr.add_source("web_crawl", {})
        self.mgr.remove_source(s1.id)
        assert len(self.mgr.list_sources()) == 1


# ---------------------------------------------------------------------------
# GlobalContextManager.list_sources
# ---------------------------------------------------------------------------

class TestGlobalContextManagerListSources:
    def setup_method(self):
        self.mgr = GlobalContextManager()

    def test_empty_manager_returns_empty_list(self):
        assert self.mgr.list_sources() == []

    def test_returns_list_of_dicts(self):
        self.mgr.add_source("github_org", {"org": "acme"})
        sources = self.mgr.list_sources()
        assert isinstance(sources, list)
        assert isinstance(sources[0], dict)

    def test_dict_has_expected_keys(self):
        self.mgr.add_source("web_crawl", {})
        s = self.mgr.list_sources()[0]
        assert "id" in s
        assert "type" in s
        assert "status" in s
        assert "schedule" in s

    def test_type_correct_in_list(self):
        self.mgr.add_source("jira", {"project": "PROJ"})
        s = self.mgr.list_sources()[0]
        assert s["type"] == "jira"


# ---------------------------------------------------------------------------
# GlobalContextManager.build_global_context
# ---------------------------------------------------------------------------

class TestBuildGlobalContext:
    def setup_method(self):
        self.mgr = GlobalContextManager()

    def test_no_sources_returns_empty(self):
        result = self.mgr.build_global_context("some query")
        assert result == ""

    def test_pending_source_not_included(self):
        src = self.mgr.add_source("github_org", {"org": "acme"})
        # status is still "pending"
        result = self.mgr.build_global_context("query")
        assert result == ""

    def test_ready_source_included(self):
        src = self.mgr.add_source("github_org", {"org": "acme"})
        src.status = "ready"
        result = self.mgr.build_global_context("query")
        assert "github_org" in result
        assert "acme" in result

    def test_multiple_ready_sources_all_included(self):
        s1 = self.mgr.add_source("github_org", {"org": "org1"})
        s2 = self.mgr.add_source("web_crawl", {"url": "https://example.com"})
        s1.status = "ready"
        s2.status = "ready"
        result = self.mgr.build_global_context("query")
        assert "github_org" in result
        assert "web_crawl" in result

    def test_error_source_not_included(self):
        src = self.mgr.add_source("jira", {"url": "https://jira.example.com"})
        src.status = "error"
        result = self.mgr.build_global_context("query")
        assert result == ""

    def test_result_contains_global_context_header(self):
        src = self.mgr.add_source("confluence", {"url": "https://wiki.example.com"})
        src.status = "ready"
        result = self.mgr.build_global_context("q")
        assert "Global Context" in result


# ---------------------------------------------------------------------------
# get_error_log_path
# ---------------------------------------------------------------------------

class TestGetErrorLogPath:
    def test_returns_string(self):
        path = get_error_log_path()
        assert isinstance(path, str)

    def test_ends_with_errors_log(self):
        path = get_error_log_path()
        assert path.endswith("errors.log")

    def test_contains_mullet_logs(self):
        from mullet.core.paths import state_dir
        path = get_error_log_path()
        assert path == str(state_dir() / "logs" / "errors.log")

    def test_log_dir_created(self):
        import os
        path = get_error_log_path()
        log_dir = os.path.dirname(path)
        assert os.path.isdir(log_dir)
