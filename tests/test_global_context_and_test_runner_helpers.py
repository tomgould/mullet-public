"""Tests for global_context dataclasses/manager and test_runner schema/constants."""

import json

from mullet.server.global_context import (
    GlobalSource,
    GlobalContextManager,
)
from mullet.core.tools.test_runner import (
    TEST_RUNNER_SCHEMA,
    MAX_OUTPUT_CHARS,
    run_tests_tool,
)


# ---------------------------------------------------------------------------
# GlobalSource dataclass
# ---------------------------------------------------------------------------

class TestGlobalSource:
    def test_fields_stored(self):
        s = GlobalSource(id="gs1", type="github_org")
        assert s.id == "gs1"
        assert s.type == "github_org"

    def test_status_default_pending(self):
        s = GlobalSource(id="gs1", type="github_org")
        assert s.status == "pending"

    def test_last_sync_default_zero(self):
        s = GlobalSource(id="gs1", type="github_org")
        assert s.last_sync == 0

    def test_schedule_default(self):
        s = GlobalSource(id="gs1", type="github_org")
        assert "2" in s.schedule  # "0 2 * * *"

    def test_config_default_empty(self):
        s = GlobalSource(id="gs1", type="web_crawl")
        assert s.config == {}

    def test_error_message_default_empty(self):
        s = GlobalSource(id="gs1", type="jira")
        assert s.error_message == ""


# ---------------------------------------------------------------------------
# GlobalContextManager
# ---------------------------------------------------------------------------

class TestGlobalContextManagerInit:
    def test_list_sources_empty_initially(self):
        mgr = GlobalContextManager()
        assert mgr.list_sources() == []


class TestAddSource:
    def test_returns_global_source(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("github_org", {"org": "myorg"})
        assert isinstance(src, GlobalSource)

    def test_source_type_stored(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("web_crawl", {})
        assert src.type == "web_crawl"

    def test_source_id_has_prefix(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("github_org", {})
        assert src.id.startswith("gs_")

    def test_config_stored(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("github_org", {"org": "myorg"})
        assert src.config["org"] == "myorg"

    def test_custom_schedule_stored(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("web_crawl", {}, schedule="0 6 * * *")
        assert src.schedule == "0 6 * * *"

    def test_source_in_list_after_add(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("github_org", {})
        ids = [s["id"] for s in mgr.list_sources()]
        assert src.id in ids


class TestRemoveSource:
    def test_removes_existing_source(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("github_org", {})
        mgr.remove_source(src.id)
        assert mgr.list_sources() == []

    def test_remove_nonexistent_no_error(self):
        mgr = GlobalContextManager()
        mgr.remove_source("nonexistent_id")  # should not raise

    def test_only_removes_specified(self):
        mgr = GlobalContextManager()
        s1 = mgr.add_source("github_org", {})
        s2 = mgr.add_source("web_crawl", {})
        mgr.remove_source(s1.id)
        remaining = [s["id"] for s in mgr.list_sources()]
        assert s2.id in remaining
        assert s1.id not in remaining


class TestListSources:
    def test_returns_list(self):
        mgr = GlobalContextManager()
        assert isinstance(mgr.list_sources(), list)

    def test_source_dict_has_required_keys(self):
        mgr = GlobalContextManager()
        mgr.add_source("github_org", {"org": "x"})
        s = mgr.list_sources()[0]
        for key in ("id", "type", "config", "schedule", "status", "last_sync"):
            assert key in s

    def test_multiple_sources(self):
        mgr = GlobalContextManager()
        mgr.add_source("github_org", {})
        mgr.add_source("web_crawl", {})
        assert len(mgr.list_sources()) == 2


class TestBuildGlobalContext:
    def test_empty_sources_returns_empty(self):
        mgr = GlobalContextManager()
        assert mgr.build_global_context("any query") == ""

    def test_pending_sources_returns_empty(self):
        mgr = GlobalContextManager()
        mgr.add_source("github_org", {"org": "myorg"})
        assert mgr.build_global_context("query") == ""

    def test_ready_source_returns_context(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("github_org", {"org": "myorg"})
        mgr._sources[src.id].status = "ready"
        ctx = mgr.build_global_context("query")
        assert "Global Context" in ctx

    def test_ready_source_config_in_context(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("github_org", {"org": "acmecorp"})
        mgr._sources[src.id].status = "ready"
        ctx = mgr.build_global_context("query")
        assert "acmecorp" in ctx

    def test_error_status_not_in_context(self):
        mgr = GlobalContextManager()
        src = mgr.add_source("github_org", {"org": "myorg"})
        mgr._sources[src.id].status = "error"
        ctx = mgr.build_global_context("query")
        assert ctx == ""

    def test_returns_string(self):
        mgr = GlobalContextManager()
        assert isinstance(mgr.build_global_context("q"), str)


# ---------------------------------------------------------------------------
# TEST_RUNNER_SCHEMA
# ---------------------------------------------------------------------------

class TestTestRunnerSchema:
    def test_has_description(self):
        assert "description" in TEST_RUNNER_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in TEST_RUNNER_SCHEMA

    def test_path_param_exists(self):
        props = TEST_RUNNER_SCHEMA["parameters"]["properties"]
        assert "path" in props

    def test_filter_param_exists(self):
        props = TEST_RUNNER_SCHEMA["parameters"]["properties"]
        assert "filter" in props

    def test_verbose_param_exists(self):
        props = TEST_RUNNER_SCHEMA["parameters"]["properties"]
        assert "verbose" in props

    def test_verbose_is_boolean_type(self):
        props = TEST_RUNNER_SCHEMA["parameters"]["properties"]
        assert props["verbose"]["type"] == "boolean"


# ---------------------------------------------------------------------------
# MAX_OUTPUT_CHARS constant
# ---------------------------------------------------------------------------

class TestMaxOutputChars:
    def test_is_positive(self):
        assert MAX_OUTPUT_CHARS > 0

    def test_is_int(self):
        assert isinstance(MAX_OUTPUT_CHARS, int)

    def test_is_large(self):
        assert MAX_OUTPUT_CHARS >= 1000


# ---------------------------------------------------------------------------
# run_tests_tool — pure validation paths
# ---------------------------------------------------------------------------

class TestRunTestsTool:
    def test_invalid_json_returns_error(self):
        result = json.loads(run_tests_tool("not json {"))
        assert "error" in result

    def test_returns_json_string(self):
        raw = run_tests_tool(path="/nonexistent_test_path_xyz")
        assert isinstance(raw, str)
        data = json.loads(raw)
        assert "framework" in data or "error" in data

    def test_kwargs_accepted(self):
        raw = run_tests_tool(path="/nonexistent_test_path_xyz")
        assert isinstance(raw, str)
