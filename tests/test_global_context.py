"""Tests for enterprise global context manager."""

from mullet.server.global_context import GlobalContextManager, GlobalSource


def test_add_source():
    mgr = GlobalContextManager()
    source = mgr.add_source("github_org", {"org": "mycompany"})
    assert source.type == "github_org"
    assert source.status == "pending"


def test_list_sources():
    mgr = GlobalContextManager()
    mgr.add_source("github_org", {"org": "co1"})
    mgr.add_source("web_crawl", {"urls": ["https://docs.example.com"]})
    sources = mgr.list_sources()
    assert len(sources) == 2


def test_remove_source():
    mgr = GlobalContextManager()
    source = mgr.add_source("github_org", {"org": "test"})
    mgr.remove_source(source.id)
    assert len(mgr.list_sources()) == 0


def test_build_global_context_empty():
    mgr = GlobalContextManager()
    ctx = mgr.build_global_context("query")
    assert ctx == ""


def test_global_source_defaults():
    s = GlobalSource(id="gs_1", type="github_org")
    assert s.status == "pending"
    assert s.last_sync == 0
    assert s.schedule == "0 2 * * *"
