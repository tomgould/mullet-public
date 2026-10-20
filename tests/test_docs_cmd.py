"""Dedicated tests for mullet.cli.docs_cmd."""

import json
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mullet.cli.docs_cmd import docs, _resolve_fetcher, _fetch_pypi_latest, _fetch_npm_latest, _fetch_crates_latest
from mullet.knowledge.library_pool import LibraryPoolConfig, LibraryPoolRegistry


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch, tmp_path):
    """Force LibraryPoolRegistry default path into tmp_path so tests don't share state."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Ensure the .mullet dir exists so registry files land there
    (tmp_path / ".mullet").mkdir(parents=True, exist_ok=True)


class TestDocsList:
    def test_list_empty(self, runner):
        result = runner.invoke(docs, ["list"])
        assert result.exit_code == 0
        assert "No library pools registered" in result.output

    def test_list_shows_pools(self, runner):
        registry = LibraryPoolRegistry()
        registry.register(
            LibraryPoolConfig(
                name="requests",
                language="python",
                version="2.31.0",
                source_url="https://pypi.org/project/requests/",
                last_synced=datetime(2024, 1, 15, 10, 30, 0),
            )
        )
        result = runner.invoke(docs, ["list"])
        assert result.exit_code == 0
        assert "requests" in result.output
        assert "python" in result.output
        assert "2.31.0" in result.output


class TestDocsSync:
    def test_sync_empty(self, runner):
        result = runner.invoke(docs, ["sync"])
        assert result.exit_code == 0
        assert "No library pools registered" in result.output

    def test_sync_dry_run_no_fetch(self, runner):
        registry = LibraryPoolRegistry()
        registry.register(
            LibraryPoolConfig(
                name="requests",
                language="python",
                version="2.31.0",
                source_url="https://pypi.org/project/requests/",
            )
        )
        result = runner.invoke(docs, ["sync", "--dry-run"])
        assert result.exit_code == 0
        assert "Dry-run" in result.output

    def test_sync_updates_version(self, runner):
        registry = LibraryPoolRegistry()
        registry.register(
            LibraryPoolConfig(
                name="requests",
                language="python",
                version="2.31.0",
                source_url="https://pypi.org/project/requests/",
            )
        )

        with mock.patch(
            "mullet.cli.docs_cmd._fetch_pypi_latest", return_value="2.32.0"
        ):
            result = runner.invoke(docs, ["sync"])

        assert result.exit_code == 0
        assert "2.31.0 → 2.32.0" in result.output
        assert "1 updated" in result.output

        updated = LibraryPoolRegistry().get("requests")
        assert updated is not None
        assert updated.version == "2.32.0"
        assert updated.last_synced is not None

    def test_sync_up_to_date(self, runner):
        registry = LibraryPoolRegistry()
        registry.register(
            LibraryPoolConfig(
                name="requests",
                language="python",
                version="2.31.0",
                source_url="https://pypi.org/project/requests/",
            )
        )

        with mock.patch(
            "mullet.cli.docs_cmd._fetch_pypi_latest", return_value="2.31.0"
        ):
            result = runner.invoke(docs, ["sync"])

        assert result.exit_code == 0
        assert "up to date" in result.output
        assert "0 updated" in result.output

    def test_sync_unknown_registry(self, runner):
        registry = LibraryPoolRegistry()
        registry.register(
            LibraryPoolConfig(
                name="custom-lib",
                language="zig",
                version="1.0.0",
                source_url="https://example.com",
            )
        )
        result = runner.invoke(docs, ["sync"])
        assert result.exit_code == 0
        assert "unknown registry" in result.output

    def test_sync_fetcher_returns_none(self, runner):
        """Lines 61-63: fetcher returns None → failed counter incremented."""
        registry = LibraryPoolRegistry()
        registry.register(
            LibraryPoolConfig(
                name="requests",
                language="python",
                version="2.31.0",
                source_url="https://pypi.org/project/requests/",
            )
        )
        with mock.patch("mullet.cli.docs_cmd._fetch_pypi_latest", return_value=None):
            result = runner.invoke(docs, ["sync"])
        assert result.exit_code == 0
        assert "could not check version" in result.output
        assert "1 failed" in result.output


class TestDocsRemove:
    def test_remove_existing(self, runner):
        registry = LibraryPoolRegistry()
        registry.register(
            LibraryPoolConfig(
                name="requests",
                language="python",
                version="2.31.0",
                source_url="https://pypi.org/project/requests/",
            )
        )
        result = runner.invoke(docs, ["remove", "requests"])
        assert result.exit_code == 0
        assert "Removed library pool 'requests'" in result.output
        assert LibraryPoolRegistry().get("requests") is None

    def test_remove_missing(self, runner):
        result = runner.invoke(docs, ["remove", "missing"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestResolveFetcher:
    def test_pypi_url(self):
        assert _resolve_fetcher("https://pypi.org/project/requests/", "python") is _fetch_pypi_latest

    def test_npm_url(self):
        assert _resolve_fetcher("https://www.npmjs.com/package/lodash", "javascript") is _fetch_npm_latest

    def test_crates_url(self):
        assert _resolve_fetcher("https://crates.io/crates/serde", "rust") is _fetch_crates_latest

    def test_language_hint_python(self):
        assert _resolve_fetcher("https://example.com", "python") is _fetch_pypi_latest

    def test_language_hint_js(self):
        assert _resolve_fetcher("https://example.com", "javascript") is _fetch_npm_latest

    def test_language_hint_ts(self):
        assert _resolve_fetcher("https://example.com", "typescript") is _fetch_npm_latest

    def test_unknown_returns_none(self):
        assert _resolve_fetcher("https://example.com", "zig") is None


class _FakeResponse:
    """Minimal fake HTTP response that works as a context manager."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestFetchers:
    def test_fetch_pypi_latest_parses_version(self, monkeypatch):
        fake_data = json.dumps({"info": {"version": "3.0.0"}}).encode()
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=None: _FakeResponse(fake_data)
        )
        assert _fetch_pypi_latest("fake-package") == "3.0.0"

    def test_fetch_pypi_latest_failure(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            raise OSError("network down")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        assert _fetch_pypi_latest("fake-package") is None

    def test_fetch_npm_latest_parses_version(self, monkeypatch):
        fake_data = json.dumps({"version": "4.5.6"}).encode()
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=None: _FakeResponse(fake_data)
        )
        assert _fetch_npm_latest("fake-pkg") == "4.5.6"

    def test_fetch_npm_latest_failure(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            raise OSError("down")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        assert _fetch_npm_latest("fake-pkg") is None

    def test_fetch_crates_latest_parses_version(self, monkeypatch):
        fake_data = json.dumps({"crate": {"max_version": "1.2.3"}}).encode()
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=None: _FakeResponse(fake_data)
        )
        assert _fetch_crates_latest("fake-crate") == "1.2.3"

    def test_fetch_crates_latest_failure(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            raise OSError("down")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        assert _fetch_crates_latest("fake-crate") is None
