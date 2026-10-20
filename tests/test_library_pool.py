"""Tests for the library pool registry."""

from datetime import datetime
from pathlib import Path

from click.testing import CliRunner

from mullet.knowledge.library_pool import LibraryPoolConfig, LibraryPoolRegistry
from mullet.cli.docs_cmd import docs


def test_register_and_get():
    registry = LibraryPoolRegistry()
    config = LibraryPoolConfig(
        name="test-lib",
        language="python",
        version="1.0.0",
        source_url="https://example.com",
        last_synced=datetime(2024, 1, 1, 12, 0, 0),
    )
    registry.register(config)
    result = registry.get("test-lib")
    assert result is not None
    assert result.name == "test-lib"
    assert result.language == "python"
    assert result.version == "1.0.0"


def test_list_empty(tmp_path: Path):
    registry = LibraryPoolRegistry(path=tmp_path / "library_pools.json")
    assert registry.list() == []


def test_persist_roundtrip(tmp_path: Path):
    path = tmp_path / "library_pools.json"
    registry = LibraryPoolRegistry(path=path)
    config = LibraryPoolConfig(
        name="persist-lib",
        language="rust",
        version="2.3.4",
        source_url="https://crates.io",
        last_synced=datetime(2025, 6, 15, 10, 30, 0),
    )
    registry.register(config)

    new_registry = LibraryPoolRegistry(path=path)
    result = new_registry.get("persist-lib")
    assert result is not None
    assert result.name == "persist-lib"
    assert result.language == "rust"
    assert result.version == "2.3.4"
    assert result.last_synced == datetime(2025, 6, 15, 10, 30, 0)


def test_unregister():
    registry = LibraryPoolRegistry()
    config = LibraryPoolConfig(
        name="remove-me",
        language="python",
        version="1.0.0",
        source_url="https://example.com",
    )
    registry.register(config)
    assert registry.get("remove-me") is not None
    assert registry.unregister("remove-me") is True
    assert registry.get("remove-me") is None
    assert registry.unregister("remove-me") is False


def test_docs_list_cli():
    runner = CliRunner()
    result = runner.invoke(docs, ["list"])
    assert result.exit_code == 0
