"""Tests for 6 small zero-coverage knowledge modules — batch 4."""

import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock

# ===========================================================================
# Module 1: mullet/knowledge/sources/base.py
# ===========================================================================

from mullet.knowledge.sources.base import SyncResult, BaseSourceConnector


class TestSyncResult:
    def test_defaults(self):
        r = SyncResult(source_type="foo")
        assert r.source_type == "foo"
        assert r.files_added == 0
        assert r.files_updated == 0
        assert r.files_removed == 0
        assert r.errors == []
        assert r.total_changes == 0
        assert r.success is True

    def test_total_changes(self):
        r = SyncResult(source_type="foo", files_added=2, files_updated=3, files_removed=1)
        assert r.total_changes == 6

    def test_success_with_errors(self):
        r = SyncResult(source_type="foo", errors=["boom"])
        assert r.success is False

    def test_success_empty_errors(self):
        r = SyncResult(source_type="foo", errors=[])
        assert r.success is True


class TestBaseSourceConnector:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseSourceConnector()

    def test_can_subclass(self):
        class Dummy(BaseSourceConnector):
            def sync(self, target_dir: str) -> SyncResult:
                return SyncResult(source_type="dummy")

            def status(self) -> dict:
                return {}

        d = Dummy()
        result = d.sync("/tmp")
        assert isinstance(result, SyncResult)
        assert result.source_type == "dummy"


# ===========================================================================
# Module 2: mullet/knowledge/sources/github_models.py
# ===========================================================================

from mullet.knowledge.sources.github_models import GitHubSourceConfig, RepoInfo


class TestGitHubSourceConfig:
    def test_defaults(self):
        c = GitHubSourceConfig()
        assert c.owner == ""
        assert c.repo == ""
        assert c.token_env == ""
        assert c.include_repos == []
        assert c.exclude_repos == []
        assert c.include_forks is False
        assert c.include_private is True
        assert c.sync_on_ingest is True

    def test_post_init_parses_owner_repo(self):
        c = GitHubSourceConfig(repo="acme/widgets")
        assert c.owner == "acme"
        assert c.repo == "widgets"

    def test_post_init_does_not_override_existing_owner(self):
        c = GitHubSourceConfig(owner="acme", repo="widgets")
        assert c.owner == "acme"
        assert c.repo == "widgets"

    def test_token_none_when_no_env(self):
        c = GitHubSourceConfig(token_env="")
        assert c.token is None

    def test_token_reads_env(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        c = GitHubSourceConfig(token_env="MY_TOKEN")
        assert c.token == "secret123"

    def test_token_missing_env_returns_none(self):
        c = GitHubSourceConfig(token_env="NONEXISTENT_VAR_12345")
        assert c.token is None

    def test_from_dict(self):
        data = {
            "owner": "acme",
            "repo": "widgets",
            "token_env": "TOK",
            "include_repos": ["a", "b"],
            "exclude_repos": ["c"],
            "include_forks": True,
            "include_private": False,
            "sync_on_ingest": False,
        }
        c = GitHubSourceConfig.from_dict(data)
        assert c.owner == "acme"
        assert c.repo == "widgets"
        assert c.token_env == "TOK"
        assert c.include_repos == ["a", "b"]
        assert c.exclude_repos == ["c"]
        assert c.include_forks is True
        assert c.include_private is False
        assert c.sync_on_ingest is False

    def test_from_dict_defaults(self):
        c = GitHubSourceConfig.from_dict({})
        assert c.owner == ""
        assert c.repo == ""
        assert c.token_env == ""
        assert c.include_repos == []
        assert c.exclude_repos == []
        assert c.include_forks is False
        assert c.include_private is True
        assert c.sync_on_ingest is True


class TestRepoInfo:
    def test_from_api_response(self):
        data = {
            "name": "repo",
            "owner": {"login": "owner"},
            "html_url": "https://github.com/owner/repo",
            "clone_url": "https://github.com/owner/repo.git",
            "private": True,
            "description": "A repo",
            "default_branch": "develop",
            "size": 42,
            "fork": False,
        }
        r = RepoInfo.from_api_response(data)
        assert r.name == "repo"
        assert r.owner == "owner"
        assert r.url == "https://github.com/owner/repo"
        assert r.clone_url == "https://github.com/owner/repo.git"
        assert r.private is True
        assert r.description == "A repo"
        assert r.default_branch == "develop"
        assert r.size_kb == 42
        assert r.is_fork is False

    def test_from_api_response_defaults(self):
        data = {
            "name": "repo",
            "owner": {"login": "owner"},
        }
        r = RepoInfo.from_api_response(data)
        assert r.url == ""
        assert r.clone_url == ""
        assert r.private is False
        assert r.description == ""
        assert r.default_branch == "main"
        assert r.size_kb == 0
        assert r.is_fork is False

    def test_from_api_response_none_description(self):
        data = {
            "name": "repo",
            "owner": {"login": "owner"},
            "description": None,
        }
        r = RepoInfo.from_api_response(data)
        assert r.description == ""


# ===========================================================================
# Module 3: mullet/knowledge/graph/export.py
# ===========================================================================

from mullet.knowledge.graph.export import graph_stats, export_for_d3


def _make_mock_entity(name, entity_type, description="", chunk_ids=None):
    m = MagicMock()
    m.name = name
    m.entity_type = entity_type
    m.description = description
    m.source_chunk_ids = set(chunk_ids or [])
    return m


def _make_mock_rel(source, target, rel_type, weight=1.0, description=""):
    m = MagicMock()
    m.source_entity = source
    m.target_entity = target
    m.relationship_type = rel_type
    m.weight = weight
    m.description = description
    return m


class TestGraphStats:
    def test_zero_entities(self):
        graph = MagicMock()
        graph.stats = {"entity_count": 0, "relationship_count": 0}
        result = graph_stats(graph)
        assert result["density"] == 0.0
        assert result["entity_count"] == 0

    def test_one_entity(self):
        graph = MagicMock()
        graph.stats = {"entity_count": 1, "relationship_count": 5}
        result = graph_stats(graph)
        assert result["density"] == 0.0

    def test_density_calculation(self):
        graph = MagicMock()
        graph.stats = {"entity_count": 4, "relationship_count": 6}
        result = graph_stats(graph)
        expected = 6 / (4 * 3)
        assert result["density"] == round(expected, 6)

    def test_preserves_existing_keys(self):
        graph = MagicMock()
        graph.stats = {"entity_count": 2, "relationship_count": 1, "extra": "value"}
        result = graph_stats(graph)
        assert result["extra"] == "value"
        assert result["density"] == round(1 / (2 * 1), 6)


class TestExportForD3:
    def test_empty_graph(self):
        graph = MagicMock()
        graph._entities = {}
        graph._relationships = {}
        result = export_for_d3(graph)
        assert result == {"nodes": [], "links": [], "type_groups": {}}

    def test_nodes_and_links(self):
        graph = MagicMock()
        graph._entities = {
            "e1": _make_mock_entity("Alice", "person", "desc1", ["c1"]),
            "e2": _make_mock_entity("Bob", "person", "desc2", ["c2"]),
            "e3": _make_mock_entity("Acme", "org", "desc3", ["c3"]),
        }
        graph._relationships = {
            "r1": _make_mock_rel("e1", "e2", "knows", 0.8, "friends"),
            "r2": _make_mock_rel("e2", "e3", "works_at", 1.0, "job"),
        }
        result = export_for_d3(graph)

        assert len(result["nodes"]) == 3
        assert len(result["links"]) == 2
        assert result["type_groups"] == {"person": 0, "org": 1}

        node_map = {n["id"]: n for n in result["nodes"]}
        assert node_map["e1"]["name"] == "Alice"
        assert node_map["e1"]["type"] == "person"
        assert node_map["e1"]["group"] == 0
        assert node_map["e1"]["chunk_count"] == 1
        assert node_map["e3"]["group"] == 1

        link = result["links"][0]
        assert "source" in link
        assert "target" in link
        assert "type" in link
        assert "weight" in link
        assert "description" in link

    def test_max_nodes_limit(self):
        graph = MagicMock()
        graph._entities = {f"e{i}": _make_mock_entity(f"E{i}", "type") for i in range(10)}
        graph._relationships = {}
        result = export_for_d3(graph, max_nodes=3)
        assert len(result["nodes"]) == 3

    def test_links_only_between_exported_nodes(self):
        graph = MagicMock()
        graph._entities = {
            "e1": _make_mock_entity("A", "t"),
            "e2": _make_mock_entity("B", "t"),
        }
        graph._relationships = {
            "r1": _make_mock_rel("e1", "e2", "rel"),
            "r2": _make_mock_rel("e2", "e3", "rel"),  # e3 not in nodes
        }
        result = export_for_d3(graph)
        assert len(result["links"]) == 1
        assert result["links"][0]["source"] == "e1"


# ===========================================================================
# Module 4: mullet/knowledge/engine/prompts.py
# ===========================================================================

from mullet.knowledge.engine.prompts import discover_prompts, load_prompt, search_prompts


class TestDiscoverPrompts:
    def test_custom_dirs(self, tmp_path):
        d = tmp_path / "prompts"
        d.mkdir()
        (d / "hello.md").write_text("hi")
        (d / "world.txt").write_text("hey")
        (d / "skip.py").write_text("nope")
        result = discover_prompts([str(d)])
        names = {r["name"] for r in result}
        assert "hello" in names
        assert "world" in names
        assert "skip" not in names
        assert all(r["format"] in ("md", "txt", "json") for r in result)

    def test_missing_dir_skipped(self, tmp_path):
        result = discover_prompts([str(tmp_path / "nope")])
        assert result == []

    def test_no_dirs_provided(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        d = tmp_path / "prompts"
        d.mkdir()
        (d / "a.md").write_text("a")
        result = discover_prompts()
        names = {r["name"] for r in result}
        assert "a" in names

    def test_returns_size(self, tmp_path):
        d = tmp_path / "p"
        d.mkdir()
        (d / "x.md").write_text("12345")
        result = discover_prompts([str(d)])
        assert result[0]["size"] == 5


class TestLoadPrompt:
    def test_load_no_variables(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_text("Hello world")
        assert load_prompt(str(f)) == "Hello world"

    def test_load_with_variables(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_text("Hello {{name}}, you are {{age}} years old.")
        result = load_prompt(str(f), {"name": "Alice", "age": 30})
        assert result == "Hello Alice, you are 30 years old."

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prompt(str(tmp_path / "missing.txt"))


class TestSearchPrompts:
    def test_search_by_name(self, tmp_path):
        d = tmp_path / "prompts"
        d.mkdir()
        (d / "alpha.md").write_text("content alpha")
        prompts = discover_prompts([str(d)])
        result = search_prompts("alp", prompts)
        assert len(result) == 1
        assert result[0]["name"] == "alpha"

    def test_search_by_content(self, tmp_path):
        d = tmp_path / "prompts"
        d.mkdir()
        (d / "beta.md").write_text("the quick brown fox")
        prompts = discover_prompts([str(d)])
        result = search_prompts("quick", prompts)
        assert len(result) == 1
        assert result[0]["name"] == "beta"

    def test_search_no_match(self, tmp_path):
        d = tmp_path / "prompts"
        d.mkdir()
        (d / "gamma.md").write_text("something")
        prompts = discover_prompts([str(d)])
        result = search_prompts("zzzzz", prompts)
        assert result == []

    def test_search_auto_discover(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        d = tmp_path / "prompts"
        d.mkdir()
        (d / "delta.md").write_text("delta force")
        result = search_prompts("delta")
        assert len(result) == 1

    def test_search_missing_file_ignored(self, tmp_path, caplog):
        prompts = [{"name": "ghost", "path": str(tmp_path / "ghost.md"), "format": "md", "size": 0}]
        result = search_prompts("anything", prompts)
        assert result == []
        assert "ghost.md" in caplog.text or "failed to read" in caplog.text


# ===========================================================================
# Module 5: mullet/knowledge/library_pool.py
# ===========================================================================

from mullet.knowledge.library_pool import LibraryPoolConfig, LibraryPoolRegistry


class TestLibraryPoolConfig:
    def test_basic(self):
        c = LibraryPoolConfig(name="numpy", language="python", version="1.0", source_url="https://example.com")
        assert c.last_synced is None

    def test_with_last_synced(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        c = LibraryPoolConfig(name="x", language="py", version="1", source_url="url", last_synced=dt)
        assert c.last_synced == dt


class TestLibraryPoolRegistry:
    def test_load_nonexistent_path(self, tmp_path):
        reg = LibraryPoolRegistry(path=tmp_path / "missing.json")
        assert reg.list() == []
        assert reg.get("foo") is None

    def test_register_and_get(self, tmp_path):
        reg = LibraryPoolRegistry(path=tmp_path / "pools.json")
        c = LibraryPoolConfig(name="lib1", language="py", version="2.0", source_url="https://lib1.com")
        reg.register(c)
        got = reg.get("lib1")
        assert got is not None
        assert got.name == "lib1"
        assert got.language == "py"

    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "pools.json"
        reg = LibraryPoolRegistry(path=path)
        c = LibraryPoolConfig(name="lib2", language="js", version="3.0", source_url="https://lib2.com")
        reg.register(c)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["pools"][0]["name"] == "lib2"

    def test_load_from_disk(self, tmp_path):
        path = tmp_path / "pools.json"
        dt = datetime(2024, 6, 15, 10, 30, 0)
        payload = {
            "pools": [
                {
                    "name": "lib3",
                    "language": "go",
                    "version": "1.0",
                    "source_url": "https://lib3.com",
                    "last_synced": dt.isoformat(),
                }
            ]
        }
        path.write_text(json.dumps(payload))
        reg = LibraryPoolRegistry(path=path)
        got = reg.get("lib3")
        assert got is not None
        assert got.language == "go"
        assert got.last_synced == dt

    def test_list(self, tmp_path):
        reg = LibraryPoolRegistry(path=tmp_path / "pools.json")
        reg.register(LibraryPoolConfig(name="a", language="py", version="1", source_url="u1"))
        reg.register(LibraryPoolConfig(name="b", language="js", version="2", source_url="u2"))
        items = reg.list()
        assert len(items) == 2
        assert {i.name for i in items} == {"a", "b"}

    def test_unregister_existing(self, tmp_path):
        reg = LibraryPoolRegistry(path=tmp_path / "pools.json")
        reg.register(LibraryPoolConfig(name="x", language="py", version="1", source_url="u"))
        assert reg.unregister("x") is True
        assert reg.get("x") is None

    def test_unregister_missing(self, tmp_path):
        reg = LibraryPoolRegistry(path=tmp_path / "pools.json")
        assert reg.unregister("nope") is False

    def test_corrupt_file_handled(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        reg = LibraryPoolRegistry(path=path)
        assert reg.list() == []

    def test_save_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "pools.json"
        reg = LibraryPoolRegistry(path=deep)
        reg.register(LibraryPoolConfig(name="deep", language="py", version="1", source_url="u"))
        assert deep.exists()

    def test_none_last_synced_roundtrip(self, tmp_path):
        reg = LibraryPoolRegistry(path=tmp_path / "pools.json")
        reg.register(LibraryPoolConfig(name="n", language="py", version="1", source_url="u", last_synced=None))
        data = json.loads((tmp_path / "pools.json").read_text())
        assert data["pools"][0]["last_synced"] is None


# ===========================================================================
# Module 6: mullet/knowledge/engine/code_extractor.py
# ===========================================================================

from mullet.knowledge.engine.code_extractor import PythonCodeExtractor, CodeChunk, register


class TestCodeChunk:
    def test_attributes(self):
        c = CodeChunk("foo", "function", "def foo(): pass", 1, 1)
        assert c.name == "foo"
        assert c.type == "function"
        assert c.content == "def foo(): pass"
        assert c.start_line == 1
        assert c.end_line == 1

    def test_repr(self):
        c = CodeChunk("bar", "class", "class Bar:", 2, 3)
        r = repr(c)
        assert "bar" in r
        assert "class" in r
        assert "2" in r
        assert "3" in r


class TestPythonCodeExtractor:
    def test_extract_function(self):
        source = "def hello():\n    pass\n"
        extractor = PythonCodeExtractor()
        chunks = extractor.extract(source)
        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].type == "function"
        assert "def hello():" in chunks[0].content

    def test_extract_class(self):
        source = "class MyClass:\n    pass\n"
        extractor = PythonCodeExtractor()
        chunks = extractor.extract(source)
        assert len(chunks) == 1
        assert chunks[0].name == "MyClass"
        assert chunks[0].type == "class"

    def test_extract_import(self):
        source = "import os\nfrom pathlib import Path\n"
        extractor = PythonCodeExtractor()
        chunks = extractor.extract(source)
        assert len(chunks) == 2
        assert all(c.type == "import" for c in chunks)

    def test_extract_mixed(self):
        source = (
            "import os\n"
            "\n"
            "def foo():\n"
            "    pass\n"
            "\n"
            "class Bar:\n"
            "    def method(self):\n"
            "        pass\n"
        )
        extractor = PythonCodeExtractor()
        chunks = extractor.extract(source)
        types = [c.type for c in chunks]
        assert "import" in types
        assert "function" in types
        assert "class" in types
        # method is inside class body, not a top-level child, so not extracted
        assert all(c.type in ("import", "function", "class") for c in chunks)

    def test_empty_source(self):
        extractor = PythonCodeExtractor()
        assert extractor.extract("") == []

    def test_lines(self):
        source = "\n\ndef foo():\n    pass\n"
        extractor = PythonCodeExtractor()
        chunks = extractor.extract(source)
        assert chunks[0].start_line == 3
        assert chunks[0].end_line == 4


class TestRegister:
    def test_register_returns_dict(self):
        result = register()
        assert "python" in result
        assert result["python"] is PythonCodeExtractor
