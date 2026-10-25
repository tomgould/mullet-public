import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.knowledge.engine.client import (
    validate_config, Client, DEFAULT_CONFIG,
    load_client, discover_clients, get_active_client,
    _match_display_name,
)


def test_validate_config_valid():
    config = {
        "machine_name": "test_client",
        "name": "Test",
        "search": {
            "bm25_weight": 0.5,
            "vector_weight": 0.5,
            "top_k": 10,
            "merge_strategy": "weighted",
            "reranker": "local",
            "score_normalization": "minmax",
            "multi_query_count": 3,
        },
        "embedding": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
        },
    }
    assert validate_config(config) == []


def test_validate_config_missing_machine_name():
    assert validate_config({}) == ["Missing required field: 'machine_name'"]


def test_validate_config_short_machine_name():
    assert validate_config({"machine_name": "ab"}) == [
        "machine_name 'ab' is too short (min 3 chars, required by ChromaDB collection naming)"
    ]


def test_validate_config_unknown_keys():
    warnings = validate_config({
        "machine_name": "test",
        "foo": 1,
        "search": {"bar": 2},
        "graph": {"baz": 3},
        "code": {"qux": 4},
    })
    assert any("Unknown top-level key: 'foo'" in w for w in warnings)
    assert any("Unknown search key: 'search.bar'" in w for w in warnings)
    assert any("Unknown graph key: 'graph.baz'" in w for w in warnings)
    assert any("Unknown code key: 'code.qux'" in w for w in warnings)


def test_validate_config_search_ranges():
    warnings = validate_config({
        "machine_name": "test",
        "search": {
            "bm25_weight": 1.5,
            "vector_weight": -0.1,
            "top_k": 0,
            "merge_strategy": "invalid",
            "reranker": "remote",
            "score_normalization": "none",
            "multi_query_count": 10,
            "bm25_k1": 0,
            "bm25_b": 1.5,
            "mmr_lambda": 1.5,
        },
    })
    assert any("search.bm25_weight should be between 0 and 1" in w for w in warnings)
    assert any("search.vector_weight should be between 0 and 1" in w for w in warnings)
    assert any("search.top_k should be > 0" in w for w in warnings)
    assert any("search.merge_strategy should be 'weighted' or 'rrf'" in w for w in warnings)
    assert any("search.reranker should be 'local' or 'api'" in w for w in warnings)
    assert any("search.score_normalization should be 'minmax', 'zscore', or 'raw'" in w for w in warnings)
    assert any("search.multi_query_count should be between 2 and 7" in w for w in warnings)
    assert any("search.bm25_k1 should be > 0" in w for w in warnings)
    assert any("search.bm25_b should be between 0 and 1" in w for w in warnings)
    assert any("search.mmr_lambda should be between 0 and 1" in w for w in warnings)


def test_validate_config_embedding_dimension_mismatch():
    warnings = validate_config({
        "machine_name": "test",
        "embedding": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 512,
        },
    })
    assert any("embedding.dimension=512 doesn't match model 'all-MiniLM-L6-v2'" in w for w in warnings)


def test_client_search_config_merge():
    client = Client(Path("/tmp/test"), {"search": {"top_k": 5}})
    assert client.search_config["top_k"] == 5
    assert client.search_config["bm25_weight"] == DEFAULT_CONFIG["search"]["bm25_weight"]


def test_client_tokenization_config_merge():
    client = Client(Path("/tmp/test"), {"search": {"tokenization": {"stemmer": "porter"}}})
    assert client.tokenization_config["stemmer"] == "porter"
    assert client.tokenization_config["synonyms"] == DEFAULT_CONFIG["search"]["tokenization"]["synonyms"]


def test_client_get_document_extensions():
    client = Client(Path("/tmp/test"), {
        "documents": {
            "extensions": [".pdf", "txt", ".docx"],
            "exclude_extensions": [".txt"],
        }
    })
    exts = client.get_document_extensions()
    assert exts == (".docx", ".pdf")


class TestValidateConfigTypeBranches:
    """Cover the type-warning branches not hit by existing tests."""

    def test_search_field_wrong_type(self):
        """Line 229: bm25_weight given as string -> type warning."""
        warnings = validate_config({
            "machine_name": "test",
            "search": {"bm25_weight": "high"},
        })
        assert any("bm25_weight" in w and "numeric" in w for w in warnings)

    def test_bm25_k1_not_numeric(self):
        """Line 267: bm25_k1 is a string -> warning."""
        warnings = validate_config({"machine_name": "test", "search": {"bm25_k1": "bad"}})
        assert any("bm25_k1 should be numeric" in w for w in warnings)

    def test_bm25_b_not_numeric(self):
        """Line 274: bm25_b is a string -> warning."""
        warnings = validate_config({"machine_name": "test", "search": {"bm25_b": "bad"}})
        assert any("bm25_b should be numeric" in w for w in warnings)

    def test_multi_query_count_not_int(self):
        """Line 287: multi_query_count is float -> warning."""
        warnings = validate_config({"machine_name": "test", "search": {"multi_query_count": 3.0}})
        assert any("multi_query_count should be int" in w for w in warnings)

    def test_graph_search_mode_invalid(self):
        """Line 300: graph.search_mode invalid -> warning."""
        warnings = validate_config({"machine_name": "test", "graph": {"search_mode": "magic"}})
        assert any("search_mode" in w for w in warnings)

    def test_code_enabled_not_bool(self):
        """Line 327: code.enabled is string -> warning."""
        warnings = validate_config({"machine_name": "test", "code": {"enabled": "yes"}})
        assert any("code.enabled must be boolean" in w for w in warnings)

    def test_code_paths_not_list(self):
        """Line 332: code.paths is string -> warning."""
        warnings = validate_config({"machine_name": "test", "code": {"paths": "/some/path"}})
        assert any("code.paths must be a list" in w for w in warnings)

    def test_code_paths_item_not_string(self):
        """Lines 335-336: code.paths[0] is int -> warning."""
        warnings = validate_config({"machine_name": "test", "code": {"paths": [123]}})
        assert any("code.paths[0] must be a string" in w for w in warnings)

    def test_code_parallel_workers_not_int(self):
        """Line 341: code.parallel_workers is float -> warning."""
        warnings = validate_config({"machine_name": "test", "code": {"parallel_workers": 2.5}})
        assert any("parallel_workers must be a positive integer" in w for w in warnings)

    def test_code_parallel_workers_zero(self):
        """Line 343: code.parallel_workers=0 -> warning."""
        warnings = validate_config({"machine_name": "test", "code": {"parallel_workers": 0}})
        assert any("parallel_workers must be > 0" in w for w in warnings)

    def test_code_max_file_size_kb_invalid(self):
        """Line 348: code.max_file_size_kb=0 -> warning."""
        warnings = validate_config({"machine_name": "test", "code": {"max_file_size_kb": 0}})
        assert any("max_file_size_kb" in w for w in warnings)

    def test_code_git_unknown_key(self):
        """Line 354: code.git.unknown_key -> warning."""
        warnings = validate_config({"machine_name": "test", "code": {"git": {"badkey": True}}})
        assert any("code.git key" in w for w in warnings)


class TestClientProperties:
    def _client(self, tmp_path, config=None):
        return Client(tmp_path, config or {})

    def test_client_dir(self, tmp_path):
        """Line 372: client_dir returns path."""
        c = self._client(tmp_path)
        assert c.client_dir == tmp_path

    def test_sources_path(self, tmp_path):
        """Line 377: sources_path is path/sources."""
        c = self._client(tmp_path)
        assert c.sources_path == tmp_path / "sources"

    def test_sources_config_empty(self, tmp_path):
        """Line 382: sources_config returns empty dict when not configured."""
        c = self._client(tmp_path)
        assert c.sources_config == {}

    def test_has_sources_false_when_missing(self, tmp_path):
        """Line 387: has_sources=False when sources dir doesn't exist."""
        c = self._client(tmp_path)
        assert c.has_sources is False

    def test_has_sources_true_when_populated(self, tmp_path):
        """Line 387: has_sources=True when sources dir has files."""
        c = self._client(tmp_path)
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "repo").mkdir()
        assert c.has_sources is True

    def test_embedding_config_merges_defaults(self, tmp_path):
        """Line 419: embedding_config merges with defaults."""
        c = self._client(tmp_path, {"embedding": {"model": "custom-model"}})
        assert c.embedding_config["model"] == "custom-model"
        assert "dimension" in c.embedding_config

    def test_llm_config_merges_defaults(self, tmp_path):
        """Line 424: llm_config merges with defaults."""
        c = self._client(tmp_path, {"llm": {"model": "gpt-4"}})
        assert c.llm_config["model"] == "gpt-4"

    def test_graph_config_returns_defaults(self, tmp_path):
        """Lines 434-445: graph_config returns defaults when not configured."""
        c = self._client(tmp_path)
        g = c.graph_config
        assert g["enabled"] is False
        assert g["search_mode"] == "naive"
        assert "entity_types" in g

    def test_graph_config_overrides_defaults(self, tmp_path):
        """Lines 434-445: graph_config merges user config."""
        c = self._client(tmp_path, {"graph": {"enabled": True, "search_mode": "hybrid"}})
        g = c.graph_config
        assert g["enabled"] is True
        assert g["search_mode"] == "hybrid"

    def test_chunking_config_returns_defaults(self, tmp_path):
        """Lines 450-458: chunking_config returns defaults."""
        c = self._client(tmp_path)
        ch = c.chunking_config
        assert ch["strategy"] == "title"
        assert ch["max_chunk_size"] == 1500

    def test_chunking_config_overrides(self, tmp_path):
        """Lines 450-458: chunking_config merges user config."""
        c = self._client(tmp_path, {"chunking": {"max_chunk_size": 2000}})
        assert c.chunking_config["max_chunk_size"] == 2000
        assert c.chunking_config["strategy"] == "title"

    def test_get_document_filenames_default(self, tmp_path):
        """Lines 476-478: get_document_filenames returns default set."""
        c = self._client(tmp_path)
        fnames = c.get_document_filenames()
        assert isinstance(fnames, tuple)
        assert len(fnames) == 0

    def test_get_document_filenames_custom(self, tmp_path):
        """Lines 476-478: custom filenames list."""
        c = self._client(tmp_path, {"documents": {"filenames": ["Makefile", "Dockerfile"]}})
        fnames = c.get_document_filenames()
        assert "Makefile" in fnames
        assert "Dockerfile" in fnames

    def test_audit_config_property(self, tmp_path):
        """Line 483: audit_config returns AuditConfig."""
        from mullet.knowledge.engine.client import AuditConfig
        c = self._client(tmp_path)
        assert isinstance(c.audit_config, AuditConfig)

    def test_reference_config_property(self, tmp_path):
        """Line 488: reference_config returns ReferenceConfig."""
        from mullet.knowledge.engine.client import ReferenceConfig
        c = self._client(tmp_path)
        assert isinstance(c.reference_config, ReferenceConfig)

    def test_get_system_prompt_none_when_missing(self, tmp_path):
        """Lines 492-495: system.txt absent -> returns None."""
        c = self._client(tmp_path)
        assert c.get_system_prompt() is None

    def test_get_system_prompt_reads_file(self, tmp_path):
        """Lines 492-494: system.txt present -> content returned."""
        c = self._client(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "system.txt").write_text("  You are helpful.  ")
        result = c.get_system_prompt()
        assert result == "You are helpful."

    def test_code_config_property(self, tmp_path):
        """Line 500: code_config returns code section or empty dict."""
        c = self._client(tmp_path, {"code": {"enabled": True}})
        assert c.code_config == {"enabled": True}

    def test_ensure_directories_creates_dirs(self, tmp_path):
        """Lines 518-520: ensure_directories creates all required dirs."""
        c = self._client(tmp_path)
        c.ensure_directories()
        assert (tmp_path / "documents").is_dir()
        assert (tmp_path / "index").is_dir()
        assert (tmp_path / "prompts").is_dir()

    def test_get_code_chunks_default_path(self, tmp_path):
        """Lines 511-514: get_code_chunks with paths=None -> uses documents_path."""
        c = self._client(tmp_path)
        c.ensure_directories()
        chunks = c.get_code_chunks()  # empty dir -> empty list
        assert isinstance(chunks, list)


class TestLoadClient:
    def test_load_client_from_dir(self, tmp_path):
        """Lines 593-654: load_client reads config.yaml and returns Client."""
        import yaml
        client_dir = tmp_path / "mypool"
        client_dir.mkdir()
        config = {"machine_name": "mypool", "name": "My Pool"}
        (client_dir / "config.yaml").write_text(yaml.dump(config))
        client = load_client("mypool", clients_dir=tmp_path)
        assert client.name == "My Pool"
        assert client.machine_name == "mypool"

    def test_load_client_config_warnings_logged(self, tmp_path, caplog):
        """Line 648: config warnings are logged."""
        import yaml
        import logging
        client_dir = tmp_path / "badpool"
        client_dir.mkdir()
        config = {"machine_name": "ab"}  # short machine_name triggers warning
        (client_dir / "config.yaml").write_text(yaml.dump(config))
        with caplog.at_level(logging.WARNING):
            client = load_client("badpool", clients_dir=tmp_path)
        assert any("Config warning" in r.message for r in caplog.records)

    def test_load_client_missing_raises_valueerror(self, tmp_path):
        """Lines 604-635: missing config without create_if_missing -> ValueError."""
        from mullet.knowledge.engine.client import load_client
        with pytest.raises(ValueError, match="not found"):
            load_client("nonexistent", clients_dir=tmp_path)

    def test_load_client_create_if_missing(self, tmp_path):
        """Lines 637-638: create_if_missing=True -> uses DEFAULT_CONFIG."""
        client = load_client("newpool", clients_dir=tmp_path, create_if_missing=True)
        assert client.machine_name == "newpool"


class TestDiscoverClients:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        """Line 672: non-existent clients_dir -> empty list."""
        result = discover_clients(tmp_path / "missing")
        assert result == []

    def test_discovers_pool_dirs(self, tmp_path):
        """Lines 674-678: finds subdirs that don't start with '.'."""
        (tmp_path / "pool_a").mkdir()
        (tmp_path / "pool_b").mkdir()
        (tmp_path / ".hidden").mkdir()
        result = discover_clients(tmp_path)
        assert "pool_a" in result
        assert "pool_b" in result
        assert ".hidden" not in result

    def test_returns_sorted(self, tmp_path):
        """Line 679: returned list is sorted."""
        (tmp_path / "z_pool").mkdir()
        (tmp_path / "a_pool").mkdir()
        result = discover_clients(tmp_path)
        assert result == sorted(result)


class TestGetActiveClient:
    def test_uses_rag_client_env_var(self, tmp_path):
        """Lines 691-694: RAG_CLIENT env var -> load_client called with that name."""
        import yaml
        client_dir = tmp_path / "mypool"
        client_dir.mkdir()
        config = {"machine_name": "mypool"}
        (client_dir / "config.yaml").write_text(yaml.dump(config))
        with patch.dict(os.environ, {"RAG_CLIENT": "mypool"}):
            client = get_active_client(clients_dir=tmp_path)
        assert client.machine_name == "mypool"

    def test_falls_back_to_first_available(self, tmp_path):
        """Lines 697-699: no RAG_CLIENT -> uses first discovered pool."""
        import yaml
        client_dir = tmp_path / "firstpool"
        client_dir.mkdir()
        config = {"machine_name": "firstpool"}
        (client_dir / "config.yaml").write_text(yaml.dump(config))
        env = {k: v for k, v in os.environ.items() if k != "RAG_CLIENT"}
        with patch.dict(os.environ, env, clear=True):
            client = get_active_client(clients_dir=tmp_path)
        assert client.machine_name == "firstpool"

    def test_returns_none_when_no_pools(self, tmp_path):
        """Line 701: no pools at all -> None."""
        env = {k: v for k, v in os.environ.items() if k != "RAG_CLIENT"}
        with patch.dict(os.environ, env, clear=True):
            result = get_active_client(clients_dir=tmp_path)
        assert result is None


class TestMatchDisplayName:
    """Cover lines 538-565: _match_display_name function."""

    def _make_pool(self, tmp_path, slug, display_name):
        import yaml
        d = tmp_path / slug
        d.mkdir()
        (d / "config.yaml").write_text(yaml.dump({"machine_name": slug, "name": display_name}))
        return slug

    def test_returns_empty_when_no_pools(self, tmp_path):
        """Lines 552-553: display_to_slug empty → []."""
        result = _match_display_name("anything", tmp_path, [])
        assert result == []

    def test_exact_match_case_insensitive(self, tmp_path):
        """Lines 555-560: exact match returns slug."""
        self._make_pool(tmp_path, "my_pool", "My Pool")
        result = _match_display_name("my pool", tmp_path, ["my_pool"])
        assert result == ["my_pool"]

    def test_fuzzy_match(self, tmp_path):
        """Lines 562-565: fuzzy match via difflib."""
        self._make_pool(tmp_path, "my_pool", "My Pool")
        result = _match_display_name("My Poo", tmp_path, ["my_pool"])
        assert "my_pool" in result

    def test_no_match_returns_empty(self, tmp_path):
        """Lines 562-565: no close match → []."""
        self._make_pool(tmp_path, "my_pool", "My Pool")
        result = _match_display_name("completely_different", tmp_path, ["my_pool"])
        assert result == []

    def test_reads_client_yaml_fallback(self, tmp_path):
        """Lines 538-540: falls back to client.yaml if config.yaml absent."""
        import yaml
        d = tmp_path / "legacy_pool"
        d.mkdir()
        (d / "client.yaml").write_text(yaml.dump({"machine_name": "legacy_pool", "name": "Legacy Pool"}))
        result = _match_display_name("legacy pool", tmp_path, ["legacy_pool"])
        assert result == ["legacy_pool"]

    def test_invalid_yaml_silenced(self, tmp_path):
        """Lines 546-549: corrupt config.yaml → exception silenced, pool skipped."""
        d = tmp_path / "bad_pool"
        d.mkdir()
        (d / "config.yaml").write_text("{invalid: yaml: content")
        result = _match_display_name("bad pool", tmp_path, ["bad_pool"])
        assert result == []


class TestLoadClientDisplayNameFallback:
    """Cover lines 613-634: display-name fuzzy match in load_client error path."""

    def _make_pool(self, tmp_path, slug, display_name):
        import yaml
        d = tmp_path / slug
        d.mkdir()
        (d / "config.yaml").write_text(yaml.dump({"machine_name": slug, "name": display_name}))
        return slug

    def test_single_display_match_resolves_and_loads(self, tmp_path):
        """Lines 613-619: one display-name match → recursive load_client with slug."""
        self._make_pool(tmp_path, "my_pool", "My Pool")
        client = load_client("My Pool", clients_dir=tmp_path)
        assert client.machine_name == "my_pool"

    def test_ambiguous_display_match_raises(self, tmp_path):
        """Line 621: multiple display-name matches → ValueError with 'ambiguous'."""
        import yaml
        for slug, name in [("pool_alpha", "Project Pool Alpha"), ("pool_beta", "Project Pool Beta")]:
            d = tmp_path / slug
            d.mkdir()
            (d / "config.yaml").write_text(yaml.dump({"machine_name": slug, "name": name}))
        with pytest.raises(ValueError, match="ambiguous"):
            load_client("Project Pool", clients_dir=tmp_path)

    def test_error_message_lists_available_pools(self, tmp_path):
        """Line 629: error mentions available pool names."""
        self._make_pool(tmp_path, "real_pool", "Real Pool")
        with pytest.raises(ValueError, match="real_pool"):
            load_client("nonexistent", clients_dir=tmp_path)

    def test_error_message_suggests_close_slug(self, tmp_path):
        """Line 634: 'Did you mean' hint for close slug match."""
        self._make_pool(tmp_path, "my_pool", "My Pool")
        with pytest.raises(ValueError, match="Did you mean"):
            load_client("my_pol", clients_dir=tmp_path)

    def test_load_client_default_dir_uses_env(self, tmp_path):
        """Lines 594-595: clients_dir=None → get_pools_dir() called (lazy import)."""
        import yaml
        (tmp_path / "envpool").mkdir()
        (tmp_path / "envpool" / "config.yaml").write_text(
            yaml.dump({"machine_name": "envpool"})
        )
        with patch("mullet.core.env.get_pools_dir", return_value=tmp_path):
            client = load_client("envpool")
        assert client.machine_name == "envpool"

    def test_discover_clients_default_dir_uses_env(self, tmp_path):
        """Lines 668-669: clients_dir=None → get_pools_dir() called (lazy import)."""
        (tmp_path / "somepool").mkdir()
        with patch("mullet.core.env.get_pools_dir", return_value=tmp_path):
            result = discover_clients()
        assert "somepool" in result
