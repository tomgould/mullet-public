"""Tests for code indexing path configuration."""

from unittest.mock import patch, MagicMock
from mullet.knowledge.engine.client import DEFAULT_CONFIG, validate_config, Client
from mullet.knowledge.engine.ingest import ingest_client_code


class TestCodeConfigDefaults:
    """Test default code configuration."""

    def test_default_config_has_code_section(self):
        """Code section should exist in DEFAULT_CONFIG."""
        assert "code" in DEFAULT_CONFIG

    def test_default_code_config_structure(self):
        """Default code config should have expected keys."""
        code_cfg = DEFAULT_CONFIG["code"]
        assert code_cfg["enabled"] is False
        assert code_cfg["paths"] == []
        assert isinstance(code_cfg["languages"], dict)
        assert code_cfg["parallel_workers"] == 4
        assert code_cfg["max_file_size_kb"] == 500
        assert isinstance(code_cfg["git"], dict)
        assert code_cfg["git"]["detect_repos"] is True
        assert code_cfg["git"]["follow_symlinks"] is True


class TestCodeConfigValidation:
    """Test validation of code configuration."""

    def test_valid_code_config_empty(self):
        """Empty code config should be valid."""
        config = {
            "machine_name": "test",
            "code": {},
        }
        warnings = validate_config(config)
        code_warnings = [w for w in warnings if "code" in w.lower()]
        assert len(code_warnings) == 0

    def test_valid_code_config_with_paths(self):
        """Code config with valid paths should be valid."""
        config = {
            "machine_name": "test",
            "code": {
                "enabled": True,
                "paths": ["src", "lib", "/absolute/path"],
            },
        }
        warnings = validate_config(config)
        code_warnings = [w for w in warnings if "code" in w.lower()]
        assert len(code_warnings) == 0

    def test_invalid_code_enabled_type(self):
        """code.enabled must be boolean."""
        config = {
            "machine_name": "test",
            "code": {"enabled": "yes"},
        }
        warnings = validate_config(config)
        assert any("code.enabled" in w and "boolean" in w for w in warnings)

    def test_invalid_code_paths_type(self):
        """code.paths must be a list."""
        config = {
            "machine_name": "test",
            "code": {"paths": "src"},
        }
        warnings = validate_config(config)
        assert any("code.paths" in w and "list" in w for w in warnings)

    def test_invalid_code_paths_element(self):
        """code.paths elements must be strings."""
        config = {
            "machine_name": "test",
            "code": {"paths": ["src", 123]},
        }
        warnings = validate_config(config)
        assert any("code.paths[1]" in w and "string" in w for w in warnings)

    def test_invalid_parallel_workers(self):
        """code.parallel_workers must be positive integer."""
        config1 = {
            "machine_name": "test",
            "code": {"parallel_workers": 0},
        }
        config2 = {
            "machine_name": "test",
            "code": {"parallel_workers": "4"},
        }
        warnings1 = validate_config(config1)
        warnings2 = validate_config(config2)
        assert any("parallel_workers" in w for w in warnings1)
        assert any("parallel_workers" in w for w in warnings2)

    def test_invalid_max_file_size(self):
        """code.max_file_size_kb must be positive."""
        config = {
            "machine_name": "test",
            "code": {"max_file_size_kb": -1},
        }
        warnings = validate_config(config)
        assert any("max_file_size_kb" in w for w in warnings)

    def test_unknown_code_key(self):
        """Unknown code keys should generate warning."""
        config = {
            "machine_name": "test",
            "code": {"unknown_key": "value"},
        }
        warnings = validate_config(config)
        assert any("Unknown code key" in w for w in warnings)

    def test_unknown_git_key(self):
        """Unknown code.git keys should generate warning."""
        config = {
            "machine_name": "test",
            "code": {
                "git": {"unknown_key": True}
            },
        }
        warnings = validate_config(config)
        assert any("Unknown code.git key" in w for w in warnings)


class TestClientCodeConfig:
    """Test Client code_config property."""

    def test_client_code_config_default(self, tmp_path):
        """Client without code config should get empty dict."""
        config_file = tmp_path / "client.yaml"
        config_file.write_text("machine_name: test\n")
        
        client = Client(tmp_path, {"machine_name": "test"})
        assert client.code_config == {}

    def test_client_code_config_with_enabled(self, tmp_path):
        """Client with enabled code config."""
        config = {
            "machine_name": "test",
            "code": {
                "enabled": True,
                "paths": ["src/"],
            },
        }
        client = Client(tmp_path, config)
        assert client.code_config["enabled"] is True
        assert client.code_config["paths"] == ["src/"]

    def test_client_get_code_chunks(self, tmp_path):
        """Client.get_code_chunks returns chunks."""
        config = {
            "machine_name": "test",
            "code": {
                "enabled": True,
                "paths": [],
                "languages": {},
            },
        }
        client = Client(tmp_path, config)
        # With empty paths, should return empty list
        chunks = client.get_code_chunks([tmp_path])
        assert isinstance(chunks, list)

    def test_default_code_languages(self):
        """Default code config should have language definitions."""
        assert "languages" in DEFAULT_CONFIG["code"]
        assert "python" in DEFAULT_CONFIG["code"]["languages"]


class TestSearchConfigValidation:
    """Validate search config sub-keys."""

    def _base(self):
        return {"machine_name": "test-pool"}

    def test_valid_search_config_no_warnings(self):
        cfg = {**self._base(), "search": {"bm25_weight": 0.5, "vector_weight": 0.5, "top_k": 10}}
        warnings = validate_config(cfg)
        assert not any("search" in w for w in warnings)

    def test_bm25_weight_out_of_range(self):
        cfg = {**self._base(), "search": {"bm25_weight": 1.5}}
        warnings = validate_config(cfg)
        assert any("bm25_weight" in w for w in warnings)

    def test_vector_weight_out_of_range(self):
        cfg = {**self._base(), "search": {"vector_weight": -0.1}}
        warnings = validate_config(cfg)
        assert any("vector_weight" in w for w in warnings)

    def test_top_k_non_positive(self):
        cfg = {**self._base(), "search": {"top_k": 0}}
        warnings = validate_config(cfg)
        assert any("top_k" in w for w in warnings)

    def test_invalid_merge_strategy(self):
        cfg = {**self._base(), "search": {"merge_strategy": "unknown"}}
        warnings = validate_config(cfg)
        assert any("merge_strategy" in w for w in warnings)

    def test_valid_merge_strategy_no_warning(self):
        for strategy in ("weighted", "rrf"):
            cfg = {**self._base(), "search": {"merge_strategy": strategy}}
            warnings = validate_config(cfg)
            assert not any("merge_strategy" in w for w in warnings)

    def test_invalid_reranker(self):
        cfg = {**self._base(), "search": {"reranker": "fancy-model"}}
        warnings = validate_config(cfg)
        assert any("reranker" in w for w in warnings)

    def test_mmr_lambda_out_of_range(self):
        cfg = {**self._base(), "search": {"mmr_lambda": 2.0}}
        warnings = validate_config(cfg)
        assert any("mmr_lambda" in w for w in warnings)

    def test_unknown_search_key_warned(self):
        cfg = {**self._base(), "search": {"totally_unknown_key": True}}
        warnings = validate_config(cfg)
        assert any("totally_unknown_key" in w for w in warnings)

    def test_missing_machine_name_warned(self):
        warnings = validate_config({})
        assert any("machine_name" in w for w in warnings)

    def test_short_machine_name_warned(self):
        warnings = validate_config({"machine_name": "ab"})
        assert any("too short" in w for w in warnings)


class TestIngestClientCodePathsFallback:
    """Regression tests for --rebuild-code with no explicit code.paths."""

    def test_fallback_to_sources_when_documents_missing(self, tmp_path):
        """When code.paths is not set and documents/ is missing, sources/ is scanned."""
        pool_dir = tmp_path / "test-pool"
        pool_dir.mkdir()

        # Set up sources with a Python file
        sources_dir = pool_dir / "sources" / "github" / "owner" / "repo"
        sources_dir.mkdir(parents=True)
        (sources_dir / "test.py").write_text("def foo(): pass")

        # Do NOT create documents/
        config = {
            "machine_name": "test-pool",
            "code": {"enabled": True},
        }
        client = Client(pool_dir, config)
        assert client.has_sources is True

        with patch("mullet.knowledge.engine.ingest.ingest_code") as mock_ingest:
            mock_ingest.return_value = ([{"id": "1"}], [[0.0] * 384])
            ingest_client_code(client, ingestor=MagicMock(), rebuild=True)

            assert mock_ingest.called, "ingest_code should be called when sources/ has code"
            code_paths = mock_ingest.call_args.kwargs["code_paths"]
            path_strs = [str(p) for p in code_paths]
            assert any("sources" in p for p in path_strs), (
                f"Expected sources/ in code_paths, got {path_strs}"
            )

    def test_fallback_includes_both_documents_and_sources(self, tmp_path):
        """When code.paths is not set and both dirs exist, both are scanned."""
        pool_dir = tmp_path / "test-pool"
        pool_dir.mkdir()
        (pool_dir / "documents").mkdir()
        (pool_dir / "sources" / "github" / "owner" / "repo").mkdir(parents=True)
        (pool_dir / "sources" / "github" / "owner" / "repo" / "test.py").write_text("def bar(): pass")

        config = {
            "machine_name": "test-pool",
            "code": {"enabled": True},
        }
        client = Client(pool_dir, config)

        with patch("mullet.knowledge.engine.ingest.ingest_code") as mock_ingest:
            mock_ingest.return_value = ([{"id": "1"}], [[0.0] * 384])
            ingest_client_code(client, ingestor=MagicMock(), rebuild=True)

            code_paths = mock_ingest.call_args.kwargs["code_paths"]
            path_strs = {str(p) for p in code_paths}
            assert str(pool_dir / "documents") in path_strs
            assert str(pool_dir / "sources") in path_strs

    def test_explicit_paths_override_fallback(self, tmp_path):
        """When code.paths is explicitly set, sources/ fallback is not used."""
        pool_dir = tmp_path / "test-pool"
        pool_dir.mkdir()
        (pool_dir / "custom").mkdir()
        (pool_dir / "sources" / "github" / "owner" / "repo").mkdir(parents=True)

        config = {
            "machine_name": "test-pool",
            "code": {
                "enabled": True,
                "paths": ["custom"],
            },
        }
        client = Client(pool_dir, config)

        with patch("mullet.knowledge.engine.ingest.ingest_code") as mock_ingest:
            mock_ingest.return_value = ([{"id": "1"}], [[0.0] * 384])
            ingest_client_code(client, ingestor=MagicMock(), rebuild=True)

            code_paths = mock_ingest.call_args.kwargs["code_paths"]
            path_strs = [str(p) for p in code_paths]
            assert any("custom" in p for p in path_strs)
            assert not any("sources" in p for p in path_strs), (
                f"Expected no sources/ in explicit-path mode, got {path_strs}"
            )

    def test_rebuild_true_ingests_files_with_no_explicit_paths(self, tmp_path):
        """rebuild=True with code.enabled but no paths still ingests source files."""
        pool_dir = tmp_path / "test-pool"
        pool_dir.mkdir()

        # Simulate a source-based pool: no documents, but sources has code
        sources_dir = pool_dir / "sources" / "github" / "owner" / "repo"
        sources_dir.mkdir(parents=True)
        (sources_dir / "module.py").write_text("def hello():\n    return 'world'\n")

        config = {
            "machine_name": "test-pool",
            "code": {"enabled": True},
        }
        client = Client(pool_dir, config)

        with patch("mullet.knowledge.engine.ingest.ingest_code") as mock_ingest:
            mock_ingest.return_value = ([{"id": "chunk-1"}], [[0.0] * 384])
            chunks, embeddings = ingest_client_code(
                client, ingestor=MagicMock(), rebuild=True
            )

            # Because ingest_code is mocked, we get the mocked return value
            assert len(chunks) == 1
            # The key assertion: ingest_code was called at all (not early-returned)
            assert mock_ingest.called


class TestDocumentFilenames:
    """Tests for Client.get_document_filenames() and DEFAULT_DOCUMENT_FILENAMES.

    documents.filenames lets pools opt-in to ingesting extension-less files
    (Jenkinsfile, Dockerfile, Makefile). Default is empty — nothing is added
    implicitly.
    """

    def test_default_when_config_missing_filenames_key(self, tmp_path):
        client = Client(tmp_path, {"machine_name": "test"})
        assert client.get_document_filenames() == ()

    def test_default_when_documents_section_present_but_no_filenames(self, tmp_path):
        client = Client(tmp_path, {
            "machine_name": "test",
            "documents": {"extensions": [".md"]},
        })
        assert client.get_document_filenames() == ()

    def test_empty_list_returns_empty_tuple(self, tmp_path):
        client = Client(tmp_path, {
            "machine_name": "test",
            "documents": {"filenames": []},
        })
        assert client.get_document_filenames() == ()

    def test_filenames_returned_sorted(self, tmp_path):
        client = Client(tmp_path, {
            "machine_name": "test",
            "documents": {"filenames": ["Makefile", "Jenkinsfile", "Dockerfile"]},
        })
        assert client.get_document_filenames() == (
            "Dockerfile", "Jenkinsfile", "Makefile",
        )

    def test_duplicates_deduped(self, tmp_path):
        client = Client(tmp_path, {
            "machine_name": "test",
            "documents": {"filenames": ["Jenkinsfile", "Jenkinsfile", "Dockerfile"]},
        })
        assert client.get_document_filenames() == ("Dockerfile", "Jenkinsfile")

    def test_returns_tuple_not_list(self, tmp_path):
        """get_document_filenames must return a tuple — callers may rely on
        it being hashable / immutable."""
        client = Client(tmp_path, {
            "machine_name": "test",
            "documents": {"filenames": ["Jenkinsfile"]},
        })
        result = client.get_document_filenames()
        assert isinstance(result, tuple)

    def test_default_constant_is_empty_tuple(self):
        from mullet.knowledge.engine.client import DEFAULT_DOCUMENT_FILENAMES
        assert DEFAULT_DOCUMENT_FILENAMES == ()
        assert isinstance(DEFAULT_DOCUMENT_FILENAMES, tuple)
