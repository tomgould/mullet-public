"""Tests for CodeChunker config-derived pure methods."""


from mullet.knowledge.code.chunker import CodeChunker


# ---------------------------------------------------------------------------
# get_extensions_for_languages
# ---------------------------------------------------------------------------

class TestGetExtensionsForLanguages:
    def test_empty_config_falls_back_to_supported(self):
        chunker = CodeChunker({})
        exts = chunker.get_extensions_for_languages()
        assert len(exts) > 0
        assert all(e.startswith(".") for e in exts)

    def test_configured_extensions_returned(self):
        config = {
            "languages": {
                "python": {"extensions": [".py"]},
                "javascript": {"extensions": [".js"]},
            }
        }
        chunker = CodeChunker(config)
        exts = chunker.get_extensions_for_languages()
        assert ".py" in exts
        assert ".js" in exts

    def test_dot_prefix_added_if_missing(self):
        config = {
            "languages": {
                "python": {"extensions": ["py"]},  # No leading dot
            }
        }
        chunker = CodeChunker(config)
        exts = chunker.get_extensions_for_languages()
        assert ".py" in exts
        assert "py" not in exts

    def test_extensions_lowercased(self):
        config = {
            "languages": {
                "python": {"extensions": [".PY", ".Py"]},
            }
        }
        chunker = CodeChunker(config)
        exts = chunker.get_extensions_for_languages()
        assert ".py" in exts
        assert ".PY" not in exts

    def test_returns_set(self):
        chunker = CodeChunker({})
        assert isinstance(chunker.get_extensions_for_languages(), set)

    def test_lang_config_without_extensions_ignored(self):
        config = {
            "languages": {
                "python": {"max_tokens": 100},  # No extensions key
            }
        }
        # Falls back to supported extensions since no extensions configured
        chunker = CodeChunker(config)
        exts = chunker.get_extensions_for_languages()
        assert len(exts) > 0

    def test_multiple_extensions_per_language(self):
        config = {
            "languages": {
                "web": {"extensions": [".html", ".css", ".js"]},
            }
        }
        chunker = CodeChunker(config)
        exts = chunker.get_extensions_for_languages()
        assert ".html" in exts
        assert ".css" in exts
        assert ".js" in exts


# ---------------------------------------------------------------------------
# get_exclude_patterns
# ---------------------------------------------------------------------------

class TestGetExcludePatterns:
    def test_empty_config_returns_empty(self):
        chunker = CodeChunker({})
        assert chunker.get_exclude_patterns() == []

    def test_patterns_from_language_config(self):
        config = {
            "languages": {
                "python": {
                    "extensions": [".py"],
                    "exclude": ["*_test.py", "conftest.py"],
                }
            }
        }
        chunker = CodeChunker(config)
        patterns = chunker.get_exclude_patterns()
        assert "*_test.py" in patterns
        assert "conftest.py" in patterns

    def test_patterns_from_multiple_languages(self):
        config = {
            "languages": {
                "python": {"extensions": [".py"], "exclude": ["tests/*"]},
                "javascript": {"extensions": [".js"], "exclude": ["node_modules/*"]},
            }
        }
        chunker = CodeChunker(config)
        patterns = chunker.get_exclude_patterns()
        assert "tests/*" in patterns
        assert "node_modules/*" in patterns

    def test_language_without_exclude_skipped(self):
        config = {
            "languages": {
                "python": {"extensions": [".py"]},  # No exclude
            }
        }
        chunker = CodeChunker(config)
        assert chunker.get_exclude_patterns() == []

    def test_returns_list(self):
        chunker = CodeChunker({})
        assert isinstance(chunker.get_exclude_patterns(), list)

    def test_non_dict_lang_config_skipped(self):
        config = {
            "languages": {
                "python": True,  # Invalid config type
            }
        }
        chunker = CodeChunker(config)
        assert chunker.get_exclude_patterns() == []


# ---------------------------------------------------------------------------
# collect_files — filesystem integration
# ---------------------------------------------------------------------------

class TestCollectFiles:
    def test_nonexistent_path_skipped(self, tmp_path):
        chunker = CodeChunker({})
        files = chunker.collect_files([tmp_path / "nonexistent"])
        assert files == []

    def test_empty_directory_returns_empty(self, tmp_path):
        chunker = CodeChunker({})
        files = chunker.collect_files([tmp_path])
        assert isinstance(files, list)

    def test_returns_list(self, tmp_path):
        chunker = CodeChunker({})
        result = chunker.collect_files([tmp_path])
        assert isinstance(result, list)

    def test_collects_py_files(self, tmp_path):
        (tmp_path / "main.py").write_text("# code")
        config = {"languages": {"python": {"extensions": [".py"]}}}
        chunker = CodeChunker(config)
        files = chunker.collect_files([tmp_path])
        assert any(f.name == "main.py" for f in files)

    def test_excludes_non_configured_extensions(self, tmp_path):
        (tmp_path / "style.css").write_text("body {}")
        config = {"languages": {"python": {"extensions": [".py"]}}}
        chunker = CodeChunker(config)
        files = chunker.collect_files([tmp_path])
        assert not any(f.name == "style.css" for f in files)
