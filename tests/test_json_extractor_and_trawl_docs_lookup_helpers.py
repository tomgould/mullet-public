"""Tests for JSONExtractor pure helpers and TrawlDocsLookup constants/construction."""

from mullet.knowledge.code.extractors.json_extractor import (
    JSONExtractor,
    _strip_jsonc,
)


# ---------------------------------------------------------------------------
# _strip_jsonc
# ---------------------------------------------------------------------------

class TestStripJsonc:
    def test_plain_json_unchanged(self):
        raw = '{"key": "value"}'
        assert _strip_jsonc(raw) == raw

    def test_single_line_comment_removed(self):
        result = _strip_jsonc('// comment\n{"key": "val"}')
        assert "//" not in result
        assert '"key"' in result

    def test_inline_comment_removed(self):
        result = _strip_jsonc('{"key": "val"} // trailing')
        assert "trailing" not in result

    def test_block_comment_removed(self):
        result = _strip_jsonc('{"a": 1, /* block */ "b": 2}')
        assert "block" not in result
        assert '"a"' in result
        assert '"b"' in result

    def test_trailing_comma_removed(self):
        result = _strip_jsonc('{"a": 1,}')
        import json
        obj = json.loads(result)
        assert obj["a"] == 1

    def test_string_with_url_preserved(self):
        src = '{"url": "http://example.com"}'
        result = _strip_jsonc(src)
        assert "http://example.com" in result

    def test_string_with_comment_like_content_preserved(self):
        src = '{"msg": "// not a comment"}'
        result = _strip_jsonc(src)
        assert "// not a comment" in result

    def test_empty_string(self):
        assert _strip_jsonc("") == ""

    def test_returns_string(self):
        assert isinstance(_strip_jsonc('{"x": 1}'), str)


# ---------------------------------------------------------------------------
# JSONExtractor.extract — generic JSON
# ---------------------------------------------------------------------------

class TestJSONExtractorGeneric:
    def setup_method(self):
        self.ext = JSONExtractor()

    def test_returns_list(self):
        assert isinstance(self.ext.extract("{}", "test.json"), list)

    def test_empty_string_returns_single_chunk(self):
        result = self.ext.extract("", "test.json")
        assert len(result) == 1

    def test_language_is_json(self):
        result = self.ext.extract('{"x": 1}', "test.json")
        assert result[0].language == "json"

    def test_symbol_name_is_file_name(self):
        result = self.ext.extract('{"x": 1}', "config.json")
        assert result[0].symbol_name == "config.json"

    def test_source_file_preserved(self):
        result = self.ext.extract('{"x": 1}', "settings.json")
        assert result[0].source_file == "settings.json"

    def test_line_start_is_one(self):
        result = self.ext.extract('{"x": 1}', "test.json")
        assert result[0].line_start == 1

    def test_content_preserved(self):
        src = '{"key": "hello"}'
        result = self.ext.extract(src, "test.json")
        assert "key" in result[0].content

    def test_multiline_json_line_end_correct(self):
        src = '{\n  "a": 1,\n  "b": 2\n}'
        result = self.ext.extract(src, "test.json")
        assert result[0].line_end >= 3

    def test_invalid_json_still_returns_chunk(self):
        result = self.ext.extract("not valid json", "test.json")
        assert len(result) >= 1

    def test_symbol_type_document_for_generic(self):
        result = self.ext.extract('{"x": 1}', "generic.json")
        assert result[0].symbol_type == "document"


# ---------------------------------------------------------------------------
# JSONExtractor — package.json
# ---------------------------------------------------------------------------

class TestJSONExtractorPackageJson:
    def setup_method(self):
        self.ext = JSONExtractor()

    def _pkg(self, extra=None):
        pkg = {"name": "myapp", "version": "1.0.0", "dependencies": {"react": "^18"}}
        if extra:
            pkg.update(extra)
        import json
        return json.dumps(pkg)

    def test_project_chunk_present(self):
        result = self.ext.extract(self._pkg(), "package.json")
        types = [c.symbol_type for c in result]
        assert "project" in types

    def test_dependencies_chunk_present(self):
        result = self.ext.extract(self._pkg(), "package.json")
        names = [c.symbol_name for c in result]
        assert "dependencies" in names

    def test_project_name_is_symbol_name(self):
        result = self.ext.extract(self._pkg(), "package.json")
        project = next(c for c in result if c.symbol_type == "project")
        assert project.symbol_name == "myapp"

    def test_returns_list(self):
        result = self.ext.extract(self._pkg(), "package.json")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# JSONExtractor — composer.json
# ---------------------------------------------------------------------------

class TestJSONExtractorComposerJson:
    def setup_method(self):
        self.ext = JSONExtractor()

    def test_project_chunk_present(self):
        src = '{"name": "vendor/pkg", "require": {"php": ">=8.0"}}'
        result = self.ext.extract(src, "composer.json")
        types = [c.symbol_type for c in result]
        assert "project" in types

    def test_require_chunk_present(self):
        src = '{"name": "vendor/pkg", "require": {"php": ">=8.0"}}'
        result = self.ext.extract(src, "composer.json")
        names = [c.symbol_name for c in result]
        assert "require" in names or "dependencies" in names

    def test_vendor_package_name_in_symbol(self):
        src = '{"name": "vendor/pkg", "require": {}}'
        result = self.ext.extract(src, "composer.json")
        project = next((c for c in result if c.symbol_type == "project"), None)
        if project:
            assert "vendor" in project.symbol_name or "pkg" in project.symbol_name


# ---------------------------------------------------------------------------
# JSONExtractor — tsconfig.json
# ---------------------------------------------------------------------------

class TestJSONExtractorTsconfig:
    def setup_method(self):
        self.ext = JSONExtractor()

    def test_returns_chunks(self):
        src = '{"compilerOptions": {"target": "ES2020"}, "include": ["src"]}'
        result = self.ext.extract(src, "tsconfig.json")
        assert len(result) >= 1

    def test_config_type_or_document(self):
        src = '{"compilerOptions": {"target": "ES2020"}}'
        result = self.ext.extract(src, "tsconfig.json")
        types = {c.symbol_type for c in result}
        assert "config" in types or "document" in types


# ---------------------------------------------------------------------------
# TrawlDocsLookup — constants and construction only (no network/GPU calls)
# ---------------------------------------------------------------------------

class TestTrawlDocsLookupConstants:
    def test_default_sources_is_list(self):
        from mullet.core.tools.trawl_docs_lookup import DEFAULT_SOURCES
        assert isinstance(DEFAULT_SOURCES, list)

    def test_default_sources_nonempty(self):
        from mullet.core.tools.trawl_docs_lookup import DEFAULT_SOURCES
        assert len(DEFAULT_SOURCES) > 0

    def test_default_sources_contains_context7(self):
        from mullet.core.tools.trawl_docs_lookup import DEFAULT_SOURCES
        assert "context7" in DEFAULT_SOURCES

    def test_default_sources_contains_trawl(self):
        from mullet.core.tools.trawl_docs_lookup import DEFAULT_SOURCES
        assert "trawl" in DEFAULT_SOURCES

    def test_default_sources_contains_web(self):
        from mullet.core.tools.trawl_docs_lookup import DEFAULT_SOURCES
        assert "web" in DEFAULT_SOURCES


class TestTrawlDocsLookupConstruction:
    def test_instantiates_with_no_args(self):
        from mullet.core.tools.trawl_docs_lookup import TrawlDocsLookup
        lookup = TrawlDocsLookup(context7_client=None)
        assert lookup is not None

    def test_custom_source_preference_web_only(self):
        from mullet.core.tools.trawl_docs_lookup import TrawlDocsLookup
        lookup = TrawlDocsLookup(context7_client=None, source_preference=["web"])
        assert lookup.source_preference == ["web"]

    def test_custom_source_preference_list_of_one(self):
        from mullet.core.tools.trawl_docs_lookup import TrawlDocsLookup
        lookup = TrawlDocsLookup(context7_client=None, source_preference=["trawl"])
        assert "trawl" in lookup.source_preference

    def test_client_stored_as_none(self):
        from mullet.core.tools.trawl_docs_lookup import TrawlDocsLookup
        lookup = TrawlDocsLookup(context7_client=None)
        assert lookup._context7_client is None
